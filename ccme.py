"""Conditional Counterfactual Density Estimator (CCDEstimator).

Implements a two-stage estimation pipeline for counterfactual conditional
densities P(Y^1 | V) using kernel mean embeddings in reproducing kernel
Hilbert spaces.  Supports three estimation methods (neural-kernel, deep
feature, kernel ridge regression), four correction modes (doubly robust,
IPW, plug-in, one-step), and optional K-fold cross-fitting.

Typical usage::

    from ccme import CCDEstimator
    from omegaconf import OmegaConf

    cfg = OmegaConf.load("configs/simulation/method/nk.yaml")
    estimator = CCDEstimator(cfg)
    estimator.fit(X=X, V=V, Y=Y, A=A)
    pdf_est = estimator.predict(V_eval=V_test, Y_grid=Y_grid)
"""

import warnings
from dataclasses import dataclass
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax
from hydra.utils import instantiate
from tqdm import tqdm

import losses
import inference
from models import RidgeModel, DeepFeatureNet, NeuralKernelNet
from utils import get_median_heuristic, train_nuisance_models


@dataclass
class FoldResult:
    """Stores the trained models and data for a single cross-fitting fold."""

    model_or: Any
    state_or: Any
    model: Any  # None for onestep
    state: Any  # None for onestep
    pi_model: Any  # None for onestep
    X_aux: Any
    Y_aux: Any
    V_target: Any
    X_target: Any
    Y_target: Any
    A_target: Any
    pi_target: Any  # None for onestep
    eval_fn: Any


@eqx.filter_jit
def train_step_target(
    model,
    state,
    optim,
    opt_state,
    V,
    X,
    Y,
    A,
    model_or,
    state_or,
    pi,
    loss_fn,
    X_ref,
    Y_ref,
):
    """Target Model Training Step (Stage 2). Requires V and X."""

    def loss_wrapper(m, s):
        # NOTE: loss_nk_dr/loss_df_dr now take (model, state, V, X, ...)
        return loss_fn(m, s, V, X, Y, A, model_or, state_or, pi, X_ref, Y_ref)

    (loss, state), grads = eqx.filter_value_and_grad(loss_wrapper, has_aux=True)(
        model, state
    )
    params = eqx.filter(model, eqx.is_inexact_array)
    updates, opt_state = optim.update(grads, opt_state, params)
    model = eqx.apply_updates(model, updates)
    return model, state, opt_state, loss


@eqx.filter_jit
def train_step_nuisance(model, state, optim, opt_state, X, Y, loss_fn):
    """Nuisance Model Training Step (Stage 1). Requires Full X."""

    def loss_wrapper(m, s):
        # Stage 1 losses just take X
        # Pass dummy values for unused args (pi, num_inputs etc not used in Stage 1 basic loss)
        return loss_fn(m, s, X, Y)

    (loss, state), grads = eqx.filter_value_and_grad(loss_wrapper, has_aux=True)(
        model, state
    )
    params = eqx.filter(model, eqx.is_inexact_array)
    updates, opt_state = optim.update(grads, opt_state, params)
    model = eqx.apply_updates(model, updates)
    return model, state, opt_state, loss


def get_indices(cfg_obj, key, default_max):
    """Extract an optional index array from a config object.

    Args:
        cfg_obj: OmegaConf config node (e.g. ``cfg.model``).
        key: Attribute name to look up (e.g. ``"idx_pi"``).
        default_max: Unused (kept for API compatibility).

    Returns:
        Integer numpy array of indices, or ``None`` if the attribute is absent
        or ``None``.
    """
    idx = getattr(cfg_obj, key, None)
    if idx is None:
        return None
    return np.array(idx, dtype=int)


def _zero_lamb(model):
    """Return a copy of the model with lamb set to zero (for unregularized validation).

    Models without a ``lamb`` attribute (e.g. NeuralKernelNet) are returned
    unchanged.
    """
    if hasattr(model, "lamb"):
        return eqx.tree_at(lambda m: m.lamb, model, 0.0)
    return model


def get_components(method_name, mode):
    """Look up the model class, loss functions, and eval function for a method/mode pair.

    Args:
        method_name: One of ``"nk"``, ``"df"``, or ``"ridge"``.
        mode: One of ``"dr"``, ``"ipw"``, ``"pi"``, or ``"onestep"``.

    Returns:
        Tuple of ``(model_cls, loss_nuis, loss_target, eval_fn)``.

    Raises:
        ValueError: If ``method_name`` is not recognized.
    """
    loss_target = None
    if method_name == "nk":
        model_cls = NeuralKernelNet
        loss_nuis = losses.loss_nk
        eval_fn = inference.estimate_nk
        if mode == "dr":
            loss_target = losses.loss_nk_dr
        elif mode == "ipw":
            loss_target = losses.loss_nk_ipw
        elif mode == "pi":
            loss_target = losses.loss_nk_pi
        elif mode == "onestep":
            loss_target = losses.loss_nk
    elif method_name == "df":
        model_cls = DeepFeatureNet
        loss_nuis = losses.loss_df
        eval_fn = inference.estimate_df
        if mode == "dr":
            loss_target = losses.loss_df_dr
        elif mode == "ipw":
            loss_target = losses.loss_df_ipw
        elif mode == "pi":
            loss_target = losses.loss_df_pi
        elif mode == "onestep":
            loss_target = losses.loss_df
            eval_fn = inference.estimate_df_onestep
    elif method_name == "ridge":
        model_cls = RidgeModel
        loss_nuis, loss_target = None, None
        eval_fn = inference.estimate_ridge
        if mode == "onestep":
            eval_fn = inference.estimate_ridge_onestep
    else:
        raise ValueError(f"Unknown method {method_name}")
    return model_cls, loss_nuis, loss_target, eval_fn


class CCDEstimator:
    def __init__(self, cfg):
        """Initialize the Conditional Counterfactual Density Estimator.

        Args:
            cfg (DictConfig): Hydra/OmegaConf configuration with the following
                top-level and nested keys:

                **Top-level parameters:**

                method (str):
                    Estimation method. One of:

                    - ``"nk"`` -- Neural-Kernel (grid-based coefficients).
                    - ``"df"`` -- Deep Feature (learned feature map).
                    - ``"ridge"`` -- Kernel ridge regression (no neural network).

                mode (str):
                    Correction mode. One of:

                    - ``"dr"`` -- Doubly robust (IPW + plug-in).
                    - ``"ipw"`` -- Inverse probability weighting only.
                    - ``"pi"`` -- Plug-in (nuisance model) only.
                    - ``"onestep"`` -- One-step pseudo-outcome (no sample splitting,
                      trains on treated data only).

                seed (int):
                    Random seed for data shuffling and model initialization.

                verbose (bool):
                    If True, display progress bars and status messages.

                n_folds (int or None):
                    Number of cross-fitting folds. ``None`` (default) uses a
                    single 50-50 sample split.  Integer >= 2 enables K-fold
                    cross-fitting where each fold serves as the target set once.
                    Ignored when ``mode="onestep"``.

                **cfg.model parameters:**

                output_dim (int):
                    Number of grid/inducing points for ``"nk"`` or output feature
                    dimension for ``"df"``.  For ``"ridge"``, controls the grid
                    resolution used to initialize inducing points.

                hidden_dim (list[int]):
                    Hidden layer widths for the neural network, e.g. ``[64, 64]``.
                    Ignored for ``"ridge"``.

                sigma_init (float or str):
                    Initial kernel bandwidth.  Can be a float or ``"median"`` to
                    use the median heuristic computed from the outcome data.

                lamb (float or None):
                    Ridge regularization parameter.  Required for ``"df"`` and
                    ``"ridge"``.  Ignored for ``"nk"`` (set to ``None``).

                learn_sigma (bool):
                    If True, the kernel bandwidth is learned during Stage 1
                    (nuisance) training.  Only applicable to ``"nk"``.

                idx_pi (list[int] or None):
                    Column indices of ``X`` to use for propensity score
                    estimation.  ``None`` uses all columns.

                idx_or (list[int] or None):
                    Column indices of ``X`` to use for the outcome (nuisance)
                    model.  ``None`` uses all columns.  In ``"onestep"`` mode,
                    this is overridden to use ``V`` instead.

                **cfg.propensity_score_model parameters:**

                _target_ (str):
                    Hydra instantiation target for the propensity score
                    classifier, e.g.
                    ``"sklearn.ensemble.RandomForestClassifier"``.

                Additional keyword arguments are passed to the classifier
                constructor (e.g. ``max_depth: 4``).

                **cfg.train parameters:**

                lr_or (float):
                    Learning rate for Stage 1 (nuisance model) training.

                lr_fi (float):
                    Learning rate for Stage 2 (target model) training.

                batch_size_or (int):
                    Mini-batch size for Stage 1 training.

                batch_size_fi (int):
                    Mini-batch size for Stage 2 training.

                epoch_or (int):
                    Maximum number of training epochs for Stage 1.

                epoch_fi (int):
                    Maximum number of training epochs for Stage 2.

                valid_size_or (float, int, or None):
                    Validation set size for Stage 1.  ``None`` disables
                    validation.  A float in (0, 1) is treated as a fraction;
                    an integer specifies the absolute number of samples.

                valid_size_fi (float, int, or None):
                    Validation set size for Stage 2 (same semantics as above).

                patience_or (int or None):
                    Early stopping patience for Stage 1.  Training stops if the
                    validation loss does not improve for this many epochs.
                    ``None`` disables early stopping.

                patience_fi (int or None):
                    Early stopping patience for Stage 2.

                **cfg.test parameters:**

                num_bin (int):
                    Number of grid points for outcome density evaluation
                    (used by experiment scripts, not by the estimator itself).
        """
        self.cfg = cfg
        self.method = cfg.method
        self.mode = cfg.mode
        self.seed = cfg.seed
        self.verbose = cfg.verbose
        self.n_folds = getattr(cfg, "n_folds", None)
        self.is_fitted = False

        if self.n_folds is not None and self.n_folds < 2:
            raise ValueError(f"n_folds must be None or >= 2, got {self.n_folds}")

        master_key = jax.random.key(self.seed)
        keys = jax.random.split(master_key, 5)
        self.k_data_shuffle = keys[0]
        self.k_nuis_init = keys[1]
        self.k_nuis_loop = keys[2]
        self.k_target_init = keys[3]
        self.k_target_loop = keys[4]

        self.pi_model = None
        self.model_or = None
        self.state_or = None
        self.model = None
        self.state = None
        self.eval_fn = None

        self.X_aux = None
        self.Y_aux = None
        self.V_target = None
        self.X_target = None
        self.Y_target = None
        self.A_target = None
        self.pi_target = None

        self._fold_results = []

    def _fit_single_fold(
        self,
        fold_idx,
        n_folds_total,
        X0_prop,
        X0_or,
        Y0,
        A0,
        X1_prop,
        X1_or,
        V1,
        Y1,
        A1,
        ModelClass,
        loss_fn_nuis,
        loss_fn_target,
        eval_fn,
        dim_X_or,
        dim_V,
        ypcl,
        sigma_init,
        k_nuis_init,
        k_nuis_loop,
        k_target_init,
        k_target_loop,
    ):
        """Train nuisance + propensity + target models for a single data fold.

        This is the core training loop extracted to support both single-split
        and K-fold cross-fitting.  It performs:

        1. Stage 1: Train the nuisance model on treated observations from the
           nuisance split (``X0``, ``Y0`` where ``A0 == 1``).
        2. Fit a propensity score model on the nuisance split (skipped for
           ``"onestep"`` mode).
        3. Stage 2: Train the target model on the target split using the DR/IPW/PI
           loss (skipped for ``"onestep"`` and ``"ridge"``).

        Args:
            fold_idx: Fold index (for progress bar labeling).
            n_folds_total: Total number of folds (1 for single-split).
            X0_prop: Nuisance-split covariates for propensity model.
            X0_or: Nuisance-split covariates for outcome model.
            Y0: Nuisance-split outcomes.
            A0: Nuisance-split treatment indicators.
            X1_prop: Target-split covariates for propensity prediction.
            X1_or: Target-split covariates for outcome model input.
            V1: Target-split conditioning variables.
            Y1: Target-split outcomes.
            A1: Target-split treatment indicators.
            ModelClass: Model class (NeuralKernelNet, DeepFeatureNet, or RidgeModel).
            loss_fn_nuis: Stage 1 loss function.
            loss_fn_target: Stage 2 loss function.
            eval_fn: Inference function for density prediction.
            dim_X_or: Dimensionality of nuisance covariates.
            dim_V: Dimensionality of conditioning variables.
            ypcl: Grid/inducing points for the kernel.
            sigma_init: Initial kernel bandwidth.
            k_nuis_init: JAX key for nuisance model initialization.
            k_nuis_loop: JAX key for nuisance training loop shuffling.
            k_target_init: JAX key for target model initialization.
            k_target_loop: JAX key for target training loop shuffling.

        Returns:
            A :class:`FoldResult` containing the trained models and associated
            split data.
        """
        X0_ref_or = X0_or[A0 == 1]
        Y0_ref = Y0[A0 == 1]

        fold_label = (
            f"Fold {fold_idx + 1}/{n_folds_total} " if n_folds_total > 1 else ""
        )

        # ----------------------------------------
        # Training First-Stage (Nuisance) Model
        # ----------------------------------------
        model_or = ModelClass(
            k_nuis_init,
            dim_X_or,
            self.cfg.model.hidden_dim,
            self.cfg.model.output_dim,
            ypcl,
            sigma_init,
            self.cfg.model.lamb,
            learn_sig=self.cfg.model.learn_sigma,
        )
        state_or = eqx.nn.State(model_or)

        if self.method != "ridge":
            optim_or = optax.sgd(self.cfg.train.lr_or, momentum=0.9)
            opt_state_or = optim_or.init(
                eqx.filter(model_or, eqx.is_inexact_array)
            )

            valid_size_or = getattr(self.cfg.train, "valid_size_or", None)
            if valid_size_or is not None and valid_size_or > 0:
                if valid_size_or < 1.0:
                    n_valid_or = int(X0_ref_or.shape[0] * valid_size_or)
                else:
                    n_valid_or = int(valid_size_or)

                n_train_or = X0_ref_or.shape[0] - n_valid_or
                X_train_or, X_valid_or = (
                    X0_ref_or[:n_train_or],
                    X0_ref_or[n_train_or:],
                )
                Y_train_or, Y_valid_or = Y0_ref[:n_train_or], Y0_ref[n_train_or:]
            else:
                X_train_or, Y_train_or = X0_ref_or, Y0_ref
                X_valid_or, Y_valid_or = None, None

            batch_size = self.cfg.train.batch_size_or
            n_batches = int(np.ceil(X_train_or.shape[0] / batch_size))

            patience_or = getattr(self.cfg.train, "patience_or", None)
            best_val_loss_or = float("inf")
            epochs_without_improvement_or = 0
            best_model_or = None
            best_state_or = None

            if n_batches > 0:
                pbar = tqdm(
                    range(self.cfg.train.epoch_or),
                    desc=f"{fold_label}Stage 1",
                    position=0,
                    disable=not self.verbose,
                    bar_format="{desc}: |{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]{postfix}",
                )
                for epoch in pbar:
                    k_nuis_loop, ksub = jax.random.split(k_nuis_loop)
                    perm_nuis = jax.random.permutation(ksub, X_train_or.shape[0])
                    X_sh, Y_sh = X_train_or[perm_nuis], Y_train_or[perm_nuis]

                    epoch_loss = 0.0
                    for i in range(n_batches):
                        bs = i * batch_size
                        be = min(bs + batch_size, X_train_or.shape[0])

                        model_or, state_or, opt_state_or, loss = (
                            train_step_nuisance(
                                model_or,
                                state_or,
                                optim_or,
                                opt_state_or,
                                X_sh[bs:be],
                                Y_sh[bs:be],
                                loss_fn_nuis,
                            )
                        )
                        epoch_loss += loss.item()

                    postfix_dict = {"train_loss": f"{epoch_loss / n_batches:.5f}"}

                    if X_valid_or is not None:
                        val_loss, _ = loss_fn_nuis(
                            _zero_lamb(model_or), state_or, X_valid_or, Y_valid_or
                        )
                        val_loss_value = val_loss.item()
                        postfix_dict["val_loss"] = f"{val_loss_value:.5f}"

                        if patience_or is not None and patience_or > 0:
                            if val_loss_value < best_val_loss_or:
                                best_val_loss_or = val_loss_value
                                epochs_without_improvement_or = 0
                                best_model_or = model_or
                                best_state_or = state_or
                                postfix_dict["patience"] = (
                                    f"0000/{patience_or:04}"
                                )
                            else:
                                epochs_without_improvement_or += 1
                                postfix_dict["patience"] = (
                                    f"{epochs_without_improvement_or:04}/{patience_or:04}"
                                )

                            if epochs_without_improvement_or >= patience_or:
                                pbar.close()
                                if self.verbose:
                                    tqdm.write(
                                        f"{fold_label}Stage 1: Early stopping at epoch {epoch + 1}, "
                                        f"restoring best model (val_loss={best_val_loss_or:.5f})"
                                    )
                                if best_model_or is not None:
                                    model_or = best_model_or
                                    state_or = best_state_or
                                break

                    pbar.set_postfix(postfix_dict)

        model_or = eqx.nn.inference_mode(model_or)

        # --------------------------------------------------
        # Training Propensity Scores and Second Stage Models
        # --------------------------------------------------
        model = None
        state = None
        pi_model = None
        pi_target = None

        if self.mode != "onestep":
            pi_model = instantiate(self.cfg.propensity_score_model)
            pi_model = train_nuisance_models(
                pi_model,
                np.array(X0_prop),
                np.array(Y0).ravel(),
                np.array(A0),
            )

            pi_target = jnp.array(
                pi_model.predict_proba(np.array(X1_prop))[:, 1]
            )
            pi_target = jnp.clip(pi_target, 0.1, 0.9)

            if self.cfg.model.learn_sigma:
                sigma_init_val = jax.nn.softplus(model_or.sig_param)
            else:
                sigma_init_val = sigma_init

            model = ModelClass(
                k_target_init,
                dim_V,
                self.cfg.model.hidden_dim,
                self.cfg.model.output_dim,
                ypcl,
                sigma_init_val,
                self.cfg.model.lamb,
                learn_sig=False,
            )
            state = eqx.nn.State(model)

            if self.method != "ridge":
                optim = optax.sgd(self.cfg.train.lr_fi, momentum=0.9)
                opt_state = optim.init(
                    eqx.filter(model, eqx.is_inexact_array)
                )

                valid_size_fi = getattr(self.cfg.train, "valid_size_fi", None)
                if valid_size_fi is not None and valid_size_fi > 0:
                    if valid_size_fi < 1.0:
                        n_valid_fi = int(X1_or.shape[0] * valid_size_fi)
                    else:
                        n_valid_fi = int(valid_size_fi)

                    n_t = X1_or.shape[0] - n_valid_fi
                    X_train_fi, X_valid_fi = X1_or[:n_t], X1_or[n_t:]
                    V_train_fi, V_valid_fi = V1[:n_t], V1[n_t:]
                    Y_train_fi, Y_valid_fi = Y1[:n_t], Y1[n_t:]
                    A_train_fi, A_valid_fi = A1[:n_t], A1[n_t:]
                    pi_train_fi, pi_valid_fi = (
                        pi_target[:n_t],
                        pi_target[n_t:],
                    )
                else:
                    n_t = X1_or.shape[0]
                    X_train_fi, V_train_fi, Y_train_fi, A_train_fi, pi_train_fi = (
                        X1_or,
                        V1,
                        Y1,
                        A1,
                        pi_target,
                    )
                    X_valid_fi = None

                batch_size = self.cfg.train.batch_size_fi
                n_batches = int(np.ceil(n_t / batch_size))

                patience_fi = getattr(self.cfg.train, "patience_fi", None)
                best_val_loss_fi = float("inf")
                epochs_without_improvement_fi = 0
                best_model_fi = None
                best_state_fi = None

                pbar = tqdm(
                    range(self.cfg.train.epoch_fi),
                    desc=f"{fold_label}Stage 2",
                    position=0,
                    disable=not self.verbose,
                    bar_format="{desc}: |{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]{postfix}",
                )
                for epoch in pbar:
                    k_target_loop, k1 = jax.random.split(k_target_loop)
                    perm1 = jax.random.permutation(k1, n_t)

                    X1_s, V1_s, Y1_s, A1_s, pi1_s = (
                        X_train_fi[perm1],
                        V_train_fi[perm1],
                        Y_train_fi[perm1],
                        A_train_fi[perm1],
                        pi_train_fi[perm1],
                    )

                    epoch_loss = 0.0
                    for i in range(n_batches):
                        idx = i * batch_size
                        be = min(idx + batch_size, n_t)
                        curr_bs = be - idx

                        idx0 = (i * batch_size) % X0_ref_or.shape[0]
                        if idx0 + curr_bs <= X0_ref_or.shape[0]:
                            X_ref_b = X0_ref_or[idx0 : idx0 + curr_bs]
                            Y_ref_b = Y0_ref[idx0 : idx0 + curr_bs]
                        else:
                            rem = (idx0 + curr_bs) - X0_ref_or.shape[0]
                            X_ref_b = jnp.concatenate(
                                [X0_ref_or[idx0:], X0_ref_or[:rem]]
                            )
                            Y_ref_b = jnp.concatenate(
                                [Y0_ref[idx0:], Y0_ref[:rem]]
                            )

                        model, state, opt_state, loss = train_step_target(
                            model,
                            state,
                            optim,
                            opt_state,
                            V1_s[idx:be],
                            X1_s[idx:be],
                            Y1_s[idx:be],
                            A1_s[idx:be],
                            model_or,
                            state_or,
                            pi1_s[idx:be],
                            loss_fn_target,
                            X_ref_b,
                            Y_ref_b,
                        )
                        epoch_loss += loss.item()

                    postfix_dict = {"train_loss": f"{epoch_loss / n_batches:.5f}"}

                    if X_valid_fi is not None:
                        val_bs = min(batch_size, X0_ref_or.shape[0])
                        X_ref_val = X0_ref_or[:val_bs]
                        Y_ref_val = Y0_ref[:val_bs]

                        val_loss, _ = loss_fn_target(
                            _zero_lamb(model),
                            state,
                            V_valid_fi,
                            X_valid_fi,
                            Y_valid_fi,
                            A_valid_fi,
                            model_or,
                            state_or,
                            pi_valid_fi,
                            X_ref_val,
                            Y_ref_val,
                        )
                        val_loss_value = val_loss.item()
                        postfix_dict["val_loss"] = f"{val_loss_value:.5f}"

                        if patience_fi is not None and patience_fi > 0:
                            if val_loss_value < best_val_loss_fi:
                                best_val_loss_fi = val_loss_value
                                epochs_without_improvement_fi = 0
                                best_model_fi = model
                                best_state_fi = state
                                postfix_dict["patience"] = (
                                    f"0000/{patience_fi:04}"
                                )
                            else:
                                epochs_without_improvement_fi += 1
                                postfix_dict["patience"] = (
                                    f"{epochs_without_improvement_fi:04}/{patience_fi:04}"
                                )

                            if epochs_without_improvement_fi >= patience_fi:
                                pbar.close()
                                if self.verbose:
                                    tqdm.write(
                                        f"{fold_label}Stage 2: Early stopping at epoch {epoch + 1}, "
                                        f"restoring best model (val_loss={best_val_loss_fi:.5f})"
                                    )
                                if best_model_fi is not None:
                                    model = best_model_fi
                                    state = best_state_fi
                                break

                    pbar.set_postfix(postfix_dict)

            model = eqx.nn.inference_mode(model)

        return FoldResult(
            model_or=model_or,
            state_or=state_or,
            model=model,
            state=state,
            pi_model=pi_model,
            X_aux=X0_ref_or,
            Y_aux=Y0_ref,
            V_target=V1,
            X_target=X1_or,
            Y_target=Y1,
            A_target=A1,
            pi_target=pi_target,
            eval_fn=eval_fn,
        )

    def fit(self, X, V, Y, A, grid_points=None):
        """Train nuisance and target models on observational data.

        Performs data shuffling, sample splitting (or K-fold partitioning),
        and sequentially trains Stage 1 (nuisance) and Stage 2 (target) models
        for each fold.

        Args:
            X: Full covariates of shape ``(N, D_x)``.
            V: Conditioning variables of shape ``(N, D_v)``.  Must satisfy
                ``V.shape[0] == X.shape[0]``.  Typically a subset of columns
                of ``X``.
            Y: Outcomes of shape ``(N,)`` or ``(N, 1)``.
            A: Binary treatment indicators of shape ``(N,)`` or ``(N, 1)``.
            grid_points: Optional inducing points of shape ``(M, D_y)`` for
                the kernel grid.  If provided, overrides ``cfg.model.output_dim``.
                Useful for high-dimensional outcomes (e.g. images) where a
                uniform grid is not appropriate.

        Returns:
            self, for method chaining.

        Raises:
            ValueError: If ``X`` and ``V`` have different numbers of rows.
        """
        X = jnp.array(X)
        V = jnp.array(V)
        Y = jnp.array(Y)
        A = jnp.array(A)

        if X.shape[0] != V.shape[0]:
            raise ValueError(
                f"Shape mismatch: X has {X.shape[0]} rows, V has {V.shape[0]}"
            )

        if Y.ndim == 1:
            Y = Y[:, None]
        if A.ndim == 2:
            A = A.squeeze()

        idx_pi = get_indices(self.cfg.model, "idx_pi", X.shape[1])
        idx_or = get_indices(self.cfg.model, "idx_or", X.shape[1])

        if self.cfg.model.sigma_init == "median":
            sigma_init = get_median_heuristic(Y, random_seed=self.seed)
        else:
            sigma_init = self.cfg.model.sigma_init

        n = X.shape[0]

        perm = jax.random.permutation(self.k_data_shuffle, n)
        X, V, Y, A = X[perm], V[perm], Y[perm], A[perm]

        X_prop = X[:, idx_pi] if idx_pi is not None else X
        if self.mode == "onestep":
            X_or = V
        else:
            X_or = X[:, idx_or] if idx_or is not None else X

        # ----------------------------------------
        # Component dispatch and grid setup
        # ----------------------------------------
        ModelClass, loss_fn_nuis, loss_fn_target, eval_fn = get_components(
            self.method, mode=self.mode
        )
        self.eval_fn = eval_fn

        if grid_points is not None:
            ypcl = jnp.array(grid_points)
            if ypcl.shape[0] != self.cfg.model.output_dim:
                warnings.warn(
                    f"Provided grid_points size ({ypcl.shape[0]}) does not match "
                    f"cfg.model.output_dim ({self.cfg.model.output_dim}). "
                    f"Adjusting num_atoms to {ypcl.shape[0]}."
                )
            num_atoms = ypcl.shape[0]
        else:
            num_atoms = self.cfg.model.output_dim
            if Y.shape[1] >= 2:
                warnings.warn(
                    "Y is multi-dimensional; randomly selecting "
                    f"{num_atoms} observations from Y as grid points."
                )
                idx = jax.random.choice(
                    self.k_data_shuffle, Y.shape[0], shape=(num_atoms,), replace=False
                )
                ypcl = Y[idx]
            else:
                y_min, y_max = jnp.min(Y), jnp.max(Y)
                ypcl = jnp.linspace(y_min, y_max, num_atoms)[:, None]

        dim_X_or = X_or.shape[1]
        dim_V = V.shape[1]

        # ----------------------------------------
        # Fit fold(s)
        # ----------------------------------------
        n_folds = self.n_folds
        if n_folds is not None and self.mode == "onestep":
            if self.verbose:
                print(
                    "Note: n_folds is ignored for onestep mode "
                    "(using full training set)."
                )
            n_folds = None

        if self.mode == "onestep":
            # onestep uses the full dataset for nuisance training
            # (no sample splitting needed since there is no second stage)
            result = self._fit_single_fold(
                fold_idx=0,
                n_folds_total=1,
                X0_prop=X_prop,
                X0_or=X_or,
                Y0=Y,
                A0=A,
                X1_prop=X_prop,
                X1_or=X_or,
                V1=V,
                Y1=Y,
                A1=A,
                ModelClass=ModelClass,
                loss_fn_nuis=loss_fn_nuis,
                loss_fn_target=loss_fn_target,
                eval_fn=eval_fn,
                dim_X_or=dim_X_or,
                dim_V=dim_V,
                ypcl=ypcl,
                sigma_init=sigma_init,
                k_nuis_init=self.k_nuis_init,
                k_nuis_loop=self.k_nuis_loop,
                k_target_init=self.k_target_init,
                k_target_loop=self.k_target_loop,
            )
            self._fold_results = [result]

            # Populate flat attributes for backward compatibility
            self.model_or = result.model_or
            self.state_or = result.state_or
            self.model = result.model
            self.state = result.state
            self.pi_model = result.pi_model
            self.X_aux = result.X_aux
            self.Y_aux = result.Y_aux
            self.V_target = result.V_target
            self.X_target = result.X_target
            self.Y_target = result.Y_target
            self.A_target = result.A_target
            self.pi_target = result.pi_target

        elif n_folds is None:
            # 50-50 sample splitting (DR, IPW, PI modes)
            n0 = n // 2
            result = self._fit_single_fold(
                fold_idx=0,
                n_folds_total=1,
                X0_prop=X_prop[:n0],
                X0_or=X_or[:n0],
                Y0=Y[:n0],
                A0=A[:n0],
                X1_prop=X_prop[n0:],
                X1_or=X_or[n0:],
                V1=V[n0:],
                Y1=Y[n0:],
                A1=A[n0:],
                ModelClass=ModelClass,
                loss_fn_nuis=loss_fn_nuis,
                loss_fn_target=loss_fn_target,
                eval_fn=eval_fn,
                dim_X_or=dim_X_or,
                dim_V=dim_V,
                ypcl=ypcl,
                sigma_init=sigma_init,
                k_nuis_init=self.k_nuis_init,
                k_nuis_loop=self.k_nuis_loop,
                k_target_init=self.k_target_init,
                k_target_loop=self.k_target_loop,
            )
            self._fold_results = [result]

            # Populate flat attributes for backward compatibility
            self.model_or = result.model_or
            self.state_or = result.state_or
            self.model = result.model
            self.state = result.state
            self.pi_model = result.pi_model
            self.X_aux = result.X_aux
            self.Y_aux = result.Y_aux
            self.V_target = result.V_target
            self.X_target = result.X_target
            self.Y_target = result.Y_target
            self.A_target = result.A_target
            self.pi_target = result.pi_target

        else:
            # K-fold cross-fitting
            k_nuis_init_folds = jax.random.split(
                self.k_nuis_init, n_folds
            )
            k_nuis_loop_folds = jax.random.split(
                self.k_nuis_loop, n_folds
            )
            k_target_init_folds = jax.random.split(
                self.k_target_init, n_folds
            )
            k_target_loop_folds = jax.random.split(
                self.k_target_loop, n_folds
            )

            fold_size = n // n_folds
            self._fold_results = []

            for k in range(n_folds):
                start_k = k * fold_size
                end_k = start_k + fold_size if k < n_folds - 1 else n
                target_mask = np.zeros(n, dtype=bool)
                target_mask[start_k:end_k] = True
                nuis_mask = ~target_mask

                result = self._fit_single_fold(
                    fold_idx=k,
                    n_folds_total=n_folds,
                    X0_prop=X_prop[nuis_mask],
                    X0_or=X_or[nuis_mask],
                    Y0=Y[nuis_mask],
                    A0=A[nuis_mask],
                    X1_prop=X_prop[target_mask],
                    X1_or=X_or[target_mask],
                    V1=V[target_mask],
                    Y1=Y[target_mask],
                    A1=A[target_mask],
                    ModelClass=ModelClass,
                    loss_fn_nuis=loss_fn_nuis,
                    loss_fn_target=loss_fn_target,
                    eval_fn=eval_fn,
                    dim_X_or=dim_X_or,
                    dim_V=dim_V,
                    ypcl=ypcl,
                    sigma_init=sigma_init,
                    k_nuis_init=k_nuis_init_folds[k],
                    k_nuis_loop=k_nuis_loop_folds[k],
                    k_target_init=k_target_init_folds[k],
                    k_target_loop=k_target_loop_folds[k],
                )
                self._fold_results.append(result)

        self.is_fitted = True
        return self

    def _predict_single_fold(self, fold_result, V_eval, Y_grid, batch_size):
        """Run prediction using one fold's trained models and stored data.

        Evaluates the appropriate inference function (determined at fit time)
        on the given evaluation points, batching over ``Y_grid`` if a
        ``batch_size`` is specified.

        Args:
            fold_result: A :class:`FoldResult` from :meth:`_fit_single_fold`.
            V_eval: Evaluation conditioning variables, shape ``(M, D_v)``.
            Y_grid: Outcome grid, shape ``(G, D_y)``.
            batch_size: Number of ``Y_grid`` rows per batch, or ``None`` to
                process all at once.

        Returns:
            Raw (un-normalized) density matrix of shape ``(M, G)``.
        """
        n_samples = Y_grid.shape[0]
        bs = batch_size if batch_size is not None else n_samples

        pdf_batches = []

        for i in range(0, n_samples, bs):
            Y_batch = Y_grid[i : i + bs]

            if self.mode == "onestep":
                eval_kwargs = {
                    "model": fold_result.model_or,
                    "state": fold_result.state_or,
                    "V_test": V_eval,
                    "yc": Y_batch,
                    "V_train": fold_result.X_aux,
                    "Y_train": fold_result.Y_aux,
                }
            else:
                eval_kwargs = {
                    "model": fold_result.model,
                    "state": fold_result.state,
                    "V_test": V_eval,
                    "yc": Y_batch,
                    "model_or": fold_result.model_or,
                    "state_or": fold_result.state_or,
                    "V_target": fold_result.V_target,
                    "X_target": fold_result.X_target,
                    "Y_target": fold_result.Y_target,
                    "A_target": fold_result.A_target,
                    "pi_target": fold_result.pi_target,
                    "X_aux": fold_result.X_aux,
                    "Y_aux": fold_result.Y_aux,
                    "mode": self.mode,
                }

            batch_pdf_est, _ = fold_result.eval_fn(**eval_kwargs)
            pdf_batches.append(np.array(batch_pdf_est))

        return np.concatenate(pdf_batches, axis=1)

    def predict(self, V_eval, Y_grid, batch_size=None):
        """Estimate the counterfactual conditional density P(Y^1 | V).

        For K-fold models, predictions from each fold are averaged before
        normalization.  The output is clipped to non-negative values and
        normalized to integrate to 1 over the provided grid.

        Args:
            V_eval: Evaluation points of shape ``(M, D_v)`` or ``(D_v,)``
                (a single point, which is automatically expanded).
            Y_grid: Outcome grid of shape ``(G, D_y)`` or ``(G,)`` (univariate
                outcomes, automatically expanded to ``(G, 1)``).
            batch_size: Number of ``Y_grid`` rows to process at once.
                ``None`` processes all rows in a single call (may require
                more memory for large grids).

        Returns:
            Normalized density estimates of shape ``(M, G)``.

        Raises:
            RuntimeError: If the estimator has not been fitted yet.
        """
        if not self.is_fitted:
            raise RuntimeError("Estimator is not fitted. Call fit() first.")

        if self.verbose:
            print(
                f"Estimating Densities... (Batch size: {batch_size if batch_size else 'Full'})"
            )

        V_eval = jnp.array(V_eval)
        if len(V_eval.shape) == 1:
            V_eval = V_eval[None, :]

        Y_grid = jnp.array(Y_grid)
        if len(Y_grid.shape) == 1:
            Y_grid = Y_grid[:, None]

        if len(self._fold_results) == 1:
            pdf_est = self._predict_single_fold(
                self._fold_results[0], V_eval, Y_grid, batch_size
            )
        else:
            pdf_estimates = []
            for fold_result in self._fold_results:
                pdf_k = self._predict_single_fold(
                    fold_result, V_eval, Y_grid, batch_size
                )
                pdf_estimates.append(pdf_k)
            pdf_est = np.mean(pdf_estimates, axis=0)

        pdf_est = np.clip(pdf_est, 0, None)

        if Y_grid.shape[0] > 1 and Y_grid.shape[1] == 1:
            dy = Y_grid[1, 0] - Y_grid[0, 0]
            sums = np.sum(pdf_est, axis=1, keepdims=True)
            sums[sums < 1e-9] = 1.0
            pdf_est = pdf_est / (sums * dy)

        return pdf_est