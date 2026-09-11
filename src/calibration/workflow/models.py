from __future__ import annotations

import arviz as az
import numpy as np
import pymc as pm
import pytensor.tensor as pt


def fit_nb(
    X: np.ndarray,
    y: np.ndarray,
    muni_codes: np.ndarray,
    n_muni: int,
    draws: int,
    tune: int,
    chains: int,
    target_accept: float,
    random_seed: int | None = None,
) -> az.InferenceData:
    with pm.Model() as model:
        x_data = pm.Data("x_data", X)
        muni_idx = pm.Data("muni_idx", muni_codes)
        alpha_muni = pm.Normal("alpha_muni", mu=0.0, sigma=1.0)
        sigma_muni = pm.HalfNormal("sigma_muni", sigma=1.0)
        muni_offset = pm.Normal("muni_offset", mu=0.0, sigma=1.0, shape=n_muni)
        muni_term = alpha_muni + sigma_muni * muni_offset[muni_idx]
        beta = pm.Normal("beta", mu=0.0, sigma=1.0, shape=X.shape[1])
        eta = muni_term + pt.dot(x_data, beta)
        mu = pm.Deterministic("mu", pt.exp(pt.clip(eta, -12.0, 12.0)))
        alpha = pm.HalfNormal("alpha", sigma=2.0)
        pm.NegativeBinomial("y", mu=mu, alpha=alpha, observed=y)
        return pm.sample(
            draws=draws,
            tune=tune,
            chains=chains,
            target_accept=target_accept,
            random_seed=random_seed,
            progressbar=False,
            return_inferencedata=True,
        )


def fit_nb_offset(
    offset_log: np.ndarray,
    y: np.ndarray,
    muni_codes: np.ndarray,
    n_muni: int,
    draws: int,
    tune: int,
    chains: int,
    target_accept: float,
    random_seed: int | None = None,
) -> az.InferenceData:
    with pm.Model() as model:
        offset_data = pm.Data("offset_data", offset_log)
        muni_idx = pm.Data("muni_idx", muni_codes)
        intercept = pm.Normal("intercept", mu=0.0, sigma=1.0)
        alpha_muni = pm.Normal("alpha_muni", mu=0.0, sigma=1.0)
        sigma_muni = pm.HalfNormal("sigma_muni", sigma=1.0)
        muni_offset = pm.Normal("muni_offset", mu=0.0, sigma=1.0, shape=n_muni)
        muni_term = alpha_muni + sigma_muni * muni_offset[muni_idx]
        eta = intercept + muni_term + offset_data
        mu = pm.Deterministic("mu", pt.exp(pt.clip(eta, -12.0, 12.0)))
        alpha = pm.HalfNormal("alpha", sigma=2.0)
        pm.NegativeBinomial("y", mu=mu, alpha=alpha, observed=y)
        return pm.sample(
            draws=draws,
            tune=tune,
            chains=chains,
            target_accept=target_accept,
            random_seed=random_seed,
            progressbar=False,
            return_inferencedata=True,
        )


def fit_nb_offset_features(
    offset_log: np.ndarray,
    X_features: np.ndarray,
    y: np.ndarray,
    muni_codes: np.ndarray,
    n_muni: int,
    draws: int,
    tune: int,
    chains: int,
    target_accept: float,
    mean_coef_sigma: float = 0.15,
    random_seed: int | None = None,
) -> az.InferenceData:
    with pm.Model() as model:
        offset_data = pm.Data("offset_data", offset_log)
        x_feat = pm.Data("x_feat", X_features)
        muni_idx = pm.Data("muni_idx", muni_codes)

        intercept = pm.Normal("intercept", mu=0.0, sigma=1.0)
        alpha_muni = pm.Normal("alpha_muni", mu=0.0, sigma=1.0)
        sigma_muni = pm.HalfNormal("sigma_muni", sigma=1.0)
        muni_offset = pm.Normal("muni_offset", mu=0.0, sigma=1.0, shape=n_muni)
        muni_term = alpha_muni + sigma_muni * muni_offset[muni_idx]

        beta_mean = pm.Normal("beta_mean", mu=0.0, sigma=mean_coef_sigma, shape=X_features.shape[1])
        eta_mean = intercept + muni_term + offset_data + pt.dot(x_feat, beta_mean)
        mu = pm.Deterministic("mu", pt.exp(pt.clip(eta_mean, -12.0, 12.0)))

        alpha_intercept = pm.Normal("alpha_intercept", mu=np.log(2.0), sigma=1.0)
        gamma_alpha = pm.Normal("gamma_alpha", mu=0.0, sigma=0.5, shape=X_features.shape[1])
        eta_alpha = alpha_intercept + pt.dot(x_feat, gamma_alpha)
        alpha = pm.Deterministic("alpha", pt.exp(pt.clip(eta_alpha, -8.0, 8.0)))

        pm.NegativeBinomial("y", mu=mu, alpha=alpha, observed=y)
        return pm.sample(
            draws=draws,
            tune=tune,
            chains=chains,
            target_accept=target_accept,
            random_seed=random_seed,
            progressbar=False,
            return_inferencedata=True,
        )


def fit_hurdle_nb(
    X_count: np.ndarray,
    X_zero: np.ndarray,
    y: np.ndarray,
    muni_codes: np.ndarray,
    n_muni: int,
    draws: int,
    tune: int,
    chains: int,
    target_accept: float,
    random_seed: int | None = None,
) -> az.InferenceData:
    zero = (y == 0).astype(int)
    pos_mask = y > 0

    with pm.Model() as model:
        x_count_data = pm.Data("x_count_data", X_count)
        x_zero_data = pm.Data("x_zero_data", X_zero)
        muni_idx = pm.Data("muni_idx", muni_codes)
        alpha_muni = pm.Normal("alpha_muni", mu=0.0, sigma=1.0)
        sigma_muni = pm.HalfNormal("sigma_muni", sigma=1.0)
        muni_offset = pm.Normal("muni_offset", mu=0.0, sigma=1.0, shape=n_muni)
        muni_term = alpha_muni + sigma_muni * muni_offset[muni_idx]

        beta_count = pm.Normal("beta_count", mu=0.0, sigma=1.0, shape=X_count.shape[1])
        beta_zero = pm.Normal("beta_zero", mu=0.0, sigma=1.0, shape=X_zero.shape[1])
        eta_count = muni_term + pt.dot(x_count_data, beta_count)
        mu = pm.Deterministic("mu", pt.exp(pt.clip(eta_count, -12.0, 12.0)))
        alpha = pm.HalfNormal("alpha", sigma=2.0)

        eta_zero = muni_term + pt.dot(x_zero_data, beta_zero)
        p_zero = pm.Deterministic("p_zero", pm.math.sigmoid(eta_zero))

        pm.Bernoulli("z_obs", p=p_zero, observed=zero)
        pm.Truncated(
            "y_pos",
            pm.NegativeBinomial.dist(mu=mu[pos_mask], alpha=alpha),
            lower=1,
            observed=y[pos_mask],
        )
        return pm.sample(
            draws=draws,
            tune=tune,
            chains=chains,
            target_accept=target_accept,
            random_seed=random_seed,
            progressbar=False,
            return_inferencedata=True,
        )


def expected_count_nb(idata: az.InferenceData, X: np.ndarray, muni_codes: np.ndarray) -> np.ndarray:
    post = idata.posterior
    beta = post["beta"].stack(sample=("chain", "draw")).values
    alpha_muni = post["alpha_muni"].stack(sample=("chain", "draw")).values
    sigma_muni = post["sigma_muni"].stack(sample=("chain", "draw")).values
    muni_offset = post["muni_offset"].stack(sample=("chain", "draw")).values
    muni_term = alpha_muni[None, :] + sigma_muni[None, :] * muni_offset[muni_codes, :]
    eta = X @ beta + muni_term
    eta = np.clip(eta, -20.0, 20.0)
    return np.exp(eta)


def expected_count_hurdle(idata: az.InferenceData, X_count: np.ndarray, X_zero: np.ndarray, muni_codes: np.ndarray) -> np.ndarray:
    post = idata.posterior
    beta_count = post["beta_count"].stack(sample=("chain", "draw")).values
    beta_zero = post["beta_zero"].stack(sample=("chain", "draw")).values
    alpha_muni = post["alpha_muni"].stack(sample=("chain", "draw")).values
    sigma_muni = post["sigma_muni"].stack(sample=("chain", "draw")).values
    muni_offset = post["muni_offset"].stack(sample=("chain", "draw")).values
    muni_term = alpha_muni[None, :] + sigma_muni[None, :] * muni_offset[muni_codes, :]
    eta_count = np.clip(X_count @ beta_count + muni_term, -20.0, 20.0)
    mu = np.exp(eta_count)
    eta_zero = np.clip(X_zero @ beta_zero + muni_term, -20.0, 20.0)
    p_zero = 1.0 / (1.0 + np.exp(-eta_zero))
    return (1.0 - p_zero) * mu


def expected_count_nb_offset(idata: az.InferenceData, offset_log: np.ndarray, muni_codes: np.ndarray) -> np.ndarray:
    post = idata.posterior
    intercept = post["intercept"].stack(sample=("chain", "draw")).values
    alpha_muni = post["alpha_muni"].stack(sample=("chain", "draw")).values
    sigma_muni = post["sigma_muni"].stack(sample=("chain", "draw")).values
    muni_offset = post["muni_offset"].stack(sample=("chain", "draw")).values
    muni_term = alpha_muni[None, :] + sigma_muni[None, :] * muni_offset[muni_codes, :]
    eta = np.clip(offset_log[:, None] + intercept[None, :] + muni_term, -20.0, 20.0)
    return np.exp(eta)


def expected_count_nb_offset_features(idata: az.InferenceData, offset_log: np.ndarray, X_features: np.ndarray, muni_codes: np.ndarray) -> np.ndarray:
    post = idata.posterior
    intercept = post["intercept"].stack(sample=("chain", "draw")).values
    alpha_muni = post["alpha_muni"].stack(sample=("chain", "draw")).values
    sigma_muni = post["sigma_muni"].stack(sample=("chain", "draw")).values
    muni_offset = post["muni_offset"].stack(sample=("chain", "draw")).values
    beta_mean = post["beta_mean"].stack(sample=("chain", "draw")).values
    muni_term = alpha_muni[None, :] + sigma_muni[None, :] * muni_offset[muni_codes, :]
    eta = np.clip(offset_log[:, None] + intercept[None, :] + muni_term + X_features @ beta_mean, -20.0, 20.0)
    return np.exp(eta)
