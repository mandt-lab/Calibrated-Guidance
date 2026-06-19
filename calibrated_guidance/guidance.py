import torch

from calibrated_guidance.diffusion_posterior.base import DiffusionPosterior, SampleOutput
from calibrated_guidance.utils import broadcast_time


def masked_likelihood_compute(log_likelihood_fn, guidance_scale, x0_samples: torch.Tensor,
                              known_likelihoods: torch.Tensor,
                              known_likelihoods_mask: torch.Tensor):
    if known_likelihoods_mask is None:
        return guidance_scale * log_likelihood_fn(x0_samples)
    else:
        log_likelihoods = torch.zeros(x0_samples.shape[0], x0_samples.shape[1], device=x0_samples.device)
        if known_likelihoods_mask.any():
            log_likelihoods[known_likelihoods_mask] = known_likelihoods[known_likelihoods_mask]
        if (~known_likelihoods_mask).any():
            log_likelihoods[~known_likelihoods_mask] = guidance_scale * log_likelihood_fn(
                x0_samples[~known_likelihoods_mask]
            )
        return log_likelihoods


def build_reinforce_mean_fn(
        diffusion_posterior: DiffusionPosterior, log_likelihood_fn, num_samples_per_step,
        guidance_scale: float = 1.0, outer_guidance_scale: float = 1.0,
        plot_samples=None
):
    """
    Returns estimate for E[x0|xt,y] with REINFORCE.

    :param diffusion_posterior:
    :param log_likelihood_fn:
    :param num_samples_per_step:
    :param guidance_scale:
    :param outer_guidance_scale:
    :param plot_samples:
    :return:
    """
    def reinforce_mean_fn(xt, t):
        assert xt.requires_grad == False, f"REINFORCE mean function does not support gradients w.r.t. xt (time {t})"
        with torch.no_grad():
            x0_samples = diffusion_posterior.sample(xt, t, num_samples_per_step)
            assert isinstance(x0_samples, SampleOutput)
            log_likelihoods = masked_likelihood_compute(
                log_likelihood_fn, guidance_scale, x0_samples.samples,
                x0_samples.known_log_likelihoods,
                x0_samples.known_likelihoods_mask
            )
            weights = torch.softmax(log_likelihoods + x0_samples.log_weights, dim=1)

        if plot_samples is not None:
            plot_samples(xt, x0_samples, log_likelihoods, t)

        weights = torch.softmax(log_likelihoods + x0_samples.log_weights, dim=1)
        guided_mean = (
                weights.reshape(weights.shape[0], weights.shape[1], *[1] * len(xt.shape[1:]))
                * x0_samples.samples
        ).sum(dim=1)
        if outer_guidance_scale != 1.0:
            unguided_mean = diffusion_posterior.mean(xt, t)
            guided_mean = unguided_mean + outer_guidance_scale * (guided_mean - unguided_mean)
        return guided_mean

    return reinforce_mean_fn


def build_reparam_mean_fn(
        diffusion_posterior: DiffusionPosterior, log_likelihood_fn, num_samples_per_step,
        guidance_scale: float = 1.0, outer_guidance_scale: float = 1.0,
        plot_samples=None,
):
    """
    Returns estimate for E[x0|xt,y] with reparameterization trick

    :param diffusion_posterior:
    :param log_likelihood_fn:
    :param num_samples_per_step:
    :param guidance_scale:
    :param outer_guidance_scale:
    :return:
    """
    def reparam_mean_fn(xt, t):
        # If t is simple
        t_mixed = torch.is_tensor(t) and len(t.shape) > 0
        if not t_mixed and t >= 1.0 - 1e-8:
            return diffusion_posterior.mean(xt, t)

        xt_grad = xt.detach().clone()
        xt_grad.requires_grad_(True)
        del xt

        x0_samples = diffusion_posterior.sample(xt_grad, t, num_samples_per_step)
        assert x0_samples.samples.requires_grad, "Samples x0_samples should have gradients attached."
        unguided_mean = x0_samples.samples.mean(dim=1).detach()

        log_likelihoods = guidance_scale * log_likelihood_fn(x0_samples.samples)
        logp_y_xt = torch.logsumexp(log_likelihoods, dim=1).sum()
        assert logp_y_xt.requires_grad, "logp_y_xt should have gradients attached."
        
        if plot_samples is not None:
            with torch.no_grad():
                plot_samples(xt_grad, x0_samples, guidance_scale * log_likelihoods, t)
        
        guidance = torch.autograd.grad(logp_y_xt, xt_grad)[0]

        t = broadcast_time(t, guidance)
        a_t = 1 - t
        b_t = t
        guided_mean = unguided_mean + b_t ** 2 / a_t * guidance * outer_guidance_scale
        return guided_mean

    return reparam_mean_fn
