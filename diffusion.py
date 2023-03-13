import torch
import torch.nn as nn
import torch.nn.functional as F


def extract(v, t, x_shape):
    """
    Extract some coefficients at specified timesteps, then reshape to
    [batch_size, 1, 1, 1, 1, ...] for broadcasting purposes.

    args:
        v : shape [T]
        t : shape [batch_size]
        x_shape : shape of x_p for some p (batch_size, img_size, img_size)
    Returns:
        Extracts the coefficients at indices specified by the values of
        t
    """
    out = torch.gather(v, index=t, dim=0).float()
    return out.view([t.shape[0]] + [1] * (len(x_shape) - 1))


class GaussianDiffusionTrainer(nn.Module):
    def __init__(self, model, beta_1, beta_T, T):
        super().__init__()

        self.model = model
        self.T = T

        self.register_buffer("betas", torch.linspace(beta_1, beta_T, T).double())
        alphas = 1.0 - self.betas
        alphas_bar = torch.cumprod(alphas, dim=0)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer("sqrt_alphas_bar", torch.sqrt(alphas_bar))
        self.register_buffer("sqrt_one_minus_alphas_bar", torch.sqrt(1.0 - alphas_bar))

    def forward(self, x_0):
        """
        Algorithm 1.
        Randomly choose one t<T; makes a forward pass from x_0 to x_t
        with random noise then predicts the noise added to x_0 to obtain
        x_0 using the mse_loss.
        Performs this on a batch (x_0 in args is a batch of x_0s)
        """
        t = torch.randint(self.T, size=(x_0.shape[0],), device=x_0.device)
        noise = torch.randn_like(x_0)
        x_t = (
            extract(self.sqrt_alphas_bar, t, x_0.shape) * x_0
            + extract(self.sqrt_one_minus_alphas_bar, t, x_0.shape) * noise
        )
        loss = F.mse_loss(self.model(x_t, t), noise, reduction="none")
        return loss


class GaussianDiffusionSampler(nn.Module):
    def __init__(
        self,
        model,
        beta_1,
        beta_T,
        T,
        img_size=32,
        mean_type="eps",
        var_type="fixedlarge",
    ):
        assert mean_type in ["xprev" "xstart", "epsilon"]
        assert var_type in ["fixedlarge", "fixedsmall"]
        super().__init__()

        self.model = model
        self.T = T
        self.img_size = img_size
        self.mean_type = mean_type
        self.var_type = var_type

        self.register_buffer("betas", torch.linspace(beta_1, beta_T, T).double())
        alphas = 1.0 - self.betas
        alphas_bar = torch.cumprod(alphas, dim=0)
        alphas_bar_prev = F.pad(alphas_bar, [1, 0], value=1)[:T]

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer("sqrt_recip_alphas_bar", torch.sqrt(1.0 / alphas_bar))
        self.register_buffer(
            "sqrt_recipm1_alphas_bar", torch.sqrt(1.0 / alphas_bar - 1)
        )

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        self.register_buffer(
            "posterior_var", self.betas * (1.0 - alphas_bar_prev) / (1.0 - alphas_bar)
        )
        # below: log calculation clipped because the posterior variance is 0 at
        # the beginning of the diffusion chain
        self.register_buffer(
            "posterior_log_var_clipped",
            torch.log(torch.cat([self.posterior_var[1:2], self.posterior_var[1:]])),
        )
        self.register_buffer(
            "posterior_mean_coef1",
            torch.sqrt(alphas_bar_prev) * self.betas / (1.0 - alphas_bar),
        )
        self.register_buffer(
            "posterior_mean_coef2",
            torch.sqrt(alphas) * (1.0 - alphas_bar_prev) / (1.0 - alphas_bar),
        )

    def q_mean_variance(self, x_0, x_t, t):
        """
        Compute the mean and variance of the diffusion posterior
        q(x_{t-1} | x_t, x_0)
        """
        assert x_0.shape == x_t.shape
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_0
            + extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )  # eq 7 paper
        posterior_log_var_clipped = extract(
            self.posterior_log_var_clipped, t, x_t.shape
        )
        return posterior_mean, posterior_log_var_clipped

    def predict_xstart_from_eps(self, x_t, t, eps):
        assert x_t.shape == eps.shape
        return (
            extract(self.sqrt_recip_alphas_bar, t, x_t.shape) * x_t
            - extract(self.sqrt_recipm1_alphas_bar, t, x_t.shape) * eps
        )  # cf between eq 8 and eq 9 in paper ( expression of xt(x0,eps) )

    def predict_xstart_from_xprev(self, x_t, t, xprev):
        assert x_t.shape == xprev.shape
        return (  # (xprev - coef2*x_t) / coef1
            extract(1.0 / self.posterior_mean_coef1, t, x_t.shape) * xprev
            - extract(
                self.posterior_mean_coef2 / self.posterior_mean_coef1, t, x_t.shape
            )
            * x_t
        )

    def p_mean_variance(self, x_t, t):
        # below: only log_variance is used in the KL computations
        model_log_var = {
            # for fixedlarge, we set the initial (log-)variance like so to
            # get a better decoder log likelihood
            "fixedlarge": torch.log(
                torch.cat([self.posterior_var[1:2], self.betas[1:]])
            ),
            "fixedsmall": self.posterior_log_var_clipped,
        }[self.var_type]
        model_log_var = extract(model_log_var, t, x_t.shape)

        # Mean parameterization
        if self.mean_type == "xprev":  # the model predicts x_{t-1}
            x_prev = self.model(x_t, t)
            x_0 = self.predict_xstart_from_xprev(x_t, t, xprev=x_prev)
            model_mean = x_prev
        elif self.mean_type == "xstart":  # the model predicts x_0
            x_0 = self.model(x_t, t)
            model_mean, _ = self.q_mean_variance(x_0, x_t, t)
        elif self.mean_type == "epsilon":  # the model predicts epsilon
            eps = self.model(x_t, t)
            x_0 = self.predict_xstart_from_eps(x_t, t, eps=eps)
            model_mean, _ = self.q_mean_variance(x_0, x_t, t)
            # cf eq 11 with \tilde{\mu} : mu_theta can be
            # computing using q_mean_variance
        else:
            raise NotImplementedError(self.mean_type)
        x_0 = torch.clip(x_0, -1.0, 1.0)

        return model_mean, model_log_var

    def forward(self, x_T):
        """
        Algorithm 2.
        Used to generate an image starting from x_T
        """
        x_t = x_T
        for time_step in reversed(range(self.T)):
            t = (
                x_t.new_ones(
                    [
                        x_T.shape[0],
                    ],
                    dtype=torch.long,
                )
                * time_step
            )  # t is a long tensor of size batch_size = x_T.shape[0]
            # with one value. ex : [1,1,1,....,1]
            mean, log_var = self.p_mean_variance(x_t=x_t, t=t)
            # no noise when t == 0
            if time_step > 0:
                noise = torch.randn_like(x_t)
            else:
                noise = 0
            x_t = mean + torch.exp(0.5 * log_var) * noise
        x_0 = x_t
        return torch.clip(x_0, -1, 1)
