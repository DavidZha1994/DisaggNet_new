#################################################################################################################
#
# @copyright : ©2025 EDF
# @author : Adrien Petralia
# @description : DiffNILM baseline
#
#################################################################################################################

# Codes are mainly taken and adapted from:
# https://github.com/maum-ai/nuwave
# https://github.com/lmnt-com/diffwave
# https://github.com/ivanvovk/WaveGrad
# https://github.com/lucidrains/denoising-diffusion-pytorch
# https://github.com/hojonathanho/diffusion

import torch
import torch.nn as nn
import torch.nn.functional as F

from math import sqrt

Linear = nn.Linear
silu = F.silu


def Conv1d(*args, **kwargs):
    layer = nn.Conv1d(*args, **kwargs)
    nn.init.kaiming_normal_(layer.weight)
    return layer


class DiffusionEmbedding(nn.Module):
    def __init__(self, pos_emb_channels, pos_emb_scale, pos_emb_dim):
        super().__init__()
        self.n_channels = pos_emb_channels
        self.scale = pos_emb_scale
        self.out_channels = pos_emb_dim

        half_dim = self.n_channels // 2
        exponents = torch.arange(half_dim, dtype=torch.float32) / float(half_dim)
        exponents = 1e-4**exponents
        self.register_buffer("exponents", exponents)
        self.projection1 = Linear(self.n_channels, self.out_channels)
        self.projection2 = Linear(self.out_channels, self.out_channels)

    # noise_level: [B]
    def forward(self, noise_level):
        x = self.scale * noise_level * self.exponents.unsqueeze(0)
        x = torch.cat([x.sin(), x.cos()], dim=-1)
        x = self.projection1(x)
        x = silu(x)
        x = self.projection2(x)
        x = silu(x)

        return x


class ResidualBlock(nn.Module):
    def __init__(self, residual_channels, dilation, pos_emb_dim):
        super().__init__()
        self.dilated_conv = Conv1d(
            residual_channels,
            2 * residual_channels,
            3,
            padding=dilation,
            dilation=dilation,
        )

        self.diffusion_projection = Linear(pos_emb_dim, residual_channels)

        self.output_projection = Conv1d(residual_channels, 2 * residual_channels, 1)

        self.agg_projection = Conv1d(
            residual_channels,
            2 * residual_channels,
            3,
            padding=dilation,
            dilation=dilation,
        )

    def forward(self, x, x_agg, noise_level):
        noise_level = self.diffusion_projection(noise_level).unsqueeze(-1)

        y = x + noise_level
        y = self.dilated_conv(y)
        y += self.agg_projection(x_agg)

        gate, filter = torch.chunk(y, 2, dim=1)
        y = torch.sigmoid(gate) * torch.tanh(filter)
        y = self.output_projection(y)
        residual, skip = torch.chunk(y, 2, dim=1)

        return (x + residual) / sqrt(2.0), skip


class DiffNILMBackbone(nn.Module):
    """
    Adaptation of NuWave backbone (https://arxiv.org/pdf/2104.02321) to NILM accoridng to DiffNILM paper: https://www.mdpi.com/1424-8220/23/7/3540

    The only difference with NuWave is that the conditioning is based on the additioon of the aggregate load curve signal and Timestamp encoded information
    """

    def __init__(
        self,
        c_in=1,
        c_embedding=3,
        residual_layers=30,
        residual_channels=128,
        dilation_cycle_length=10,
        pos_emb_channels=128,
        pos_emb_scale=50000,
        pos_emb_dim=512,
    ):
        super().__init__()

        self.input_projection = Conv1d(c_in, residual_channels, 1)
        self.agg_projection = Conv1d(c_in, residual_channels, 1)
        self.time_projection = Conv1d(c_embedding, residual_channels, 1)

        self.diffusion_embedding = DiffusionEmbedding(
            pos_emb_channels, pos_emb_scale, pos_emb_dim
        )

        self.residual_layers = nn.ModuleList(
            [
                ResidualBlock(
                    residual_channels, 2 ** (i % dilation_cycle_length), pos_emb_dim
                )
                for i in range(residual_layers)
            ]
        )
        self.len_res = len(self.residual_layers)

        self.skip_projection = Conv1d(residual_channels, residual_channels, 1)
        self.output_projection = Conv1d(residual_channels, 1, 1)

        nn.init.kaiming_normal_(self.output_projection.weight)

    def forward(self, x, x_agg, timestamp_enc, noise_level):
        """
        x: 2D Tensor (B, L)
        x_agg: 2D Tensor (B, L)
        timestamp_enc: 3D Tensor (B, 3, L) for hour, dow, month
        """

        # Appliance proj embedding and clipping
        x = self.input_projection(x.unsqueeze(1))
        x = silu(x)

        # Conditional proj embedding (aggregate power and timestamp info)
        x_agg = torch.add(
            self.agg_projection(x_agg.unsqueeze(1)), self.time_projection(timestamp_enc)
        )
        x_agg = silu(x_agg)

        noise_level = self.diffusion_embedding(noise_level)

        skip = 0.0
        for layer in self.residual_layers:
            x, skip_connection = layer(x, x_agg, noise_level)
            skip += skip_connection

        x = skip / sqrt(self.len_res)
        x = self.skip_projection(x)
        x = silu(x)
        x = self.output_projection(x)

        return x.squeeze(1)


@torch.jit.script
def lognorm(pred, target):
    """
    lognorm Loss as used in NuWave
    """
    return (pred - target).abs().mean(dim=-1).clamp(min=1e-20).log().mean()


class DiffNILM(nn.Module):
    """
    Adaptation of NuWave (https://arxiv.org/pdf/2104.02321) to NILM accoridng to DiffNILM paper: https://www.mdpi.com/1424-8220/23/7/3540

    NuWave PL class 'NuWave(pl.LightningModule)' taken and adapted to perform Diffusion training
    """

    def __init__(
        self,
        backbone=DiffNILMBackbone(),
        max_step=1000,
        infer_step=8,
        infer_schedule=None,
        **kwargs,
    ):
        super().__init__()

        self.model_backbone = backbone
        self.infer_step = infer_step
        self.optimizer = self.configure_optimizers(**kwargs)

        if infer_schedule is None:
            infer_schedule = torch.tensor(
                [1e-6, 2e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 9e-1]
            )

        # Store parameters for reuse
        self.max_step = max_step
        self.infer_step = infer_step
        self.infer_schedule = infer_schedule

        # Init noise schedling buffers
        self.init_noise_schedule(train=self.training)

    def init_noise_schedule(self, train=True):
        self.max_step = self.max_step if train else self.infer_step
        noise_schedule = (
            torch.linspace(1e-6, 0.006, self.max_step) if train else self.infer_schedule
        )

        self.register_buffer("betas", noise_schedule, False)
        self.register_buffer("alphas", 1 - self.betas, False)
        self.register_buffer("alphas_cumprod", self.alphas.cumprod(dim=0), False)
        self.register_buffer(
            "alphas_cumprod_prev",
            torch.cat([torch.FloatTensor([1.0]), self.alphas_cumprod[:-1]]),
            False,
        )
        alphas_cumprod_prev_with_last = torch.cat(
            [torch.FloatTensor([1.0]), self.alphas_cumprod]
        )
        self.register_buffer(
            "sqrt_alphas_cumprod_prev", alphas_cumprod_prev_with_last.sqrt(), False
        )
        self.register_buffer("sqrt_alphas_cumprod", self.alphas_cumprod.sqrt(), False)
        self.register_buffer(
            "sqrt_recip_alphas_cumprod", (1.0 / self.alphas_cumprod).sqrt(), False
        )
        self.register_buffer(
            "sqrt_alphas_cumprod_m1",
            (1.0 - self.alphas_cumprod).sqrt() * self.sqrt_recip_alphas_cumprod,
            False,
        )
        posterior_variance = (
            self.betas * (1 - self.alphas_cumprod_prev) / (1 - self.alphas_cumprod)
        )
        posterior_variance = torch.stack(
            [posterior_variance, torch.FloatTensor([1e-20] * self.max_step)]
        )
        posterior_log_variance_clipped = posterior_variance.max(dim=0).values.log()
        posterior_mean_coef1 = (
            self.betas * self.alphas_cumprod_prev.sqrt() / (1 - self.alphas_cumprod)
        )
        posterior_mean_coef2 = (
            (1 - self.alphas_cumprod_prev)
            * self.alphas.sqrt()
            / (1 - self.alphas_cumprod)
        )
        self.register_buffer(
            "posterior_log_variance_clipped", posterior_log_variance_clipped, False
        )
        self.register_buffer("posterior_mean_coef1", posterior_mean_coef1, False)
        self.register_buffer("posterior_mean_coef2", posterior_mean_coef2, False)

    def forward(self, batch):
        if self.training:
            agg, x, _ = batch
            x = x[
                :, 0, :
            ]  # True appliance load curve (2D Tensor to meet NuWave convention, stand for wav)
            encoding = agg[
                :, 1:, :
            ]  # Timestamp encoding (3D Tensor: [B, 3 (hour, dow, month), values])
            agg = agg[
                :, 0, :
            ]  # Aggregate load curve (2D Tensor to meet NuWave convention, stand for wav_l)

            step = torch.randint(0, self.max_step, (x.shape[0],), device=x.device) + 1
            loss, *_ = self.common_step(x, agg, encoding, step)

            return loss

        else:
            if isinstance(batch, tuple):
                agg, _, _ = batch
            else:
                agg = batch

            encoding = agg[:, 1:, :]
            agg = agg[:, 0, :]

            y = self.sample(agg, encoding, self.infer_step)

            return y.unsqueeze(1)

    def train_one_epoch(self, loader, optimizer, device="cuda"):
        """
        Train model for one epoch

        optimizer not used as it is already define in the model
        """
        self.train()

        total_loss = 0

        for seqs, labels_energy, status in loader:
            seqs, labels_energy, status = (
                torch.Tensor(seqs.float()).to(device),
                torch.Tensor(labels_energy.float()).to(device),
                torch.Tensor(status.float()).to(device),
            )

            self.optimizer.zero_grad()
            loss = self.forward((seqs, labels_energy, status))
            total_loss += loss.item()

            loss.backward()
            self.optimizer.step()

        total_loss = total_loss / len(loader)

        return total_loss

    """
    Diffusion backbone directly taken from NuWave

    Argument for conditioning wav_l (low-resolution sound in NuWave) is
    replaced by agg (aggregate load curve) and encoding (timestamp info),
    the two conditioning variables used for diffusion in DiffNILM.
    """

    def sample_continuous_noise_level(self, step):
        rand = torch.rand_like(step, dtype=torch.float, device=step.device)
        continuous_sqrt_alpha_cumprod = self.sqrt_alphas_cumprod_prev[
            step - 1
        ] * rand + self.sqrt_alphas_cumprod_prev[step] * (1.0 - rand)
        return continuous_sqrt_alpha_cumprod.unsqueeze(-1)

    def q_sample(self, y_0, step=None, noise_level=None, eps=None):
        if noise_level is not None:
            continuous_sqrt_alpha_cumprod = noise_level
        elif step is not None:
            continuous_sqrt_alpha_cumprod = self.sqrt_alphas_cumprod_prev[step]
        assert step is not None or noise_level is not None

        if isinstance(eps, type(None)):
            eps = torch.randn_like(y_0, device=y_0.device)

        outputs = (
            continuous_sqrt_alpha_cumprod * y_0
            + (1.0 - continuous_sqrt_alpha_cumprod**2).sqrt() * eps
        )

        return outputs

    def q_posterior(self, y_0, y, step):
        posterior_mean = (
            self.posterior_mean_coef1[step] * y_0 + self.posterior_mean_coef2[step] * y
        )
        posterior_log_variance_clipped = self.posterior_log_variance_clipped[step]
        return posterior_mean, posterior_log_variance_clipped

    @torch.no_grad()
    def predict_start_from_noise(self, y, t, eps):
        return (
            self.sqrt_recip_alphas_cumprod[t].unsqueeze(-1) * y
            - self.sqrt_alphas_cumprod_m1[t].unsqueeze(-1) * eps
        )

    # t: interger not tensor
    @torch.no_grad()
    def p_mean_variance(self, y, y_down, encoding, t, clip_denoised: bool):
        batch_size = y.shape[0]
        noise_level = self.sqrt_alphas_cumprod_prev[t + 1].repeat(batch_size, 1)
        eps_recon = self.model_backbone(y, y_down, encoding, noise_level)
        y_recon = self.predict_start_from_noise(y, t, eps_recon)
        if clip_denoised:
            y_recon.clamp_(-1.0, 1.0)

        model_mean, posterior_log_variance_clipped = self.q_posterior(y_recon, y, t)
        return model_mean, posterior_log_variance_clipped

    @torch.no_grad()
    def compute_inverse_dynamincs(self, y, y_down, encoding, t, clip_denoised=False):
        model_mean, model_log_variance = self.p_mean_variance(
            y, y_down, encoding, t, clip_denoised
        )
        eps = torch.randn_like(y) if t > 0 else torch.zeros_like(y)

        return model_mean + eps * (0.5 * model_log_variance).exp()

    @torch.no_grad()
    def sample(
        self,
        y_down,
        encoding,
        start_step=None,
        init_noise=True,
        store_intermediate_states=False,
    ):
        batch_size = y_down.shape[0]

        start_step = (
            self.max_step if start_step is None else min(start_step, self.max_step)
        )
        step = torch.tensor(
            [start_step] * batch_size, dtype=torch.long, device=y_down.device
        )

        y_t = (
            torch.randn_like(y_down, device=y_down.device)
            if init_noise
            else self.q_sample(y_down, step=step)
        )
        ys = [y_t]
        t = start_step - 1
        while t >= 0:
            y_t = self.compute_inverse_dynamincs(y_t, y_down, encoding, t)
            ys.append(y_t)
            t -= 1
        return ys if store_intermediate_states else ys[-1]

    def common_step(self, y, y_low, encoding, step):
        noise_level = (
            self.sample_continuous_noise_level(step)
            if self.training
            else self.sqrt_alphas_cumprod_prev[step].unsqueeze(-1)
        )

        eps = torch.randn_like(y, device=y.device)
        y_noisy = self.q_sample(y, noise_level=noise_level, eps=eps)
        eps_recon = self.model_backbone(y_noisy, y_low, encoding, noise_level)
        loss = lognorm(eps_recon, eps)

        return loss, y, y_low, y_noisy, eps, eps_recon

    """
    Optimizer directly init in model with patemeters reported in DiffNILM (copy paste of NuWave as the entire paper)
    """

    def configure_optimizers(
        self, lr=0.00003, eps=1e-9, betas=(0.5, 0.999), weight_decay=0.00
    ):
        opt = torch.optim.Adam(
            self.parameters(), lr=lr, eps=eps, betas=betas, weight_decay=weight_decay
        )

        return opt

    """
    Update Noise schedule buffers according to training or validation mode
    """

    def set_noise_schedule(self, train):
        max_step = self.max_step if train else self.infer_step
        noise_schedule = torch.linspace(1e-6, 0.006, max_step)
        noise_schedule = noise_schedule if train else self.infer_schedule

        self.betas = noise_schedule.type_as(self.betas)
        self.alphas = (1 - self.betas).type_as(self.alphas)

        self.alphas_cumprod = self.alphas.cumprod(dim=0).type_as(self.alphas_cumprod)
        self.alphas_cumprod_prev = (
            torch.cat(
                [
                    torch.FloatTensor([1.0]).type_as(self.alphas_cumprod),
                    self.alphas_cumprod[:-1],
                ]
            )
        ).type_as(self.alphas_cumprod_prev)

        alphas_cumprod_prev_with_last = torch.cat(
            [torch.FloatTensor([1.0]).type_as(self.alphas_cumprod), self.alphas_cumprod]
        )

        self.sqrt_alphas_cumprod_prev = (alphas_cumprod_prev_with_last.sqrt()).type_as(
            self.sqrt_alphas_cumprod_prev
        )

        self.sqrt_alphas_cumprod = (self.alphas_cumprod.sqrt()).type_as(
            self.sqrt_alphas_cumprod
        )
        self.sqrt_recip_alphas_cumprod = ((1.0 / self.alphas_cumprod).sqrt()).type_as(
            self.sqrt_recip_alphas_cumprod
        )
        self.sqrt_alphas_cumprod_m1 = (
            (1.0 - self.alphas_cumprod).sqrt() * self.sqrt_recip_alphas_cumprod
        ).type_as(self.sqrt_alphas_cumprod_m1)

        posterior_variance = (
            self.betas * (1 - self.alphas_cumprod_prev) / (1 - self.alphas_cumprod)
        )
        posterior_variance = torch.stack(
            [posterior_variance, torch.full_like(posterior_variance, 1e-20)]
        )
        posterior_log_variance_clipped = posterior_variance.max(dim=0).values.log()

        posterior_mean_coef1 = (
            self.betas * self.alphas_cumprod_prev.sqrt() / (1 - self.alphas_cumprod)
        )
        posterior_mean_coef2 = (
            (1 - self.alphas_cumprod_prev)
            * self.alphas.sqrt()
            / (1 - self.alphas_cumprod)
        )

        self.posterior_log_variance_clipped = posterior_log_variance_clipped.type_as(
            self.posterior_log_variance_clipped
        )
        self.posterior_mean_coef1 = posterior_mean_coef1.type_as(
            self.posterior_mean_coef1
        )
        self.posterior_mean_coef2 = posterior_mean_coef2.type_as(
            self.posterior_mean_coef2
        )

    def update_buffers(self):
        self.set_noise_schedule(train=self.training)

    def train(self, mode=True):
        super(DiffNILM, self).train(mode)
        self.update_buffers()
