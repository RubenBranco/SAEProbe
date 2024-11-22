from typing import Union, Optional, Tuple, List, Dict
from dataclasses import dataclass
import math
import yaml
from multiprocessing import cpu_count

import torch
from torch import Tensor
from jaxtyping import Float
from torch import nn
from torch.nn import functional as F
from einops import einsum
import lightning as L


@dataclass
class SAEConfig:
    # SAE Parameters
    input_dim: int
    latent_dim: int
    sparsity_coefficient: float = 5
    expansion_coefficient: Optional[float] = None
    w_dec_l2_norm_init: float = 0.01
    apply_pre_encode_bias: bool = False

    # Training Parameters

    lr: float = 5e-5
    lr_decay: bool = True
    lr_decay_start_percentage: float = 0.8  # last 20% of training
    sparsity_coefficient_increase_end: float = 0.05  # 5% of training
    dead_neuron_threshold: int = 12500
    batch_size: int = 128
    val_percent: float = 0.1

    @classmethod
    def from_yaml(cls, yaml_path: str) -> "SAEConfig":
        """Load configuration from YAML file."""
        with open(yaml_path, "r") as f:
            config_dict = yaml.safe_load(f)
        return cls(**config_dict)

    def save_yaml(self, yaml_path: str) -> None:
        """Save configuration to YAML file."""
        with open(yaml_path, "w") as f:
            yaml.dump(self.__dict__, f, default_flow_style=False)


class SAEPLDataset(L.LightningDataModule):
    def __init__(self, data: Float[Tensor, "... input_dim"], sae_config: SAEConfig):
        super().__init__()
        self.data = data
        self.sae_config = sae_config
        self.scaled = False
        self.train_dataset = None
        self.val_dataset = None

    def setup(self, stage: str) -> None:
        if stage == "fit" and self.train_dataset is None:
            self.data = self.scale_data(self.data)
            # split train / val
            dataset = torch.utils.data.TensorDataset(self.data)
            train, val = torch.utils.data.random_split(
                dataset,
                [
                    int(len(self.data) * (1 - self.sae_config.val_percent)),
                    int(len(self.data) * self.sae_config.val_percent),
                ],
                generator=torch.Generator().manual_seed(42),
            )
            self.train_dataset = train
            self.val_dataset = val

    def scale_data(
        self, data: Float[Tensor, "... input_dim"]
    ) -> Float[Tensor, "... input_dim"]:
        # https://transformer-circuits.pub/2024/april-update/index.html#training-saes
        # The dataset is scaled by a single constant such that E[||x||_2] = sqrt(input_dim)
        scale_factor = math.sqrt(self.data.size(-1)) / torch.mean(
            self.data.norm(p=2, dim=-1)
        )
        return data * scale_factor

    def train_dataloader(self) -> torch.utils.data.DataLoader:
        return torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.sae_config.batch_size,
            shuffle=True,
            num_workers=cpu_count(),
            persistent_workers=True,
        )

    def val_dataloader(self) -> torch.utils.data.DataLoader:
        return torch.utils.data.DataLoader(
            self.val_dataset,
            batch_size=self.sae_config.batch_size,
            shuffle=False,
            num_workers=cpu_count(),
            persistent_workers=True,
        )


class SAE(nn.Module):
    """
    Anthropics Sparse Autoencoder

    My interpretation and implementation as described in:

    https://transformer-circuits.pub/2023/monosemantic-features#appendix-autoencoder
    https://transformer-circuits.pub/2024/april-update/index.html#training-saes
    https://transformer-circuits.pub/2024/scaling-monosemanticity/index.html

    Example of Trainer:
    ```
    trainer = L.Trainer(
        max_steps=100000,
        accelerator="gpu",
        gradient_clip_val=1.0,
        gradient_clip_algorithm="norm",
    )
    ```
    """

    def __init__(
        self,
        input_dim: int,
        latent_dim: Union[int, None],
        sparsity_coefficient: float,
        expansion_coefficient: Optional[float] = None,
        w_dec_l2_norm_init: float = 0.01,
        apply_pre_encode_bias: bool = False,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.expansion_coefficient = expansion_coefficient
        self.sparsity_coefficient = sparsity_coefficient
        self.w_dec_l2_norm_init = w_dec_l2_norm_init
        self.apply_pre_encode_bias = apply_pre_encode_bias

        if latent_dim is None:
            assert (
                expansion_coefficient is not None
            ), "If latent_dim is None, expansion_coefficient must be provided."
            self.latent_dim = int(input_dim * expansion_coefficient)

        self.w_enc = nn.Parameter(torch.empty(latent_dim, input_dim))
        self.b_enc = nn.Parameter(torch.zeros(latent_dim))

        self.w_dec = nn.Parameter(torch.empty(input_dim, latent_dim))
        self.b_dec = nn.Parameter(torch.zeros(input_dim))
        self._init_decoder_weights(w_dec_l2_norm_init)

        self.w_enc.data = self.w_dec.data.clone().t()

    @property
    def feature_directions(self) -> Float[Tensor, "input_dim latent_dim"]:
        # https://transformer-circuits.pub/2024/scaling-monosemanticity/index.html
        # "(...) unit-normalized decoder vectors W_d / ||W_d||_2 as feature vectors or feature directions."
        return F.normalize(self.w_dec, p=2, dim=0)

    def feature_activations(
        self, z: Float[Tensor, "batch latent_dim"]
    ) -> Float[Tensor, "batch latent_dim"]:
        # https://transformer-circuits.pub/2024/april-update/index.html#training-saes
        # "Conceptually a feature's activation is now f_i \dot ||W_d||_2"
        return z * torch.norm(self.w_dec, p=2, dim=0)

    def _init_decoder_weights(self, l2_norm: float) -> None:
        # https://transformer-circuits.pub/2024/april-update/index.html#training-saes
        # "The elements of W_d are initialized such that the columns point in random directions"
        w = torch.randn(self.input_dim, self.latent_dim)

        # "and are normalized to have a fixed L2 norm."
        w = F.normalize(w, p=2, dim=0)
        w *= l2_norm

        self.w_dec.data = w

    def pre_encode_bias(
        self, x: Float[Tensor, "batch input_dim"]
    ) -> Float[Tensor, "batch input_dim"]:
        # https://transformer-circuits.pub/2023/monosemantic-features#appendix-autoencoder
        "As shown, we subtract the decoder bias from the inputs, and call this a pre-encoder bias."
        return x - self.b_dec

    def encode(
        self, x: Float[Tensor, "batch input_dim"]
    ) -> Float[Tensor, "batch latent_dim"]:
        if self.apply_pre_encode_bias:
            x = self.pre_encode_bias(x)

        return F.relu(einsum(x, self.w_enc, "b i, j i -> b j") + self.b_enc)

    def decode(
        self, z: Float[Tensor, "batch latent_dim"]
    ) -> Float[Tensor, "batch input_dim"]:
        return einsum(z, self.w_dec, "b j, i j -> b i") + self.b_dec

    def forward(
        self, x: Float[Tensor, "batch input_dim"]
    ) -> Tuple[Float[Tensor, "batch latent_dim"], Float[Tensor, "batch input_dim"]]:
        z = self.encode(x)
        return z, self.decode(z)
    

class SAEPLModel(L.LightningModule):
    def __init__(self, sae_cfg: SAEConfig):
        super().__init__()
        self.save_hyperparameters(sae_cfg.__dict__)
        self.sae = SAE(
            sae_cfg.input_dim,
            sae_cfg.latent_dim,
            sae_cfg.sparsity_coefficient,
            sae_cfg.expansion_coefficient,
            sae_cfg.w_dec_l2_norm_init,
            sae_cfg.apply_pre_encode_bias,
        )
        self.cur_sparsity_coefficient = 0.0
        self.lr_decay_step_start = None
        self.sparsity_increase_step_end = None

        self.register_buffer(
            "num_steps_since_neurons_fired",
            torch.zeros(sae_cfg.latent_dim, dtype=torch.long),
        )

    @property
    def feature_directions(self) -> Float[Tensor, "input_dim latent_dim"]:
        return self.sae.feature_directions

    @property
    def dead_neurons(self) -> Tensor:
        return (
            self.num_steps_since_neurons_fired > self.hparams["dead_neuron_threshold"]
        )

    def feature_activations(
        self, z: Float[Tensor, "batch latent_dim"]
    ) -> Float[Tensor, "batch latent_dim"]:
        return self.sae.feature_activations(z)

    def on_train_start(self) -> None:
        if self.hparams["lr_decay"]:
            self.lr_decay_step_start = round(
                self.trainer.max_steps * self.hparams["lr_decay_start_percentage"]
            )
        self.sparsity_increase_step_end = round(
            self.trainer.max_steps * self.hparams["sparsity_coefficient_increase_end"]
        )

    def forward(
        self, x: Float[Tensor, "batch input_dim"]
    ) -> Tuple[Float[Tensor, "batch latent_dim"], Float[Tensor, "batch input_dim"]]:
        return self.sae(x)

    def encode(
        self, x: Float[Tensor, "batch input_dim"]
    ) -> Float[Tensor, "batch latent_dim"]:
        return self.sae.encode(x)

    def decode(
        self, z: Float[Tensor, "batch latent_dim"]
    ) -> Float[Tensor, "batch input_dim"]:
        return self.sae.decode(z)

    def increase_sparsity_coefficient(self, step: int) -> None:
        if (
            step >= self.sparsity_increase_step_end
            and self.cur_sparsity_coefficient < self.hparams["sparsity_coefficient"]
        ):
            self.cur_sparsity_coefficient = self.hparams["sparsity_coefficient"]

        if step < self.sparsity_increase_step_end:
            self.cur_sparsity_coefficient = min(
                self.hparams["sparsity_coefficient"],
                (step / self.sparsity_increase_step_end)
                * self.hparams["sparsity_coefficient"],
            )

    def update_neuron_firing(self, z: Float[Tensor, "batch latent_dim"]) -> None:
        with torch.no_grad():
            feat_act = self.feature_activations(z)
            self.num_steps_since_neurons_fired = torch.where(
                (feat_act > 0).float().sum(0) > 0,
                0,
                self.num_steps_since_neurons_fired + 1,
            )
            self.log_dict({"dead_neurons": self.dead_neurons.sum()}, prog_bar=True)

    def firing_rate(
        self, z: Float[Tensor, "batch latent_dim"], log_key: str = ""
    ) -> None:
        fire_rate = (self.feature_activations(z) > 0).float().sum(dim=1).mean(dim=0)
        self.log_dict({f"{log_key}/firing_rate": fire_rate}, prog_bar=True)

    def training_step(self, batch: List[Float[Tensor, "batch input_dim"]], _) -> Tensor:
        x = batch[0]
        z, x_hat = self(x)
        loss = self.calculate_loss(x, z, x_hat, log=True, log_key="train")
        self.increase_sparsity_coefficient(self.global_step)
        # logging coefficients

        self.log_dict(
            {
                "lr": self.optimizers().param_groups[0]["lr"],
                "sparsity_coefficient": self.cur_sparsity_coefficient,
            },
            prog_bar=True,
        )

        self.update_neuron_firing(z)
        self.firing_rate(z, log_key="train")

        return loss

    def calculate_loss(
        self,
        x: Float[Tensor, "batch input_dim"],
        z: Float[Tensor, "batch latent_dim"],
        x_hat: Float[Tensor, "batch input_dim"],
        log: bool = False,
        log_key: str = "",
    ) -> Tensor:
        loss_mse = F.mse_loss(x, x_hat)
        loss_sparsity = self.cur_sparsity_coefficient * (
            z.abs() * self.sae.w_dec.norm(p=2, dim=0)
        )
        loss_sparsity = loss_sparsity.sum(dim=1).mean()
        loss = loss_mse + loss_sparsity

        if log:
            self.log_dict(
                {
                    f"{log_key}/loss": loss,
                    f"{log_key}/loss_mse": loss_mse,
                    f"{log_key}/loss_sparsity": loss_sparsity,
                },
                prog_bar=True,
            )
        return loss

    def validation_step(self, batch, _):
        x = batch[0]
        z, x_hat = self(x)
        loss = self.calculate_loss(x, z, x_hat, log=True, log_key="val")
        return loss

    def test_step(self, batch, _):
        x = batch[0]
        z, x_hat = self(x)
        loss = self.calculate_loss(x, z, x_hat, log=False)
        return loss

    def lr_decay(self, step: int) -> float:
        if self.lr_decay_step_start is None or step < self.lr_decay_step_start:
            return 1.0

        decay_steps = self.trainer.max_steps - self.lr_decay_step_start
        decay_factor = 1.0 - (step - self.lr_decay_step_start) / decay_steps

        return max(decay_factor, 0.0)

    def configure_optimizers(self) -> Tuple[List[torch.optim.Optimizer], List[Dict]]:
        opt = torch.optim.Adam(self.parameters(), lr=self.hparams["lr"], weight_decay=0)
        scheduler = {
            "scheduler": torch.optim.lr_scheduler.LambdaLR(opt, self.lr_decay),
            "interval": "step",
            "frequency": 1,
        }
        return [opt], [scheduler]
