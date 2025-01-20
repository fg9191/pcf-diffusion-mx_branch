import torch

from src.PCF_with_empirical_measure import PCF_with_empirical_measure
from src.trainers.trainer import Trainer
from src.utils.utils import cat_linspace_times

# TODO 12/08/2024 nie_k: Add a way to add a zero at the beginning of a sequence without having to sample it for Swissroll.
# TODO 12/08/2024 nie_k: Alternative plot for swiss roll.

PERIOD_PLOT_VAL = 25


class PCFGANTrainer(Trainer):
    def __init__(
        self,
        generator,
        config,
        learning_rate_gen,
        learning_rate_disc,
        num_D_steps_per_G_step,
        num_samples_pcf,
        hidden_dim_pcf,
        test_metrics_train,
        test_metrics_test,
    ):
        super().__init__(
            test_metrics_train=test_metrics_train,
            test_metrics_test=test_metrics_test,
            num_epochs=config.num_epochs,
            # TODO 14/08/2024 nie_k: technically this is almost correct but would be good to do it properly.
            feature_dim_time_series=config.input_dim - 1,
        )

        # Parameter for pytorch lightning
        self.automatic_optimization = False

        # Training params
        self.config = config
        self.lr_gen = learning_rate_gen
        self.lr_disc = learning_rate_disc

        # Generator Params
        self.generator = generator

        # Discriminator Params
        self.num_samples_pcf = num_samples_pcf
        self.hidden_dim_pcf = hidden_dim_pcf
        self.discriminator = PCF_with_empirical_measure(
            num_samples=self.num_samples_pcf,
            hidden_size=self.hidden_dim_pcf,
            input_size=self.config.input_dim,
        )
        self.D_steps_per_G_step = num_D_steps_per_G_step

        self.output_dir_images = config.exp_dir
        return

    def forward(self, num_seq: int, seq_len: int) -> torch.Tensor:
        # We follow batch first convention. (N,L,D).
        return self.generator(num_seq, seq_len, self.device)

    def augmented_forward(self, num_seq: int, seq_len: int) -> torch.Tensor:
        out = self(
            num_seq=num_seq,
            seq_len=seq_len,
        )
        out = cat_linspace_times(out)
        return out

    def configure_optimizers(self):
        optim_gen = torch.optim.Adam(
            self.generator.parameters(),
            lr=self.lr_gen,
            weight_decay=0,
            betas=(0, 0.9),
        )
        optim_discr = torch.optim.Adam(
            self.discriminator.parameters(), lr=self.lr_disc, weight_decay=0
        )
        return [optim_gen, optim_discr], []

    def training_step(self, batch, batch_nb):
        (targets,) = batch
        optim_gen, optim_discr = self.optimizers()

        loss_gen = self._training_step_gen(optim_gen, targets)

        for i in range(self.D_steps_per_G_step):
            self._training_step_disc(optim_discr, targets)

        # Discriminator and Generator share the same loss so no need to report both.
        self.log(
            name="train_pcfd",
            value=loss_gen,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
        )
        return

    def validation_step(self, batch, batch_nb):
        (targets,) = batch
        fake_samples = self.augmented_forward(
            num_seq=targets.shape[0],
            seq_len=targets.shape[1],
        )
        loss_gen = self.discriminator.distance_measure(
            targets, fake_samples, Lambda=0.1
        )

        self.log(
            name="val_pcfd",
            value=loss_gen,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
        )

        # TODO 11/08/2024 nie_k: A bit of a hack, I usually code this better but will do the trick for now.
        if not (self.current_epoch + 1) % PERIOD_PLOT_VAL:
            path = (
                self.output_dir_images
                + f"pred_vs_true_epoch_{str(self.current_epoch + 1)}"
            )
            self.evaluate(fake_samples, targets, path)
        return

    def _training_step_gen(self, optim_gen, targets: torch.Tensor) -> float:
        optim_gen.zero_grad()

        fake_samples = self.augmented_forward(
            num_seq=targets.shape[0],
            seq_len=targets.shape[1],
        )
        loss_gen = self.discriminator.distance_measure(
            targets, fake_samples, Lambda=0.1
        )

        self.manual_backward(loss_gen)
        optim_gen.step()
        return loss_gen.item()

    def _training_step_disc(self, optim_discr, targets: torch.Tensor) -> float:
        optim_discr.zero_grad()

        with torch.no_grad():
            fake_samples = self.augmented_forward(
                num_seq=targets.shape[0],
                seq_len=targets.shape[1],
            )
        loss_disc = -self.discriminator.distance_measure(
            targets, fake_samples, Lambda=0.1
        )
        self.manual_backward(loss_disc)
        optim_discr.step()

        return loss_disc.item()
