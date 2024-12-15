from itertools import chain
from typing import Any, Dict, Tuple

import torch
import wandb
from lightning import LightningModule
from torch.nn import functional as F
from torchmetrics import MeanMetric
from torchvision.utils import make_grid
from transformers import SiglipTextModel


class RATGANLitModule(LightningModule):
    """Example of a `LightningModule` for MNIST classification.

    A `LightningModule` implements 8 key methods:

    ```python
    def __init__(self):
    # Define initialization code here.

    def setup(self, stage):
    # Things to setup before each stage, 'fit', 'validate', 'test', 'predict'.
    # This hook is called on every process when using DDP.

    def training_step(self, batch, batch_idx):
    # The complete training step.

    def validation_step(self, batch, batch_idx):
    # The complete validation step.

    def test_step(self, batch, batch_idx):
    # The complete test step.

    def predict_step(self, batch, batch_idx):
    # The complete predict step.

    def configure_optimizers(self):
    # Define and configure optimizers and LR schedulers.
    ```

    Docs:
        https://lightning.ai/docs/pytorch/latest/common/lightning_module.html
    """

    def __init__(
        self,
        generator: torch.nn.Module,
        discriminator_feature_extractor: torch.nn.Module,
        discriminator: torch.nn.Module,
        clip_model: torch.nn.Module,
        optimizer_g: torch.optim.Optimizer,
        optimizer_d: torch.optim.Optimizer,
        compile: bool,
    ) -> None:
        """Initialize a `MNISTLitModule`.

        :param net: The model to train.
        :param optimizer: The optimizer to use for training.
        :param scheduler: The learning rate scheduler to use for training.
        """
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        self.automatic_optimization = False

        self.generator = generator
        self.discriminator = discriminator
        self.discriminator_feature_extractor = discriminator_feature_extractor
        self.clip_model = clip_model

        for param in self.clip_model.parameters():
            param.requires_grad = False
        self.clip_model.eval()

        # loss function
        self.criterion = torch.nn.CrossEntropyLoss()

        # metric objects for calculating and averaging accuracy across batches
        self.train_loss_disc = MeanMetric()
        self.train_loss_disc_gp = MeanMetric()
        self.train_loss_gen = MeanMetric()
        self.train_loss_disc_fake = MeanMetric()
        self.train_loss_disc_real = MeanMetric()
        self.train_loss_disc_mismatch = MeanMetric()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Perform a forward pass through the model `self.net`.

        :param x: A tensor of images.
        :return: A tensor of logits.
        """
        return self.net(x)

    def on_train_start(self) -> None:
        """Lightning hook that is called when training begins."""
        # by default lightning executes validation step sanity checks before training starts,
        # so it's worth to make sure validation metrics don't store results from these checks
        pass

    def generator_step(self, text_emb: torch.Tensor) -> torch.Tensor:
        noise = torch.randn(text_emb.shape[0], 100, device=text_emb.device)
        self.generator.lstm.init_hidden(noise)

        return self.generator(noise, text_emb)

    def training_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        """Perform a single training step on a batch of data from the training set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        :return: A tensor of losses between model predictions and targets.
        """
        images, tokens = batch
        optimizer_g, optimizer_d = self.optimizers()

        with torch.no_grad():
            if isinstance(self.clip_model, SiglipTextModel):
                text_emb = self.clip_model(**tokens)[1].float()
            else:
                text_emb = self.clip_model(**tokens).text_embeds.float()

        # generate fake images first
        fake_images = self.generator_step(text_emb)

        # train discriminator
        real_image_feature = self.discriminator_feature_extractor(images)
        disc_output_real = self.discriminator(real_image_feature, text_emb)
        loss_disc_real = F.relu(1.0 - disc_output_real).mean()

        disc_output_mismatch = self.discriminator(
            real_image_feature.roll(1, dims=0), text_emb
        )
        loss_disc_mismatch = F.relu(1.0 + disc_output_mismatch).mean()

        # detach to avoid backprop through G
        fake_image_feature = self.discriminator_feature_extractor(fake_images.detach())
        disc_output_fake = self.discriminator(fake_image_feature, text_emb)
        loss_disc_fake = F.relu(1.0 + disc_output_fake).mean()

        loss_disc = loss_disc_real + 0.5 * (loss_disc_mismatch + loss_disc_fake)

        optimizer_d.zero_grad()
        self.manual_backward(loss_disc)
        optimizer_d.step()

        # MA-GP ???
        interpolated = images.data.requires_grad_()
        text_emb_interpolated = text_emb.data.requires_grad_()
        features = self.discriminator_feature_extractor(interpolated)
        disc_output_interpolated = self.discriminator(features, text_emb_interpolated)
        gradients = torch.autograd.grad(
            outputs=disc_output_interpolated,
            inputs=(interpolated, text_emb_interpolated),
            grad_outputs=torch.ones(disc_output_interpolated.size()).to(self.device),
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )
        gradients_0 = gradients[0].view(gradients[0].size(0), -1)
        gradients_1 = gradients[1].view(gradients[1].size(0), -1)
        gradients = torch.cat([gradients_0, gradients_1], dim=1)
        gradients_l2_norm = torch.sqrt(torch.sum(gradients**2, dim=1) + 1e-12)
        loss_disc_gp = 2 * (gradients_l2_norm**6).mean()

        optimizer_d.zero_grad()
        self.manual_backward(loss_disc_gp)
        optimizer_d.step()

        # train generator
        fake_image_feature = self.discriminator_feature_extractor(fake_images)
        disc_output_fake = self.discriminator(fake_image_feature, text_emb)
        loss_gen = -disc_output_fake.mean()

        optimizer_g.zero_grad()
        self.manual_backward(loss_gen)
        optimizer_g.step()

        # update and log metrics
        self.train_loss_disc(loss_disc)
        self.train_loss_disc_gp(loss_disc_gp)
        self.train_loss_gen(loss_gen)
        self.train_loss_disc_fake(loss_disc_fake)
        self.train_loss_disc_real(loss_disc_real)
        self.train_loss_disc_mismatch(loss_disc_mismatch)

        self.log_dict(
            {
                "train/loss_disc": self.train_loss_disc,
                "train/loss_disc_gp": self.train_loss_disc_gp,
                "train/loss_gen": self.train_loss_gen,
            },
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        self.log_dict(
            {
                "train/loss_disc_fake": self.train_loss_disc_fake,
                "train/loss_disc_real": self.train_loss_disc_real,
                "train/loss_disc_mismatch": self.train_loss_disc_mismatch,
            },
            on_step=False,
            on_epoch=True,
            prog_bar=False,
        )

    def on_train_epoch_end(self) -> None:
        "Lightning hook that is called when a training epoch ends."
        pass

    def validation_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> None:
        """Perform a single validation step on a batch of data from the validation set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        """
        if batch_idx == 0 and self.global_rank == 0:
            _, tokens = batch
            if isinstance(self.clip_model, SiglipTextModel):
                text_emb = self.clip_model(**tokens)[1].float()
            else:
                text_emb = self.clip_model(**tokens).text_embeds.float()
            fake_images = self.generator_step(text_emb)
            images_to_log = fake_images[:16]
            images_to_log = (
                make_grid(images_to_log, nrow=4, normalize=True)
                .mul(255)
                .add_(0.5)
                .clamp_(0, 255)
            )
            self.logger.experiment.log(
                {
                    "val/sample_images": wandb.Image(
                        images_to_log, caption="Generated Image"
                    )
                }
            )

    def on_validation_epoch_end(self) -> None:
        "Lightning hook that is called when a validation epoch ends."
        pass

    def test_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> None:
        """Perform a single test step on a batch of data from the test set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        """
        pass

    def on_test_epoch_end(self) -> None:
        """Lightning hook that is called when a test epoch ends."""
        pass

    def setup(self, stage: str) -> None:
        """Lightning hook that is called at the beginning of fit (train + validate), validate,
        test, or predict.

        This is a good hook when you need to build models dynamically or adjust something about
        them. This hook is called on every process when using DDP.

        :param stage: Either `"fit"`, `"validate"`, `"test"`, or `"predict"`.
        """
        if self.hparams.compile and stage == "fit":
            self.net = torch.compile(self.net)

    def configure_optimizers(self) -> Dict[str, Any]:
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        Examples:
            https://lightning.ai/docs/pytorch/latest/common/lightning_module.html#configure-optimizers

        :return: A dict containing the configured optimizers and learning-rate schedulers to be used for training.
        """
        optimizer_g = self.hparams.optimizer_g(self.generator.parameters())
        optimizer_d = self.hparams.optimizer_d(
            chain(
                self.discriminator.parameters(),
                self.discriminator_feature_extractor.parameters(),
            )
        )
        return optimizer_g, optimizer_d


if __name__ == "__main__":
    _ = RATGANLitModule(None, None, None, None)
