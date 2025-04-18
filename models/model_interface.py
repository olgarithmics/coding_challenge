import inspect
import importlib
from MyOptimizer import create_optimizer
from MyLoss import create_loss
from utils.utils import get_cosine_schedule_with_warmup
import torch
import pytorch_lightning as pl
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
import wandb
from sklearn.preprocessing import StandardScaler
import warnings
from umap.umap_ import UMAP
warnings.filterwarnings("ignore", category=FutureWarning)

class ModelInterface(pl.LightningModule):
    """A PyTorch Lightning module for SimCLR-style multi-view contrastive learning.

    This module supports:
    - Multi-view contrastive loss (NT-Xent)
    - Optional classifier for downstream tasks
    - Training/validation/test steps
    - Periodic t-SNE visualization
    """

    def __init__(self, model, loss, optimizer, **kwargs):
        """Initializes the model, optimizer, and training configs.

        Args:
            model (Namespace): Model configuration including name and hyperparameters.
            loss (str): Loss function name to be created via `create_loss()`.
            optimizer (Namespace): Optimizer configuration.
            **kwargs: Additional arguments like log directory, learning rate, epochs, etc.
        """
        super(ModelInterface, self).__init__()
        self.save_hyperparameters()
        self.load_model()

        self.loss = create_loss(loss)
        self.optimizer = optimizer

        self.n_classes = model.n_classes
        self.log_path = kwargs['log']
        self.epochs = kwargs['epochs']
        self.lr = self.optimizer.lr

    def get_progress_bar_dict(self):
        """Removes 'v_num' (version) from Lightning's progress bar."""
        items = super().get_progress_bar_dict()
        items.pop("v_num", None)
        return items

    def training_step(self, batch, batch_idx):
        """Performs a single training step.

        Args:
            batch (Tuple): A tuple of (images, labels).
            batch_idx (int): Index of the batch.

        Returns:
            dict: Contains 'loss', 'feats', and 'label'.
        """
        images, label = batch
        B, V = images.shape[:2]
        flattened = images.view(B * V, 3, 224, 224)

        embeddings_dict = self.model(flattened)
        embeddings = embeddings_dict['projected']
        feats = embeddings_dict['features']

        embeddings = embeddings.view(V, B, -1).permute(1, 0, 2)
        loss = self.loss(embeddings)

        return {'loss': loss, 'feats': feats, 'label': label}

    def training_epoch_end(self, training_step_outputs):
        """Logs training loss and t-SNE plots every 50 epochs."""
        loss = torch.stack([x['loss'] for x in training_step_outputs])
        feats = torch.cat([x['feats'] for x in training_step_outputs])
        labels = torch.cat([x['label'] for x in training_step_outputs])

        self.log('train_loss', torch.mean(loss), logger=True)
        lr = self.trainer.optimizers[0].param_groups[0]['lr']
        self.log("lr", lr, prog_bar=True)

        num_views = feats.shape[0] // labels.shape[0]
        labels = labels.repeat_interleave(num_views)

        if self.current_epoch % 20 == 0:
            fig = self.plot_features(feats, labels, self.current_epoch, self.n_classes, stage='train')
            if isinstance(self.logger, pl.loggers.WandbLogger):
                self.logger.experiment.log({f"tsne_train_epoch_{self.current_epoch}": wandb.Image(fig), "epoch": self.current_epoch})
            plt.close(fig)

    def validation_step(self, batch, batch_idx):
        """Performs a single validation step."""
        flattened, label = batch
        B, V = flattened.shape[:2]
        flattened = flattened.view(B * V, 3, 224, 224)

        embeddings_dict = self.model(flattened)
        embeddings = embeddings_dict['projected']
        feats = embeddings_dict['features']

        embeddings = embeddings.view(V, B, -1).permute(1, 0, 2)
        loss = self.loss(embeddings)

        return {'feats': feats, 'label': label, 'loss': loss}

    def validation_epoch_end(self, valid_step_outputs):
        """Logs validation loss and t-SNE visualizations."""
        loss = torch.stack([x['loss'] for x in valid_step_outputs])
        feats = torch.cat([x['feats'] for x in valid_step_outputs])
        labels = torch.cat([x['label'] for x in valid_step_outputs])

        self.log('val_loss', torch.mean(loss), logger=True)

        num_views = feats.shape[0] // labels.shape[0]
        labels = labels.repeat_interleave(num_views)

        if self.current_epoch % 20 == 0:
            fig = self.plot_features(feats, labels, self.current_epoch, self.n_classes, stage='val')
            if isinstance(self.logger, pl.loggers.WandbLogger):
                self.logger.experiment.log({f"tsne_val_epoch_{self.current_epoch}": wandb.Image(fig), "epoch": self.current_epoch})
            plt.close(fig)

    def test_step(self, batch, batch_idx):
        """Runs one step during test phase."""
        flattened, label = batch
        embeddings_dict = self.model(flattened)
        feats = embeddings_dict['features']
        return {'feats': feats, 'label': label}

    def test_epoch_end(self, output_results):
        """Logs t-SNE on test data (no augmentation)."""
        feats = torch.cat([x['feats'] for x in output_results])
        labels = torch.cat([x['label'] for x in output_results])

        num_views = feats.shape[0] // labels.shape[0]
        labels = labels.repeat_interleave(num_views)

        epoch = getattr(self, "_epoch_for_plot", 0)

        fig = self.plot_features(feats, labels, epoch=epoch, num_classes=self.n_classes, stage='test')
        self.logger.experiment.log({
            f"tsne_TEST_epoch_{epoch}": wandb.Image(fig),
            "epoch": epoch
        })
        plt.close(fig)

    def configure_optimizers(self):
        """Creates optimizer and cosine learning rate scheduler."""
        optimizer = create_optimizer(self.optimizer, self.model)
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            warmup_epochs=10,
            total_epochs=self.epochs,
            base_lr=self.lr,
            final_lr=self.lr / 50,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
            }
        }

    def plot_features(self, feats, labels, epoch, num_classes=3, stage='val'):
        """Generates a t-SNE plot of features.

        Args:
            feats (Tensor): Feature matrix [N, D]
            labels (Tensor): Corresponding labels [N]
            epoch (int): Epoch number for logging
            num_classes (int): Number of distinct label classes
            stage (str): Dataset stage ('train', 'val', 'test')

        Returns:
            matplotlib.figure.Figure: The t-SNE plot
        """
        if isinstance(feats, torch.Tensor):
            feats = feats.detach().cpu().numpy()
            feats = StandardScaler().fit_transform(feats)
        if isinstance(labels, torch.Tensor):
            labels = labels.detach().cpu().numpy()

        # tsne = TSNE(
        #     n_components=2,
        #     perplexity=40,
        #     learning_rate='auto',
        #     init='random',
        #     random_state=42
        # )
        umap = UMAP(n_neighbors=15, min_dist=0.1, metric='cosine')
        x_feats = umap.fit_transform(feats)
        #x_feats = tsne.fit_transform(feats)

        fig, ax = plt.subplots(figsize=(8, 6))
        for i in range(num_classes):
            mask = labels == i
            ax.scatter(x_feats[mask, 0], x_feats[mask, 1], label=str(i), alpha=0.6)

        ax.legend(title='Classes')
        ax.set_title(f"t-SNE ({stage.capitalize()} - Epoch {epoch})")
        fig.tight_layout()
        return fig

    def load_model(self):
        """Dynamically loads the model class from models/<name>.py."""
        camel_name = self.hparams.model.name
        try:
            Model = getattr(importlib.import_module(f'models.{camel_name}'), camel_name)
        except:
            raise ValueError('Invalid Module File Name or Invalid Class Name!')
        self.model = self.instancialize(Model)

    def instancialize(self, Model, **other_args):
        """Instantiates a model using the hyperparameters dictionary.

        Args:
            Model (class): Model class to be instantiated.
            **other_args: Optional arguments to override hparams.

        Returns:
            nn.Module: Instantiated model.
        """
        class_args = inspect.getargspec(Model.__init__).args[1:]
        inkeys = self.hparams.model.keys()
        args1 = {arg: getattr(self.hparams.model, arg) for arg in class_args if arg in inkeys}
        args1.update(other_args)
        return Model(**args1)

