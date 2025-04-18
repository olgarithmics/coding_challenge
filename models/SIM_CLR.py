import torch.nn as nn
import torchvision.models as models

class SIM_CLR(nn.Module):
    """SimCLR-style model with encoder, projection head, and optional classifier.

    This model includes:
    - A ResNet-based encoder (pretrained)
    - A 2-layer MLP projector for contrastive learning
    - An optional classifier for downstream tasks (e.g. linear evaluation)

    Attributes:
        encoder (nn.Module): ResNet backbone without the original classification head.
        projector (nn.Sequential): MLP projection head.
        classifier (nn.Linear or None): Optional linear classifier.
        use_classifier (bool): Flag indicating whether classifier is included.
    """

    def __init__(self, base_model='resnet18', out_dim=128, num_classes=3):
        """Initializes the SIM_CLR model.

        Args:
            base_model (str): Backbone ResNet variant (e.g., 'resnet18', 'resnet50').
            out_dim (int): Output dimension of the projection head.
            num_classes (int): Number of output classes for optional classifier. Set to None to disable.
        """
        super().__init__()

        # Load base ResNet model and remove classification head
        resnet = models.__dict__[base_model](pretrained=True)
        num_ftrs = resnet.fc.in_features
        resnet.fc = nn.Identity()  # Remove classification layer

        # Save the encoder
        self.encoder = resnet

        # Define projection head (MLP)
        self.projector = nn.Sequential(
            nn.Linear(num_ftrs, num_ftrs),
            nn.ReLU(),
            nn.Linear(num_ftrs, out_dim)
        )

        # Optional classifier head
        self.use_classifier = num_classes is not None
        if self.use_classifier:
            self.classifier = nn.Linear(num_ftrs, num_classes)

    def forward(self, x, return_logits=True):
        """Forward pass through encoder, projector, and optionally classifier.

        Args:
            x (torch.Tensor): Input tensor of shape (B, 3, H, W).
            return_logits (bool): Whether to return classifier logits (if available).

        Returns:
            dict: A dictionary with keys:
                - 'features': Features from encoder.
                - 'projected': Output of projection head.
                - 'logits': Classifier output (if enabled and requested), else None.
        """
        h = self.encoder(x)           # Backbone features
        z = self.projector(h)         # Projected features
        logits = self.classifier(h) if self.use_classifier and return_logits else None

        return {
            "features": h,
            "projected": z,
            "logits": logits
        }
