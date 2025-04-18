import torch
import torch.nn.functional as F
import numpy as np


class NTXentLoss(torch.nn.Module):
    def __init__(self, temperature=0.5, use_cosine_similarity=True):
        super(NTXentLoss, self).__init__()
        self.temperature = temperature
        self.similarity_function = self._get_similarity_function(use_cosine_similarity)
        self.criterion = torch.nn.CrossEntropyLoss(reduction="sum")

    def _get_similarity_function(self, use_cosine_similarity):
        if use_cosine_similarity:
            self._cosine_similarity = torch.nn.CosineSimilarity(dim=-1)
            return self._cosine_similarity_fn
        else:
            return self._dot_similarity

    def _get_correlated_mask(self, batch_size):
        diag = np.eye(2 * batch_size)
        l1 = np.eye(2 * batch_size, 2 * batch_size, k=-batch_size)
        l2 = np.eye(2 * batch_size, 2 * batch_size, k=batch_size)
        mask = torch.from_numpy((diag + l1 + l2))
        mask = (1 - mask).type(torch.bool)
        return mask

    @staticmethod
    def _dot_similarity(x, y):
        return torch.matmul(x, y.T)

    def _cosine_similarity_fn(self, x, y):
        return self._cosine_similarity(x.unsqueeze(1), y.unsqueeze(0)).squeeze()

    def forward(self, zis, zjs):
        batch_size = zis.size(0)
        device = zis.device

        representations = torch.cat([zjs, zis], dim=0)  # (2N, C)
        similarity_matrix = self.similarity_function(representations, representations)

        # Create mask dynamically
        mask = self._get_correlated_mask(batch_size).to(device)

        # Positive pairs
        l_pos = torch.diag(similarity_matrix, batch_size)
        r_pos = torch.diag(similarity_matrix, -batch_size)
        positives = torch.cat([l_pos, r_pos]).view(2 * batch_size, 1)

        # Negative pairs
        negatives = similarity_matrix[mask].view(2 * batch_size, -1)

        # Combine
        logits = torch.cat([positives, negatives], dim=1)
        logits /= self.temperature

        # Labels
        labels = torch.zeros(2 * batch_size, dtype=torch.long).to(device)

        # Loss
        loss = self.criterion(logits, labels)

        loss = loss / (2 * batch_size)

        return loss


class MultiViewNTXentLoss(torch.nn.Module):
    """Multi-View NT-Xent Loss for contrastive learning.

    Extends the SimCLR NT-Xent loss to support multiple augmented views per image.
    Encourages different views of the same image to have similar embeddings,
    and different images to have dissimilar embeddings.

    Attributes:
        temperature (float): Scaling factor for softmax temperature.
        similarity_function (Callable): Cosine or dot product similarity.
    """

    def __init__(self, temperature=0.5, use_cosine_similarity=True):
        """Initializes the MultiViewNTXentLoss.

        Args:
            temperature (float): Temperature parameter for scaling similarities.
            use_cosine_similarity (bool): Whether to use cosine similarity (True)
                or dot product (False).
        """
        super().__init__()
        self.temperature = temperature
        self.similarity_function = (
            self._cosine_similarity if use_cosine_similarity else self._dot_similarity
        )
        self.criterion = torch.nn.CrossEntropyLoss(reduction="sum")

    def _cosine_similarity(self, x, y):
        """Computes cosine similarity between two sets of vectors.

        Args:
            x (torch.Tensor): Tensor of shape (N, D).
            y (torch.Tensor): Tensor of shape (N, D).

        Returns:
            torch.Tensor: Similarity matrix of shape (N, N).
        """
        x = F.normalize(x, dim=-1)
        y = F.normalize(y, dim=-1)
        return torch.matmul(x, y.T)

    def _dot_similarity(self, x, y):
        """Computes dot product similarity between two sets of vectors.

        Args:
            x (torch.Tensor): Tensor of shape (N, D).
            y (torch.Tensor): Tensor of shape (N, D).

        Returns:
            torch.Tensor: Similarity matrix of shape (N, N).
        """
        return torch.matmul(x, y.T)

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Computes the multi-view contrastive loss.

        Args:
            embeddings (torch.Tensor): Input tensor of shape (B, V, D), where:
                B = number of base images,
                V = number of views per image,
                D = feature dimension.

        Returns:
            torch.Tensor: Scalar loss value.
        """
        B, V, D = embeddings.shape
        device = embeddings.device


        # Flatten to [B * V, D]
        embeddings = embeddings.reshape(B * V, D)

        # Compute pairwise similarity
        sim_matrix = self.similarity_function(embeddings, embeddings)  # [B*V, B*V]

        # Mask self-similarities (diagonal)
        self_mask = torch.eye(B * V, device=device).bool()
        sim_matrix.masked_fill_(self_mask, -9e15)

        # Construct positive mask (same image, different views)
        labels = torch.arange(B, device=device).repeat_interleave(V)  # [0, 0, 1, 1, ...]
        pos_mask = labels.unsqueeze(0) == labels.unsqueeze(1)  # [B*V, B*V]
        pos_mask &= ~self_mask  # remove self-pairs

        # Apply temperature scaling and compute log-probabilities
        log_prob = F.log_softmax(sim_matrix / self.temperature, dim=1)

        # Mean log-likelihood over all positive pairs
        mean_log_prob_pos = (log_prob * pos_mask).sum(1) / pos_mask.sum(1)

        # Final NT-Xent loss
        loss = -mean_log_prob_pos.mean()
        return loss

