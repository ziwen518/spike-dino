import torch
import torch.nn as nn
import torch.nn.functional as F


class NTXentLoss(nn.Module):
    """
    Normalized Temperature-scaled Cross Entropy Loss.
    Used in contrastive learning frameworks like SimCLR.
    """

    def __init__(self, temperature=0.5):
        """
        Args:
            temperature: Temperature scaling parameter (default: 0.5)
        """
        super().__init__()
        self.temperature = temperature

    def forward(self, z_i, z_j):
        """
        Args:
            z_i: Embeddings from augmented view 1, shape (batch_size, embedding_dim)
            z_j: Embeddings from augmented view 2, shape (batch_size, embedding_dim)

        Returns:
            loss: NT-Xent loss value
        """
        batch_size = z_i.shape[0]

        # Normalize embeddings
        z_i = F.normalize(z_i, dim=1)
        z_j = F.normalize(z_j, dim=1)

        # Concatenate representations
        representations = torch.cat([z_i, z_j], dim=0)  # (2 * batch_size, embedding_dim)

        # Compute similarity matrix
        similarity_matrix = F.cosine_similarity(
            representations.unsqueeze(1),
            representations.unsqueeze(0),
            dim=2
        )  # (2 * batch_size, 2 * batch_size)

        # Create mask to remove self-similarity
        mask = torch.eye(2 * batch_size, dtype=torch.bool, device=z_i.device)
        similarity_matrix = similarity_matrix.masked_fill(mask, -9e15)

        # Scale by temperature
        similarity_matrix = similarity_matrix / self.temperature

        # Create positive pairs mask
        positives = torch.cat([
            torch.arange(batch_size, 2 * batch_size),
            torch.arange(batch_size)
        ], dim=0).to(z_i.device)

        # Compute loss using cross entropy
        loss = F.cross_entropy(similarity_matrix, positives)

        return loss


class TemporalNTXentLoss(NTXentLoss):
    """
    NT-Xent loss for temporal (spiking) embeddings of shape (T, B, N).

    Two reduction strategies:
        - ``'naive'``: average embeddings across T first, then
          compute a single NT-Xent loss on the resulting (B, N) tensors.
        - ``'temporal'``: compute NT-Xent loss independently at each
          time-step and average the T scalar losses.
    """

    STRATEGIES = ("naive", "temporal")

    def __init__(self, temperature: float = 0.5, strategy: str = "naive"):
        """
        Args:
            temperature: Temperature scaling parameter (default: 0.5).
            strategy:    One of ``'mean_then_loss'`` or ``'loss_then_mean'``
                         (default: ``'loss_then_mean'``).
        """
        if strategy not in self.STRATEGIES:
            raise ValueError(
                f"Unknown strategy '{strategy}'. Choose from {self.STRATEGIES}."
            )
        super().__init__(temperature=temperature)
        self.strategy = strategy

    def forward(self, z_i: torch.Tensor, z_j: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z_i: Embeddings from augmented view 1, shape (T, B, N).
            z_j: Embeddings from augmented view 2, shape (T, B, N).

        Returns:
            loss: Scalar NT-Xent loss.
        """
        if self.strategy == "naive":
            # Average over the time dimension, then compute one NT-Xent loss
            return super().forward(z_i.mean(dim=0), z_j.mean(dim=0))

        # Compute NT-Xent loss at each time-step and average
        T = z_i.shape[0]
        forward_fn = super().forward
        total_loss = sum(forward_fn(z_i[t], z_j[t]) for t in range(T))

        return total_loss / T

