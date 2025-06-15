import torch
import torch.nn as nn

class StatExLayer(nn.Module):
    def __init__(self, prob):
        super(StatExLayer, self).__init__()
        self.prob = prob

    def forward(self, X):
        """
        X: tensor of shape (batch, ..., ...)
        y: tensor of shape (batch, num_classes)
        Returns:
            - during training: (out_X, out_y) with mixup/statistics-exchanged versions
            - during evaluation: (X, y) (unchanged)
        """

        # Reverse the batch ordering for "paired" samples
        X_rev = torch.flip(X, dims=[0])

        # Compute "temporal" statistics exchange (axis=3) 
        mean_dim2     = X.mean(dim=3, keepdim=True)
        std_dim2      = X.std(dim=3, keepdim=True) + 1e-16
        mean_rev_dim2 = X_rev.mean(dim=3, keepdim=True)
        std_rev_dim2  = X_rev.std(dim=3, keepdim=True)

        X_tex = ((X - mean_dim2) / std_dim2) * std_rev_dim2 + mean_rev_dim2


        # Decide which statistic-exchange to use for each sample (batch-wise random 0/1).
        #   Note: in the original code, tf.random.uniform < 0 always yields False, but we replicate that here.
        batch_size = X.size(0)
        device = X.device

        X_ex = X_tex

        # Decide whether to apply mixup/statistics-exchange based on self.prob
        dec_mix = (torch.rand(batch_size, device=device) < self.prob).float()
        dec_mix_X = dec_mix.view(batch_size, *([1] * (X.dim() - 1)))
        out_X = dec_mix_X * X + (1.0 - dec_mix_X) * X_ex
        return out_X, X