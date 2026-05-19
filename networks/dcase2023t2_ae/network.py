import torch
from torch import nn

class ResidualBlock(nn.Module):
    def __init__(self, dim, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
        )
        self.act = nn.SiLU()

    def forward(self, x):
        return self.act(x + self.net(x))


class AENet(nn.Module):
    def __init__(
        self,
        input_dim,
        block_size,
        num_conditions=0,
        latent_dim=64,
        hidden_dim=512,
        dropout=0.1,
    ):
        super(AENet,self).__init__()
        self.input_dim = input_dim
        self.num_conditions = num_conditions
        self.latent_dim = latent_dim
        self.cov_source = nn.Parameter(torch.zeros(block_size, block_size), requires_grad=False)
        self.cov_target = nn.Parameter(torch.zeros(block_size, block_size), requires_grad=False)
        self.latent_mean_source = nn.Parameter(torch.zeros(num_conditions, latent_dim), requires_grad=False)
        self.latent_mean_target = nn.Parameter(torch.zeros(num_conditions, latent_dim), requires_grad=False)
        precision = torch.eye(latent_dim).unsqueeze(0).repeat(num_conditions, 1, 1)
        self.latent_precision_source = nn.Parameter(precision.clone(), requires_grad=False)
        self.latent_precision_target = nn.Parameter(precision.clone(), requires_grad=False)
        self.latent_count_source = nn.Parameter(torch.zeros(num_conditions), requires_grad=False)
        self.latent_count_target = nn.Parameter(torch.zeros(num_conditions), requires_grad=False)
        self.score_mean = nn.Parameter(torch.zeros(2), requires_grad=False)
        self.score_std = nn.Parameter(torch.ones(2), requires_grad=False)

        cond_dim = num_conditions
        enc_in = self.input_dim + cond_dim
        dec_in = latent_dim + cond_dim

        self.encoder = nn.Sequential(
            nn.Linear(enc_in, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            ResidualBlock(hidden_dim, dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            ResidualBlock(hidden_dim // 2, dropout),
            nn.Linear(hidden_dim // 2, latent_dim),
            nn.LayerNorm(latent_dim),
        )

        self.decoder = nn.Sequential(
            nn.Linear(dec_in, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            ResidualBlock(hidden_dim // 2, dropout),
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            ResidualBlock(hidden_dim, dropout),
            nn.Linear(hidden_dim,self.input_dim)
        )

    def _condition(self, x, condition):
        x = x.view(-1, self.input_dim)
        if self.num_conditions == 0:
            return x, None
        if condition is None:
            condition = x.new_zeros((x.size(0), self.num_conditions))
        condition = condition.to(device=x.device, dtype=x.dtype)
        return torch.cat([x, condition], dim=1), condition

    def forward(self, x, condition=None):
        encoder_input, condition = self._condition(x, condition)
        z = self.encoder(encoder_input)
        if condition is not None:
            z_cond = torch.cat([z, condition], dim=1)
        else:
            z_cond = z
        return self.decoder(z_cond), z
