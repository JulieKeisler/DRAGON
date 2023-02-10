import torch
import torch.nn as nn
from evodags.utils.tools import logger

# noinspection PyArgumentList
class SpatialAttention(nn.Module):
    def __init__(self, T, F, d_in, Nh, d_out, gaussian_init_delta_std=2.0, alpha=46,
                 init="random"):
        super(SpatialAttention, self).__init__()
        assert init in ["random", "conv"]
        self.Nh = Nh
        self.gaussian_init_delta_std = gaussian_init_delta_std
        if init == "random":
            att_center = torch.zeros(Nh, 1).normal_(0.0, gaussian_init_delta_std) # , generator=torch.manual_seed(1)
            self.attention_center = nn.Parameter(torch.cat([att_center.unsqueeze(0) for _ in range(T)], axis=0))
            self.alpha = nn.Parameter(torch.randn(T))
            self.W_query = nn.Parameter(torch.randn(T, F, d_in, Nh, 2))
            self.W_key = nn.Parameter(torch.randn(T, F, d_in, Nh, 2))
            self.Wkey_ = nn.Parameter(torch.randn(T, 2, 2))
            self.u = nn.Parameter(torch.randn(T, F, 2))

        elif init == "conv":
            att_center = (torch.arange(Nh) - Nh // 2).float().view(Nh, 1)
            self.attention_center = nn.Parameter(torch.cat([att_center.unsqueeze(0) for _ in range(T)], axis=0))
            self.alpha = nn.Parameter(torch.Tensor([alpha for _ in range(T)]))
            self.W_query = nn.Parameter(torch.zeros(T, F, d_in, Nh, 2))
            self.W_key = nn.Parameter(torch.zeros(T, F, d_in, Nh, 2))
            self.Wkey_ = nn.Parameter(torch.cat([torch.eye(2).unsqueeze(0) for _ in range(T)], axis=0))
            self.u = nn.Parameter(torch.zeros(T, F, 2))
        relative_indices = torch.arange(F).view(1, -1) - torch.arange(F).view(-1, 1)
        R = torch.cat([relative_indices.unsqueeze(-1) ** 2, relative_indices.unsqueeze(-1)],
                      dim=-1).float()
        self.R = nn.Parameter(torch.cat([R.unsqueeze(0) for _ in range(T)], axis=0))
        self.W = nn.Parameter(torch.randn(T, Nh * d_in, d_out))
        self.b = nn.Parameter(torch.randn(F, T, d_out))

    def get_attention_probs(self, X):
        if isinstance(X, list):
            X = sum(X)
        bs, T, F, d_in = X.shape
        query = torch.einsum("btfd,tfdhe->btfhe", X, self.W_query).float()  # X @ W_Q
        key = torch.einsum("btfd,tfdhe->btfhe", X, self.W_key).float()  # X @ W_K
        r_delta = torch.einsum('tde,tfle->tfld', self.Wkey_, self.R)
        centers = torch.cat([torch.ones_like(self.attention_center), -2 * self.attention_center], dim=-1)
        v = torch.einsum("t, tde -> tde", -self.alpha, centers)

        # X @ W_Q @ W_k @ X -> output shape: [bs, F, T, H, T]
        first_attention = torch.einsum("btfhe,btlhe->btfhl", query, key)

        # X @ W_Q @ Wkey @ r
        second_attention = torch.einsum("btfhe,tfle->btfhl", query, r_delta)

        # u @ W_K @ X
        third_attention = torch.einsum('tle,btfhe->btfhl', self.u, key)

        # v @ Wkey @ r
        fourth_attention = torch.einsum('the,tfle->tfhl', v, r_delta)

        attention_scores = first_attention + second_attention + third_attention + fourth_attention
        # Softmax
        attention_probs = nn.Softmax(dim=-1)(attention_scores.view(bs, T, F, self.Nh, -1)).view(bs, T, F, self.Nh, F)
        return attention_probs

    def forward(self, X):
        if isinstance(X, list):
            X = sum(X)
        bs, F, T, d_in = X.shape
        X = torch.permute(X, (0, 2, 1, 3))
        attention_probs = self.get_attention_probs(X)
        input_values = torch.einsum('btfhl,btld->btfhd', attention_probs, X.float())
        input_values = input_values.contiguous().view(bs, T, F, -1)
        output_value = torch.einsum('btfd,tdo->bfto', input_values, self.W) + self.b
        # output_value = output_value.permute(0, 2, 3, 3)
        return output_value


class TemporalAttention(nn.Module):
    # noinspection PyArgumentList
    def __init__(self, T, F, d_in, Nh, d_out, gaussian_init_delta_std=2.0, alpha=46,
                 init="random"):
        super(TemporalAttention, self).__init__()
        assert init in ["random", "conv"], logger.error(f"Init shoud be in ['random', 'conv'], got{init} instead")
        self.Nh = Nh
        self.gaussian_init_delta_std = gaussian_init_delta_std
        if init == "random":
            att_center = torch.zeros(Nh, 1).normal_(0.0, gaussian_init_delta_std)
            self.attention_center = nn.Parameter(torch.cat([att_center.unsqueeze(0) for _ in range(F)], axis=0))
            self.alpha = nn.Parameter(torch.randn(F))
            self.W_query = nn.Parameter(torch.randn(F, T, d_in, Nh, 2))
            self.W_key = nn.Parameter(torch.randn(F, T, d_in, Nh, 2))
            self.Wkey_ = nn.Parameter(torch.randn(F, 2, 2))
            self.u = nn.Parameter(torch.randn(F, T, 2))

        elif init == "conv":
            att_center = (torch.arange(Nh) - Nh // 2).float().view(Nh, 1)
            self.attention_center = nn.Parameter(torch.cat([att_center.unsqueeze(0) for _ in range(F)], axis=0))
            self.alpha = nn.Parameter(torch.Tensor([alpha for _ in range(F)]))
            self.W_query = nn.Parameter(torch.zeros(F, T, d_in, Nh, 2))
            self.W_key = nn.Parameter(torch.zeros(F, T, d_in, Nh, 2))
            self.Wkey_ = nn.Parameter(torch.cat([torch.eye(2).unsqueeze(0) for _ in range(F)], axis=0))
            self.u = nn.Parameter(torch.zeros(F, T, 2))
        relative_indices = torch.arange(T).view(1, -1) - torch.arange(T).view(-1, 1)
        R = torch.cat([relative_indices.unsqueeze(-1) ** 2, relative_indices.unsqueeze(-1)],
                      dim=-1).float()
        self.R = nn.Parameter(torch.cat([R.unsqueeze(0) for _ in range(F)], axis=0))
        self.W = nn.Parameter(torch.randn(F, Nh * d_in, d_out))
        self.b = nn.Parameter(torch.randn(F, T, d_out))

    def get_attention_probs(self, X):
        if isinstance(X, list):
            return sum(X)
        bs, F, T, d_in = X.shape
        query = torch.einsum("bftd,ftdhe->bfthe", X, self.W_query).float()  # X @ W_Q
        key = torch.einsum("bftd,ftdhe->bfthe", X, self.W_key).float()  # X @ W_K
        r_delta = torch.einsum('fde,ftle->ftld', self.Wkey_, self.R)
        centers = torch.cat([torch.ones_like(self.attention_center), -2 * self.attention_center], dim=-1)
        v = torch.einsum("f, fde -> fde", -self.alpha, centers)
        # X @ W_Q @ W_k @ X -> output shape: [bs, F, T, H, T]
        first_attention = torch.einsum("bfthe,bflhe->bfthl", query, key)

        # X @ W_Q @ Wkey @ r
        second_attention = torch.einsum("bfthe,ftle->bfthl", query, r_delta)

        # u @ W_K @ X
        third_attention = torch.einsum('fle,bfthe->bfthl', self.u, key)

        # v @ Wkey @ r
        fourth_attention = torch.einsum('fhe,ftle->fthl', v, r_delta)

        attention_scores = first_attention + second_attention + third_attention + fourth_attention
        # Softmax
        attention_probs = nn.Softmax(dim=-1)(attention_scores.view(bs, F, T, self.Nh, -1)).view(bs, F, T, self.Nh, T)
        return attention_probs

    def forward(self, X):
        if isinstance(X, list):
            X = sum(X)
        bs, F, T, d_in = X.shape
        attention_probs = self.get_attention_probs(X)
        input_values = torch.einsum('bfthl,bfld->bfthd', attention_probs, X.float())
        input_values = input_values.contiguous().view(bs, F, T, -1)
        output_value = torch.einsum('bftd,fdo->bfto', input_values, self.W) + self.b
        return output_value


class Attention1D(nn.Module):
    def __init__(self, T, d_in, Nh, d_out, gaussian_init_delta_std=2.0, alpha=46,
                 init="random"):
        super(Attention1D, self).__init__()
        assert init in ["random", "conv"]
        self.Nh = Nh
        self.gaussian_init_delta_std = gaussian_init_delta_std
        relative_indices = torch.arange(T).view(1, -1) - torch.arange(T).view(-1, 1)
        self.d_k = 2
        if init == "random":
            self.attention_center = nn.Parameter(torch.zeros(Nh, 1).normal_(0.0, gaussian_init_delta_std, generator=torch.manual_seed(1)),
                                                 requires_grad=True)
            self.alpha = nn.Parameter(torch.randn(1), requires_grad=True)
            self.W_query = nn.Linear(d_in, 2 * Nh)
            self.W_key = nn.Linear(d_in, 2 * Nh)
            self.Wkey_ = nn.Parameter(torch.randn(2, 2), requires_grad=True)
            self.u = nn.Parameter(torch.randn(T, 2), requires_grad=True)
            self.R = nn.Parameter(torch.cat([relative_indices.unsqueeze(-1) ** 2, relative_indices.unsqueeze(-1)],
                                            dim=-1).float(), requires_grad=True)


        elif init == "conv":
            self.attention_center = nn.Parameter((torch.arange(Nh) - Nh // 2).float().view(Nh, 1),
                                                 requires_grad=True)
            self.alpha = nn.Parameter(torch.Tensor([alpha]),
                                      requires_grad=True)
            self.W_query = nn.Linear(d_in, 2 * Nh)
            self.W_query.weight = nn.init.zeros_(self.W_query.weight)
            self.W_query.bias = nn.init.zeros_(self.W_query.bias)
            self.W_key = nn.Linear(d_in, 2 * Nh)
            self.W_key.weight = nn.init.zeros_(self.W_key.weight)
            self.W_key.bias = nn.init.zeros_(self.W_key.bias)
            self.Wkey_ = nn.Parameter(torch.eye(2), requires_grad=True)
            self.u = nn.Parameter(torch.zeros(T, 2), requires_grad=True)
            self.R = nn.Parameter(torch.cat([relative_indices.unsqueeze(-1) ** 2, relative_indices.unsqueeze(-1)],
                                            dim=-1).float(), requires_grad=True)

        self.W = nn.Linear(T, d_out, bias=True)

    def get_attention_probs(self, X):
        bs, T, d_in = X.shape
        query = self.W_query(X).view(bs, T, self.Nh, self.d_k)  # X @ W_Q
        key = self.W_key(X).view(bs, T, self.Nh, self.d_k)  # X @ W_K
        r_delta = torch.einsum('de,tle->tld', self.Wkey_, self.R)
        v = - self.alpha * torch.cat([torch.ones_like(self.attention_center), -2 * self.attention_center], dim=-1)

        # X @ W_Q @ W_k @ X
        first_attention = torch.einsum("bthd,blhd->bthl", query, key)

        # X @ W_Q @ Wkey @ r
        second_attention = torch.einsum("bthd,tld->bthl", query, r_delta)

        # u @ W_K @ X
        third_attention = torch.einsum('ld,bthd->blht', self.u, key)

        # v @ Wkey @ r
        fourth_attention = torch.einsum('hd,tld->thl', v, r_delta)

        attention_scores = first_attention + second_attention + third_attention + fourth_attention
        # Softmax
        attention_probs = nn.Softmax(dim=-1)(attention_scores.view(bs, T, self.Nh, -1)).view(bs, T, self.Nh, T)
        return attention_probs

    def forward(self, X):
        if isinstance(X, list):
            X = sum(X)
        init_shape = X.shape
        if len(init_shape) < 3:
            X = X.unsqueeze(-1)
        bs, T, d_in = X.shape
        attention_probs = self.get_attention_probs(X)
        input_values = torch.einsum('bthl,bld->bthd', attention_probs, X)
        input_values = input_values.contiguous().view(bs, T, -1)
        input_values = input_values.permute(0, 2, 1)
        output_value = self.W(input_values)
        if len(init_shape) < 3:
            output_value = output_value.squeeze(-1)
        return output_value
