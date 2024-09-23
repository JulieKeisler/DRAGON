import numpy as np
import torch
import torch.nn as nn
from dragon.search_space import Brick
from dragon.utils.tools import logger

# noinspection PyArgumentList
class SpatialAttention(Brick):
    def __init__(self, input_shape, Nh, d_out, gaussian_init_delta_std=2.0, alpha=46,
                 init="random"):
        super(SpatialAttention, self).__init__(input_shape)
        assert init in ["random", "conv"]
        self.Nh = Nh
        self.F, self.T, self.d_in = input_shape
        self.gaussian_init_delta_std = gaussian_init_delta_std
        if init == "random":
            att_center = torch.zeros(Nh, 1).normal_(0.0, gaussian_init_delta_std) # , generator=torch.manual_seed(1)
            self.attention_center = nn.Parameter(torch.cat([att_center.unsqueeze(0) for _ in range(self.T)], axis=0))
            self.alpha = nn.Parameter(torch.randn(self.T))
            self.W_query = nn.Parameter(torch.randn(self.T, self.F, self.d_in, Nh, 2))
            self.W_key = nn.Parameter(torch.randn(self.T, self.F, self.d_in, Nh, 2))
            self.Wkey_ = nn.Parameter(torch.randn(self.T, 2, 2))
            self.u = nn.Parameter(torch.randn(self.T, self.F, 2))

        elif init == "conv":
            att_center = (torch.arange(Nh) - Nh // 2).float().view(Nh, 1)
            self.attention_center = nn.Parameter(torch.cat([att_center.unsqueeze(0) for _ in range(self.T)], axis=0))
            self.alpha = nn.Parameter(torch.Tensor([alpha for _ in range(self.T)]))
            self.W_query = nn.Parameter(torch.zeros(self.T, self.F, self.d_in, Nh, 2))
            self.W_key = nn.Parameter(torch.zeros(self.T, self.F, self.d_in, Nh, 2))
            self.Wkey_ = nn.Parameter(torch.cat([torch.eye(2).unsqueeze(0) for _ in range(self.T)], axis=0))
            self.u = nn.Parameter(torch.zeros(self.T, self.F, 2))
        relative_indices = torch.arange(self.F).view(1, -1) - torch.arange(self.F).view(-1, 1)
        R = torch.cat([relative_indices.unsqueeze(-1) ** 2, relative_indices.unsqueeze(-1)],
                      dim=-1).float()
        self.R = nn.Parameter(torch.cat([R.unsqueeze(0) for _ in range(self.T)], axis=0))
        self.W = nn.Parameter(torch.randn(self.T, Nh * self.d_in, d_out))
        self.b = nn.Parameter(torch.randn(self.F, self.T, d_out))

    def get_attention_probs(self, X):
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
        bs, F, T, d_in = X.shape
        X = torch.permute(X, (0, 2, 1, 3))
        attention_probs = self.get_attention_probs(X)
        input_values = torch.einsum('btfhl,btld->btfhd', attention_probs, X.float())
        input_values = input_values.contiguous().view(bs, T, F, -1)
        try:
            output_value = torch.einsum('btfd,tdo->bfto', input_values, self.W) + self.b
        except Exception as e:
            logger.info(f'Input: {input_values.shape}, W: {self.W.shape},\nX: {F}, {T}, {d_in}\nself: {self.F}, {self.T}, {self.d_in}, Nh = {self.Nh}')
            raise e
        # output_value = output_value.permute(0, 2, 3, 3)
        return output_value
    
    def modify_operation(self, input_shape):
        
        F, T, d_in = input_shape

        T_diff = T - self.T
        F_diff = F - self.F
        d_in_diff = d_in - self.d_in

        F_sign = F_diff / np.abs(F_diff) if F_diff !=0 else 1
        T_sign = T_diff / np.abs(T_diff) if T_diff !=0 else 1
        d_sign = d_in_diff / np.abs(d_in_diff) if d_in_diff !=0 else 1

        pad_att = (0,0,0,0, int(T_sign * np.ceil(np.abs(T_diff)/2)), int(T_sign * np.floor(np.abs(T_diff))/2))
        self.attention_center.data = nn.functional.pad(self.attention_center, pad_att)
        self.Wkey_.data = nn.functional.pad(self.Wkey_, pad_att)

        self.alpha.data = nn.functional.pad(self.alpha, (int(T_sign * np.ceil(np.abs(T_diff)/2)), int(T_sign * np.floor(np.abs(T_diff))/2)))

        pad_Wq = (0, 0, 0, 0, int(d_sign * np.ceil(np.abs(d_in_diff)/2)), int(d_sign * np.floor(np.abs(d_in_diff))/2),
                  int(F_sign * np.ceil(np.abs(F_diff)/2)), int(F_sign * np.floor(np.abs(F_diff))/2),
                  int(T_sign * np.ceil(np.abs(T_diff)/2)), int(T_sign * np.floor(np.abs(T_diff))/2))
        self.W_query.data = nn.functional.pad(self.W_query, pad_Wq)
        self.W_key.data = nn.functional.pad(self.W_key, pad_Wq)

        pad_u = (0,0, int(F_sign * np.ceil(np.abs(F_diff)/2)), int(F_sign * np.floor(np.abs(F_diff))/2),
                  int(T_sign * np.ceil(np.abs(T_diff)/2)), int(T_sign * np.floor(np.abs(T_diff))/2))
        self.u.data = nn.functional.pad(self.u, pad_u)

        pad_R = (0,0, int(F_sign * np.ceil(np.abs(F_diff)/2)), int(F_sign * np.floor(np.abs(F_diff))/2),
                      int(F_sign * np.ceil(np.abs(F_diff)/2)), int(F_sign * np.floor(np.abs(F_diff))/2),
                      int(T_sign * np.ceil(np.abs(T_diff)/2)), int(T_sign * np.floor(np.abs(T_diff))/2))
        self.R.data = nn.functional.pad(self.R, pad_R)

        pad_WW = (0,0,int(d_sign*np.ceil((self.Nh*np.abs(d_in_diff))/2)), int(d_sign*np.floor((self.Nh*np.abs(d_in_diff))/2)),
                      int(T_sign * np.ceil(np.abs(T_diff)/2)), int(T_sign * np.floor(np.abs(T_diff))/2))
        self.W.data = nn.functional.pad(self.W, pad_WW)

        pad_b = (0,0, int(T_sign * np.ceil(np.abs(T_diff)/2)), int(T_sign * np.floor(np.abs(T_diff))/2),
                      int(F_sign * np.ceil(np.abs(F_diff)/2)), int(F_sign * np.floor(np.abs(F_diff))/2))
        self.b.data = nn.functional.pad(self.b, pad_b)

        self.T = T
        self.F = F
        self.d_in = d_in


    def load_state_dict(self, state_dict, **kwargs):
        T, F, d_in = state_dict['W_query'].shape[:3]
        self.modify_operation((F, T, d_in))
        super(SpatialAttention, self).load_state_dict(state_dict, **kwargs)


class TemporalAttention(Brick):
    # noinspection PyArgumentList
    def __init__(self, input_shape, Nh, d_out, gaussian_init_delta_std=2.0, alpha=46,
                 init="random"):
        super(TemporalAttention, self).__init__(input_shape)
        assert init in ["random", "conv"], logger.error(f"Init shoud be in ['random', 'conv'], got{init} instead")
        self.F, self.T, self.d_in = input_shape
        self.Nh = Nh
        self.gaussian_init_delta_std = gaussian_init_delta_std
        if init == "random":
            att_center = torch.zeros(Nh, 1).normal_(0.0, gaussian_init_delta_std)
            self.attention_center = nn.Parameter(torch.cat([att_center.unsqueeze(0) for _ in range(self.F)], axis=0))
            self.alpha = nn.Parameter(torch.randn(self.F))
            self.W_query = nn.Parameter(torch.randn(self.F, self.T, self.d_in, Nh, 2))
            self.W_key = nn.Parameter(torch.randn(self.F, self.T, self.d_in, Nh, 2))
            self.Wkey_ = nn.Parameter(torch.randn(self.F, 2, 2))
            self.u = nn.Parameter(torch.randn(self.F, self.T, 2))

        elif init == "conv":
            att_center = (torch.arange(Nh) - Nh // 2).float().view(Nh, 1)
            try:
                self.attention_center = nn.Parameter(torch.cat([att_center.unsqueeze(0) for _ in range(self.F)], axis=0))
            except NotImplementedError as e:
                logger.info(f'Nh = {Nh}, F = {self.F}, T = {self.T}, d_in = {self.d_in}, d_out={d_out}, att_center = {att_center}')
                raise e
            self.alpha = nn.Parameter(torch.Tensor([alpha for _ in range(self.F)]))
            self.W_query = nn.Parameter(torch.zeros(self.F, self.T, self.d_in, Nh, 2))
            self.W_key = nn.Parameter(torch.zeros(self.F, self.T, self.d_in, Nh, 2))
            self.Wkey_ = nn.Parameter(torch.cat([torch.eye(2).unsqueeze(0) for _ in range(self.F)], axis=0))
            self.u = nn.Parameter(torch.zeros(self.F, self.T, 2))
        relative_indices = torch.arange(self.T).view(1, -1) - torch.arange(self.T).view(-1, 1)
        R = torch.cat([relative_indices.unsqueeze(-1) ** 2, relative_indices.unsqueeze(-1)],
                      dim=-1).float()
        self.R = nn.Parameter(torch.cat([R.unsqueeze(0) for _ in range(self.F)], axis=0))
        self.W = nn.Parameter(torch.randn(self.F, Nh * self.d_in, d_out))
        self.b = nn.Parameter(torch.randn(self.F, self.T, d_out))

    def get_attention_probs(self, X):
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
        bs, F, T, d_in = X.shape
        attention_probs = self.get_attention_probs(X)
        input_values = torch.einsum('bfthl,bfld->bfthd', attention_probs, X.float())
        input_values = input_values.contiguous().view(bs, F, T, -1)
        output_value = torch.einsum('bftd,fdo->bfto', input_values, self.W) + self.b
        return output_value
    
    def modify_operation(self, input_shape):
        F, T, d_in = input_shape
        T_diff = T - self.T
        F_diff = F - self.F
        d_in_diff = d_in - self.d_in

        F_sign = F_diff / np.abs(F_diff) if F_diff !=0 else 1
        T_sign = T_diff / np.abs(T_diff) if T_diff !=0 else 1
        d_sign = d_in_diff / np.abs(d_in_diff) if d_in_diff !=0 else 1

        pad_att = (0,0,0,0, int(F_sign * np.ceil(np.abs(F_diff)/2)), int(F_sign * np.floor(np.abs(F_diff))/2))
        self.attention_center.data = nn.functional.pad(self.attention_center, pad_att)
        self.Wkey_.data = nn.functional.pad(self.Wkey_, pad_att)

        self.alpha.data = nn.functional.pad(self.alpha, (int(F_sign * np.ceil(np.abs(F_diff)/2)), int(F_sign * np.floor(np.abs(F_diff))/2)))

        pad_Wq = (0, 0, 0, 0, int(d_sign * np.ceil(np.abs(d_in_diff)/2)), int(d_sign * np.floor(np.abs(d_in_diff))/2),
                  int(T_sign * np.ceil(np.abs(T_diff)/2)), int(T_sign * np.floor(np.abs(T_diff))/2),
                  int(F_sign * np.ceil(np.abs(F_diff)/2)), int(F_sign * np.floor(np.abs(F_diff))/2))

        self.W_query.data = nn.functional.pad(self.W_query, pad_Wq)
        self.W_key.data = nn.functional.pad(self.W_key, pad_Wq)

        pad_u = (0,0, int(T_sign * np.ceil(np.abs(T_diff)/2)), int(T_sign * np.floor(np.abs(T_diff))/2),
                  int(F_sign * np.ceil(np.abs(F_diff)/2)), int(F_sign * np.floor(np.abs(F_diff))/2))
        self.u.data = nn.functional.pad(self.u, pad_u)

        pad_R = (0,0, int(T_sign * np.ceil(np.abs(T_diff)/2)), int(T_sign * np.floor(np.abs(T_diff))/2),
                      int(T_sign * np.ceil(np.abs(T_diff)/2)), int(T_sign * np.floor(np.abs(T_diff))/2),
                      int(F_sign * np.ceil(np.abs(F_diff)/2)), int(F_sign * np.floor(np.abs(F_diff))/2))
        self.R.data = nn.functional.pad(self.R, pad_R)

        pad_WW = (0,0,int(d_sign*np.ceil((self.Nh*np.abs(d_in_diff))/2)), int(d_sign*np.floor((self.Nh*np.abs(d_in_diff))/2)),
                      int(F_sign * np.ceil(np.abs(F_diff)/2)), int(F_sign * np.floor(np.abs(F_diff))/2))
        self.W.data = nn.functional.pad(self.W, pad_WW)

        pad_b = (0,0, int(T_sign * np.ceil(np.abs(T_diff)/2)), int(T_sign * np.floor(np.abs(T_diff))/2),
                      int(F_sign * np.ceil(np.abs(F_diff)/2)), int(F_sign * np.floor(np.abs(F_diff))/2))
        self.b.data = nn.functional.pad(self.b, pad_b)

        self.T = T
        self.F = F
        self.d_in = d_in

    def load_state_dict(self, state_dict, **kwargs):
        F, T, d_in = state_dict['W_query'].shape[:3]
        self.modify_operation((F, T, d_in))
        super(TemporalAttention, self).load_state_dict(state_dict, **kwargs)


import torch
import torch.nn as nn

from dragon.search_space.cells import Brick

class Attention1D(Brick):
    def __init__(self, input_shape, Nh, d_out=None, gaussian_init_delta_std=2.0, alpha=46,
                 init="random"):
        super(Attention1D, self).__init__(input_shape)
        assert init in ["random", "conv"]
        self.Nh = Nh
        if len(self.input_shape) > 1:
            self.T, self.d_in = input_shape
        else:
            self.T = input_shape[0]
            self.d_in = 1
        if d_out is None:
            d_out = self.d_in
        self.gaussian_init_delta_std = gaussian_init_delta_std
        relative_indices = torch.arange(self.T).view(1, -1) - torch.arange(self.T).view(-1, 1)
        self.d_k = 2
        if init == "random":
            self.attention_center = nn.Parameter(torch.zeros(Nh, 1).normal_(0.0, gaussian_init_delta_std, generator=torch.manual_seed(1)),
                                                 requires_grad=True)
            self.alpha = nn.Parameter(torch.randn(1), requires_grad=True)
            self.W_query = nn.Linear(self.d_in, 2 * Nh)
            self.W_key = nn.Linear(self.d_in, 2 * Nh)
            self.Wkey_ = nn.Parameter(torch.randn(2, 2), requires_grad=True)
            self.u = nn.Parameter(torch.randn(self.T, 2), requires_grad=True)
            self.R = nn.Parameter(torch.cat([relative_indices.unsqueeze(-1) ** 2, relative_indices.unsqueeze(-1)],
                                            dim=-1).float(), requires_grad=True)


        elif init == "conv":
            self.attention_center = nn.Parameter((torch.arange(Nh) - Nh // 2).float().view(Nh, 1),
                                                 requires_grad=True)
            self.alpha = nn.Parameter(torch.Tensor([alpha]),
                                      requires_grad=True)
            self.W_query = nn.Linear(self.d_in, 2 * Nh)
            self.W_query.weight = nn.init.zeros_(self.W_query.weight)
            self.W_query.bias = nn.init.zeros_(self.W_query.bias)
            self.W_key = nn.Linear(self.d_in, 2 * Nh)
            self.W_key.weight = nn.init.zeros_(self.W_key.weight)
            self.W_key.bias = nn.init.zeros_(self.W_key.bias)
            self.Wkey_ = nn.Parameter(torch.eye(2), requires_grad=True)
            self.u = nn.Parameter(torch.zeros(self.T, 2), requires_grad=True)
            self.R = nn.Parameter(torch.cat([relative_indices.unsqueeze(-1) ** 2, relative_indices.unsqueeze(-1)],
                                            dim=-1).float(), requires_grad=True)

        self.W = nn.Linear(self.Nh*self.d_in, d_out, bias=True)

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
        init_shape = X.shape
        if len(init_shape) < 3:
            X = X.unsqueeze(-1)
        bs, T, d_in = X.shape
        attention_probs = self.get_attention_probs(X)
        input_values = torch.einsum('bthl,bld->bthd', attention_probs, X)
        input_values = input_values.contiguous().view(bs, T, -1)
        #input_values = input_values.permute(0, 2, 1)
        output_value = self.W(input_values)
        if len(init_shape) < 3:
            output_value = output_value.squeeze(-1)
        return output_value
    
    def modify_operation(self, input_shape):
        T, d_in = input_shape
        diff_T = T - self.T
        sign_T = diff_T / np.abs(diff_T) if diff_T !=0 else 1
        pad = (0,0, int(sign_T * np.ceil(np.abs(diff_T)/2)), int(sign_T * np.floor(np.abs(diff_T))/2), int(sign_T * np.ceil(np.abs(diff_T)/2)), int(sign_T * np.floor(np.abs(diff_T))/2))
        self.R.data = nn.functional.pad(self.R, pad)
        pad = (0,0, int(sign_T * np.ceil(np.abs(diff_T)/2)), int(sign_T * np.floor(np.abs(diff_T))/2))
        self.u.data = nn.functional.pad(self.u, pad)

        diff = d_in - self.d_in
        sign = diff / np.abs(diff) if diff !=0 else 1
        pad = (int(sign * np.ceil(np.abs(diff)/2)), int(sign * np.floor(np.abs(diff))/2))
        self.W_query.weight.data = nn.functional.pad(self.W_query.weight, pad)
        self.W_key.weight.data = nn.functional.pad(self.W_key.weight, pad)
        diff = self.Nh*(d_in - self.d_in)
        sign = diff / np.abs(diff) if diff !=0 else 1
        pad = (int(sign * np.ceil(np.abs(diff)/2)), int(sign * np.floor(np.abs(diff))/2))
        self.W.weight.data = nn.functional.pad(self.W.weight, pad)
        self.T = T
        self.d_in = d_in

    def load_state_dict(self, state_dict, **kwargs):
        T = state_dict['u'].shape[0]
        self.modify_operation((T,))
        super(Attention1D, self).load_state_dict(state_dict, **kwargs)


class Attention4DT(Brick):
    def __init__(self, input_shape, Nh, d_out, init):
        super(Attention4DT, self).__init__(input_shape)
        self.T, self.H, self.W, self.d_in = input_shape
        self.t_attn = TemporalAttention(input_shape = (self.H*self.W, self.T, self.d_in), Nh=Nh, d_out=d_out, init=init)
    
    def forward(self, X):
        bs, T, H, W, d_in = X.shape
        X_t = X.permute(0, 2, 3, 1, 4)
        X_t = X_t.reshape(bs, H*W, T, d_in)
        X_t = self.t_attn(X_t)
        X_t = X_t.reshape(bs, H, W, T, -1)
        return X_t.permute(0, 3, 1, 2, 4)
    
    def modify_operation(self, input_shape):
        T, H, W, d_in = input_shape
        input_shape = (H*W, T, d_in)
        self.t_attn.modify_operation(input_shape)

    def load_state_dict(self, state_dict, **kwargs):
        self.t_attn.load_state_dict(state_dict, **kwargs)

class Attention4DH(Brick):
    def __init__(self, input_shape, Nh, d_out, init):
        super(Attention4DH, self).__init__(input_shape)
        self.T, self.H, self.W, self.d_in = input_shape
        self.h_attn = TemporalAttention(input_shape = (self.W*self.T, self.H, self.d_in), Nh=Nh, d_out=d_out, init=init)
    
    def forward(self, X):
        bs, T, H, W, d_in = X.shape
        X_h = X.permute(0, 3, 1, 2, 4)
        X_h = X_h.reshape(bs, W*T, H, d_in)
        X_h = self.h_attn(X_h)
        X_h = X_h.reshape(bs, W, T, H, -1)
        return X_h.permute(0, 2, 3, 1, 4)
    
    def modify_operation(self, input_shape):
        T, H, W, d_in = input_shape
        input_shape = (W*T, H, d_in)
        self.h_attn.modify_operation(input_shape)

    def load_state_dict(self, state_dict, **kwargs):
        self.h_attn.load_state_dict(state_dict, **kwargs)

class Attention4DW(Brick):
    def __init__(self, input_shape, Nh, d_out, init):
        super(Attention4DW, self).__init__(input_shape)
        self.T, self.H, self.W, self.d_in = input_shape
        self.w_attn = TemporalAttention(input_shape = (self.T*self.H, self.W, self.d_in), Nh=Nh, d_out=d_out, init=init)
    
    def forward(self, X):
        bs, T, H, W, d_in = X.shape
        X_w = X.reshape(bs, T*H, W, d_in)
        X_w = self.w_attn(X_w)
        return X_w.reshape(bs, T, H, W, -1)
    
    def modify_operation(self, input_shape):
        T, H, W, d_in = input_shape
        input_shape = (T*H, W, d_in)
        self.w_attn.modify_operation(input_shape)

    def load_state_dict(self, state_dict, **kwargs):
        self.w_attn.load_state_dict(state_dict, **kwargs)

class Attention4DHW(Brick):
    def __init__(self, input_shape, Nh, d_out, init):
        super(Attention4DHW, self).__init__(input_shape)
        self.T, self.H, self.W, self.d_in = input_shape
        self.t_attn = TemporalAttention(input_shape = (self.T, self.H*self.W, self.d_in), Nh=Nh, d_out=d_out, init=init)
    
    def forward(self, X):
        bs, T, H, W, d_in = X.shape
        X_t = X.reshape(bs, T, H*W, d_in)
        X_t = self.t_attn(X_t)
        return X_t.reshape(bs, T, H, W, -1)
    
    def modify_operation(self, input_shape):
        T, H, W, d_in = input_shape
        input_shape = T, H*W, d_in
        self.t_attn.modify_operation(input_shape)

    def load_state_dict(self, state_dict, **kwargs):
        self.t_attn.load_state_dict(state_dict, **kwargs)
