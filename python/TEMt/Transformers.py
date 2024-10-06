import torch
from torch import Tensor
import torch.nn.functional as F
import torch.nn as nn
from torch.nn import Transformer

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

import einops
from einops import rearrange
from einops.layers.torch import Rearrange

from .Sensorium import Resnet18Sensorium, OpticalEncoder

class DynamicsEncoder(nn.Module):
    """
    Given a batch of sequences in the form of [BATCH_SIZE, SEQ_LEN, EMBED_DIM], produce a batch of position
    encodings [BATCH_SIZE, SEQ_LEN, ENCODE_DIM] using attractor dynamics.
    
    for each element of a sequence, we produce a set of fast parameters using the activation 
    of a trainable dense network, Wa_t = f(x_t).  we then update the current encoding autoregressively such that
    
    et+1 = sigma(et*Wa_t),
    
    starting with e0 which is also learned (sigma is an activation function).
    
    """
    def __init__(self, embedding_dim: int, h_dim: int, encoding_dim: int = 0):
        super().__init__()
        
        self.encoding_dim = embedding_dim if encoding_dim == 0 else encoding_dim
        
        self.f     = nn.Linear(embedding_dim, h_dim)
        self.f_h   = nn.Linear(h_dim, self.encoding_dim**2)
        self.e0    = nn.Parameter(torch.randn(1, encoding_dim)/10.0)
        self.sigma = nn.Tanh() #best option is probably linear with cutoff
        
        code    = torch.zeros(encoding_dim)
        self.register_buffer("code", code, persistent = False)
        
        
    def forward(self, action_sequence: Tensor):
        """
        the action sequence is a BATCH_SIZE batch of sequences.  Every sequence is of length SEQ_LEN, and 
        each element of the sequence is a vector of EMBED_DIM elements.
        
        action_sequence.shape = [BATCH_SIZE, SEQ_LEN, EMBED_DIM]
        """
        # print("action_sequence_shape:", action_sequence.shape)
        batch_size = action_sequence.shape[0]
        seq_len = action_sequence.shape[1]
        shape = [*action_sequence.shape[:-1], self.encoding_dim, self.encoding_dim]

        flattened = action_sequence.flatten(end_dim=-2) #we probably don't need dropout here
        # print("encoder flattened shape",flattened.shape)
        hidden = torch.tanh(self.f(flattened)) # add another layer to generate Wa_t! *this is what james did.  tanh then linear
        Wa_t = self.f_h(hidden).view(*shape)  # [BATCH_SIZE, SEQ_LEN, ENCODE_DIM, ENCODE_DIM]
                                          # does this undo the flatten operation correctly?  How can I verify it?
        
        # we need to activate recurrently over the sequence.
        # there's probably a more elegant way to do this...
        batch_size = shape[0]
        seq_len = shape[1]
        
        encodings = self.code.repeat(batch_size, seq_len, 1)
        et = self.e0.repeat(batch_size, 1)    #save this # a single entry is [BATCH_SIZE, ENCODE_DIM].  don't need dropout here probably
        encodings[:,0] = et                   # 0'th entry is a special case (the learned bias)

        # We need to figure out how to remove this loop
        for i in range(1, seq_len):
            
            # [BATCH_SIZE, 1, ENCODE_DIM] bmm [BATCH_SIZE, ENCODE_DIM, ENCODE_DIM] 
            # print(i, et.unsqueeze(1).shape, Wa_t[:,i].shape)
            et_next = torch.clamp(et.unsqueeze(1).bmm(Wa_t[:,i]).squeeze(1), min=-5, max=5) 
            encodings[:,i] = et_next #should this be encodings[:,i] = encodings[:,i-1] + et_next ? did i forget the path integration all this time?!!?!? no, you didn't, it's in the linear algebra of the previous line
            et = et_next
        
        return encodings
    
def masked_scaled_dot_product_attention(query: Tensor, key: Tensor, value: Tensor) -> Tensor:
    """
    This is A(Q,K,V) = softmax(QK^T / sqrt(d_k))V where d_k is the number of keys / queries (the sequence length)
    
    because this is operating as part of a network, it must handle batched inputs, so there is an extra dimension
    prepended to all tensors, which is the batch size.  therefore we use BATCHED matrix multiplication, bmm
    
    dims[Q] = dims[K] = [batch_size, sequence_length, d_k]
    dims[V] = [batch_size, sequence_length, d_v]
    
    the result of the softmax is a normalized distribution over the "time" dimension of the sequence.  this
    distribution represents how much of each value in V we "care" about, and the inner product is the resulting 
    weighted sum of all v
    """
    # print("inside MSDPA")
    
    temp = query.bmm(key.transpose(1,2))            # QK^T.  dims[temp] = [batch, seq_len, seq_len]
    scale = query.size(-1) ** 0.5                   # sqrt(d_k)
    
    """
    when you mask, do not  set to 0, set to a large negative value.  this is becuase we want our normalization 
    from softmax to only care about the terms we have already seen.
    """
    mask = torch.ones(*temp.shape).triu()*-1*float("Inf")  #elements that are -inf become zero after softmax. double check causality here
    mask = mask.to(query.device)
    mask = torch.nan_to_num(mask)
    
    attention = F.softmax((temp + mask) / scale, dim=-1)     # save this # dims[attention] = [batch, seq_len] nuance here. we might need to modify softmax for dealing with the mask
    
    attended_inner_product = attention.bmm(value)                             
    
    # print(attention)
    
    return attended_inner_product, attention

class MaskedAttentionHead(nn.Module):
    """
    generates our QKV from sequence of features
    """
    def __init__(self, in_kq_dim: int, in_v_dim: int, out_kq_dim: int, out_v_dim: int = None):
        super().__init__()
        out_v_dim = out_kq_dim if out_v_dim is None else out_v_dim
        self.K = nn.Linear(in_kq_dim, out_kq_dim, bias=False)
        self.Q = nn.Linear(in_kq_dim, out_kq_dim, bias=False)
        self.V = nn.Linear(in_v_dim, out_v_dim, bias=False)
        
    def forward(self, query: Tensor, key: Tensor, value: Tensor):
        # print(query.shape, key.shape, value.shape)
        Q = self.Q(query) # maybe save Q and K?
        K = self.K(key)
        V = self.V(value)
        output, attention = masked_scaled_dot_product_attention(Q, K, V)
        self._attention = attention.detach().clone()
        return output

class MaskedMultiHeadAttention(nn.Module):
    """
    there might be many features we want to attend to independently
    """
    def __init__(self,num_heads: int,  in_kq_dim: int, in_v_dim: int, out_kq_dim: int, out_v_dim: int = None):
        super().__init__()
        out_v_dim = out_kq_dim if out_v_dim is None else out_v_dim
        self.heads = nn.ModuleList([MaskedAttentionHead(in_kq_dim, in_v_dim, out_kq_dim, out_v_dim) for _ in range(num_heads)])
        
        # the output features of QK^T * V don't need to have the same dimensions as the prediction
        # you might need to use information from multiple view to make an accurate prediciton.  the dense layer performs that additional feature synthesis
        self.unifyheads1 = nn.Linear(out_v_dim*num_heads, int(out_v_dim*num_heads/2)) #maybe nonlinearity or additional dense layers here for processing
        self.unifyheads2 = nn.Linear(int(out_v_dim*num_heads/2), int(out_v_dim*num_heads/4))
        self.unifyheads3 = nn.Linear(int(out_v_dim*num_heads/4), out_v_dim)
        
        # self.unifyheads = nn.Linear(out_v_dim*num_heads, out_v_dim)
        
    def forward(self, query: Tensor, key: Tensor, value: Tensor):
        #traditional: linear combination to unify heads, followed by layer norm, followed by dense network with linearities
        # after multhihead attention maybe run this through another deep network to feed to another transformer layer or something else
        uh = F.elu(self.unifyheads1(torch.cat([h(query, key, value) for h in self.heads], dim=-1)))
        uhh = F.elu(self.unifyheads2(uh))
        uhhh = F.elu(self.unifyheads3(uhh)) 
        #uh = F.elu(self.unifyheads(torch.cat([h(query, key, value) for h in self.heads], dim=-1)))
        return uhhh
        
        # out = self.unifyheads(torch.cat([h(query, key, value) for h in self.heads], dim=-1))
        # return out

class TEMt(nn.Module):
    def __init__(self, input_x_dim: int, input_a_dim: int, Xtilde_dim: int, Etilde_dim: int, num_heads: int):
        super().__init__()
        self._num_heads = num_heads
        hidden_dim = 32 #E
        self._Fae = DynamicsEncoder(input_a_dim, hidden_dim, hidden_dim)
        self._Fe = nn.Linear(hidden_dim, Etilde_dim)
        self._Fx  = nn.Linear(input_x_dim, 2*Xtilde_dim)
        self._Fx1 = nn.Linear(2*Xtilde_dim, Xtilde_dim)
        
        self._enorm = nn.LayerNorm(Etilde_dim)

        self._sigma = nn.ReLU()
        
        self._multihead_attention = MaskedMultiHeadAttention(num_heads, Etilde_dim, Xtilde_dim, Etilde_dim, Xtilde_dim)
        self._uhfc1 = nn.Linear(Xtilde_dim, input_x_dim)
        self._xnorm = nn.LayerNorm(input_x_dim)
        self._Xtilde = None
        self._Etilde = None #regularizing the norm of Etilde is another option to prevent exploding
        
    def forward(self, X, A):
        Xtilde = self._sigma(self._Fx(X)) #james didn't need to do anything here because he was using one-hot for X
        Xtilde = torch.sigmoid(self._Fx1(Xtilde))

        E = self._Fae(A)
        # Etilde = self._enorm(self._Fe(E)) #james used a single linear transformation with layer norm for activation
        Etilde = torch.sigmoid(self._Fe(E))

        Xpred = self._multihead_attention(Etilde, Etilde, Xtilde) #should we process these into K Q and V more? probably

        Xpred = torch.tanh(self._xnorm(self._uhfc1(Xpred)))

        self._Xtilde = Xtilde.detach().clone()
        self._Etilde = Etilde.detach().clone()

        return Xpred, Xtilde, Etilde

class EndToEndTEMt(nn.Module):
    def __init__(self, input_x_dim: int, input_a_dim: int, Xtilde_dim: int, Etilde_dim: int, num_heads: int, seq_len: int = 100):
        super().__init__()
        self._num_heads = num_heads
        self._seq_len = seq_len
        self._input_x_dim = input_x_dim
        hidden_dim = 32 #E

        self._coder = Resnet18Sensorium(input_x_dim)
        # self._coder = OpticalEncoder(3,64,64,self._input_x_dim)
        # self._front_flatten = Rearrange('b s c h w -> (b s) c h w')
        # self._front_unflatten = Rearrange('(b s) c h w -> b s c h w', s = self._seq_len)
        # self._code_unflatten = Rearrange('(b s) d -> b s d', s = self._seq_len, d = input_x_dim)
        self._Fae = DynamicsEncoder(input_a_dim, hidden_dim, hidden_dim)
        self._Fe = nn.Linear(hidden_dim, Etilde_dim)
        self._Fx  = nn.Linear(input_x_dim, 2*Xtilde_dim)
        self._Fx1 = nn.Linear(2*Xtilde_dim, Xtilde_dim)
        
        self._enorm = nn.LayerNorm(Etilde_dim)

        self._sigma = nn.ReLU()
        
        self._multihead_attention = MaskedMultiHeadAttention(num_heads, Etilde_dim, Xtilde_dim, Etilde_dim, Xtilde_dim)
        # self._multihead_attention = nn.MultiheadAttention(self._input_x_dim, self._num_heads, batch_first=True)
        self._uhfc1 = nn.Linear(Xtilde_dim, input_x_dim)
        self._xnorm = nn.LayerNorm(input_x_dim)
        self._Xtilde = None
        self._Etilde = None #regularizing the norm of Etilde is another option to prevent exploding
        
    def forward(self, obs, acts):

        seq_len = obs.shape[1]
        obs = rearrange(obs, 'b s c h w -> (b s) c h w')
        X_decoded, codes, logvars = self._coder(obs)
        X_decoded = rearrange(X_decoded, '(b s) c h w -> b s c h w', s = seq_len)
        X = rearrange(codes,'(b s) d -> b s d', s = seq_len, d = self._input_x_dim)
        logvars = rearrange(logvars,'(b s) d -> b s d', s = seq_len, d = self._input_x_dim)

        Xtilde = self._sigma(self._Fx(X)) #james didn't need to do anything here because he was using one-hot for X
        Xtilde = torch.tanh(self._Fx1(Xtilde))

        E = self._Fae(acts)
        # Etilde = self._enorm(self._Fe(E)) #james used a single linear transformation with layer norm for activation
        Etilde = torch.sigmoid(self._Fe(E))

        # mask = Transformer.generate_square_subsequent_mask(self._seq_len).to(device)
        # Xpred, _ = self._multihead_attention(Etilde, Etilde, Xtilde, attn_mask = mask, is_causal=True) #should we process these into K Q and V more? probably
        Xpred = self._multihead_attention(Etilde, Etilde, Xtilde)

        Xpred = torch.tanh(self._xnorm(self._uhfc1(Xpred)))

        self._Xtilde = Xtilde.detach().clone()
        self._Etilde = Etilde.detach().clone()

        return Xpred, Xtilde, Etilde, X_decoded, X, logvars

class ResidualTEMt(nn.Module):
    def __init__(self, input_x_dim: int, input_a_dim: int, Xtilde_dim: int, Etilde_dim: int, num_heads: int):
        super().__init__()
        self._num_heads = num_heads
        hidden_dim = 32
        self._Fae = DynamicsEncoder(input_a_dim, hidden_dim, hidden_dim)
        self._Fe = nn.Linear(hidden_dim, Etilde_dim)
        self._Fx  = nn.Linear(input_x_dim, 2*Xtilde_dim)
        self._Fx1 = nn.Linear(2*Xtilde_dim, Xtilde_dim)
        
        self._enorm = nn.LayerNorm(Etilde_dim)
        self._xnorm = nn.LayerNorm(Xtilde_dim)
        self._xnorm2 = nn.LayerNorm(Xtilde_dim)

        self._sigma = nn.ReLU()
        
        self._multihead_attention = MaskedMultiHeadAttention(num_heads, Etilde_dim, Xtilde_dim, Etilde_dim, Xtilde_dim)
        self._multihead_attention2 = MaskedMultiHeadAttention(num_heads, Etilde_dim, Xtilde_dim, Etilde_dim, Xtilde_dim)
        self._uhfc1 = nn.Linear(Xtilde_dim, input_x_dim)
        self._Xtilde = None
        self._Etilde = None #regularizing the norm of Etilde is another option to prevent exploding
        
    def forward(self, X, A):
        Xtilde = self._sigma(self._Fx(X)) #james didn't need to do anything here because he was using one-hot for X
        Xtilde = torch.tanh(self._Fx1(Xtilde))

        E = self._Fae(A)
        # Etilde = self._enorm(self._Fe(E)) #james used a single linear transformation with layer norm for activation
        Etilde = torch.sigmoid(self._Fe(E))

        Xpred = self._xnorm(self._multihead_attention(Etilde, Etilde, Xtilde) + Xtilde) #should we process these into K Q and V more? probably
        Xpred = self._xnorm2(self._multihead_attention2(Etilde, Etilde, Xpred) + Xpred)

        Xpred = torch.tanh(self._uhfc1(Xpred))

        self._Xtilde = Xtilde.detach().clone()
        self._Etilde = Etilde.detach().clone()

        return Xpred, Xtilde, Etilde