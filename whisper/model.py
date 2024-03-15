import base64
import gzip
from dataclasses import dataclass
from typing import Dict, Iterable, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .decoding import decode as decode_function
from .decoding import detect_language as detect_language_function
from .transcribe import transcribe as transcribe_function


@dataclass
class ModelDimensions:
    n_mels: int
    n_audio_ctx: int
    n_audio_state: int
    n_audio_head: int
    n_audio_layer: int
    n_vocab: int
    n_text_ctx: int
    n_text_state: int
    n_text_head: int
    n_text_layer: int


class LayerNorm(nn.LayerNorm):
    def forward(self, x: Tensor) -> Tensor:
        return super().forward(x.float()).type(x.dtype)


class Linear(nn.Linear):
    def forward(self, x: Tensor) -> Tensor:
        return F.linear(
            x,
            self.weight.to(x.dtype),
            None if self.bias is None else self.bias.to(x.dtype),
        )


class Conv1d(nn.Conv1d):
    def _conv_forward(
        self, x: Tensor, weight: Tensor, bias: Optional[Tensor]
    ) -> Tensor:
        return super()._conv_forward(
            x, weight.to(x.dtype), None if bias is None else bias.to(x.dtype)
        )


# * inputs
#   - length : n_ctx   
#     ('n_audio_ctx': 1500)
#   - channels : n_state  
#     ('n_audio_state': 384)  
#
# * torch.arange(length) = Tensor ([0,1,2,...,length-1]) 
#
# * tensor([0, 1, 2, 3, 4])[:, np.newaxis] = 
# tensor([[0],
#         [1],
#         [2],
#         [3],
#         [4]])
#
# * inv_timescales : shape (1, length)
# * scaled_time : (length, 1) x (1, channels // 2) = (length, channels // 2) = (1500, 192)
#
# * output
#   (length, channels) = (n_audio_ctx, n_audio_state) = (1500, 384)
def sinusoids(length, channels, max_timescale=10000):
    """Returns sinusoids for positional embedding"""
    assert channels % 2 == 0
    log_timescale_increment = np.log(max_timescale) / (channels // 2 - 1)
    inv_timescales = torch.exp(-log_timescale_increment * torch.arange(channels // 2))
    scaled_time = torch.arange(length)[:, np.newaxis] * inv_timescales[np.newaxis, :]
    return torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)], dim=1)


# * n_state 가 입력으로 들어가서 n_state 가 출력으로 나온다. 
class MultiHeadAttention(nn.Module):
    def __init__(self, n_state: int, n_head: int):
        super().__init__()
        self.n_head = n_head
        self.query = Linear(n_state, n_state)
        self.key = Linear(n_state, n_state, bias=False)
        self.value = Linear(n_state, n_state)
        self.out = Linear(n_state, n_state)

    # * inputs
    #   x
    #   xa
    #   mask
    #   kv_cache
    #
    # * self attention 일 때는 x와 self.key/self.value 를 쓰는 것 같다.
    # * cross attention 일 때는 xa와 kv_cache 를 쓰는 것 같다.  
    #
    # * x 는 (*,*,n_state) 의 shape 을 가질 듯. FIXME
    # * xa 도 (*,*,n_state) 의 shape 을 가질 듯. FIXME`
    def forward(
        self,
        x: Tensor,
        xa: Optional[Tensor] = None,
        mask: Optional[Tensor] = None,
        kv_cache: Optional[dict] = None,
    ):
        q = self.query(x)

        if kv_cache is None or xa is None or self.key not in kv_cache:
            # hooks, if installed (i.e. kv_cache is not None), will prepend the cached kv tensors;
            # otherwise, perform key/value projections for self- or cross-attention as usual.
            k = self.key(x if xa is None else xa)
            v = self.value(x if xa is None else xa)
        else:
            # for cross-attention, calculate keys and values once and reuse in subsequent calls.
            k = kv_cache[self.key]
            v = kv_cache[self.value]

        wv, qk = self.qkv_attention(q, k, v, mask)
        return self.out(wv), qk

    def qkv_attention(
        self, q: Tensor, k: Tensor, v: Tensor, mask: Optional[Tensor] = None
    ):
        n_batch, n_ctx, n_state = q.shape
        scale = (n_state // self.n_head) ** -0.25
        q = q.view(*q.shape[:2], self.n_head, -1).permute(0, 2, 1, 3) * scale
        k = k.view(*k.shape[:2], self.n_head, -1).permute(0, 2, 3, 1) * scale
        v = v.view(*v.shape[:2], self.n_head, -1).permute(0, 2, 1, 3)

        qk = q @ k
        if mask is not None:
            qk = qk + mask[:n_ctx, :n_ctx]
        qk = qk.float()

        w = F.softmax(qk, dim=-1).to(q.dtype)
        return (w @ v).permute(0, 2, 1, 3).flatten(start_dim=2), qk.detach()

# * n_state 가 입력으로 들어가서 n_state 가 출력으로 나온다. 
#
# * input 
# - n_state : state 
# - n_head : head 수
# - cross_attention : 기본은 self attention이다. 
class ResidualAttentionBlock(nn.Module):
    def __init__(self, n_state: int, n_head: int, cross_attention: bool = False):
        super().__init__()

        self.attn = MultiHeadAttention(n_state, n_head)
        self.attn_ln = LayerNorm(n_state)

        self.cross_attn = (
            MultiHeadAttention(n_state, n_head) if cross_attention else None
        )
        self.cross_attn_ln = LayerNorm(n_state) if cross_attention else None

        n_mlp = n_state * 4
        self.mlp = nn.Sequential(
            Linear(n_state, n_mlp), nn.GELU(), Linear(n_mlp, n_state)
        )
        self.mlp_ln = LayerNorm(n_state)

    # * inputs
    #   x
    #   xa
    #   mask
    #   kv_cache
    #
    # * self attention 일 때는 x와 self.attn/self.attn_ln 를 쓰는 것 같다.
    # * cross attention 일 때는 xa와 self.cross_attn/self.cross_attn_ln 를 쓰는 것 같다.  
    #
    # * x 는 (*,*,n_state) 의 shape 을 가질 듯. FIXME
    # * xa 도 (*,*,n_state) 의 shape 을 가질 듯. FIXME
    def forward(
        self,
        x: Tensor,
        xa: Optional[Tensor] = None,
        mask: Optional[Tensor] = None,
        kv_cache: Optional[dict] = None,
    ):
        x = x + self.attn(self.attn_ln(x), mask=mask, kv_cache=kv_cache)[0]
        if self.cross_attn:
            x = x + self.cross_attn(self.cross_attn_ln(x), xa, kv_cache=kv_cache)[0]
        x = x + self.mlp(self.mlp_ln(x))
        return x


class AudioEncoder(nn.Module):
    # * AudioEncoder 입력 (값은 tiny.pt 기준)  FIXME 
    #   n_mels : mel band 개수? (1 frame)  80
    #   n_ctx : 입력 embedding 의 size??     1500
    #   n_state : AudiEncoder 출력 size      384   
    #   n_head : Attention head 개수??           6 
    #   n_layer : ResidualAttentionBlock 개수    4
    #
    # * Conv1D에서
    # - channels_in 은 Number of channels in the input image 
    # - channels_out 은 Number of channels produced by the convolution
    #
    # 따라서 self.conv1 은 n_mels (80개 mel band, 즉 1개 frame) 의 입력을 받아서 n_state (크기 384) 의 출력을 내보낸다. n_state 는 positional embedding 의 size 이기도 하다. n_state는 encoder block 의 입력/출력 크기이기도 하다. 
    #
    # https://pytorch.org/docs/stable/generated/torch.nn.Conv1d.html#torch.nn.Conv1d
    #
    # * PyTorch 모델을 구현할 때 모델 내부적으로 가지고 있으면서, 학습은 필요 없는 tensor들을 사용해야 할 때가 있다. register_buffer 는 이 때 사용한다.
    #
    # * n_ctx 는 tiny.pt 모델의 경우 1500 이다. 
    #  tiny.pt 의 경우 전체 dims 는 아래와 같았다. 
    #
    #   'dims': 
    #     {
    #       'n_mels': 80, 'n_vocab': 51865, 'n_audio_ctx': 1500, 'n_audio_state': 384, 'n_audio_head': 6, 'n_audio_layer': 4, 'n_text_ctx': 448, 'n_text_state': 384, 'n_text_head': 6, 'n_text_layer': 4
    #     },     
    def __init__(
        self, n_mels: int, n_ctx: int, n_state: int, n_head: int, n_layer: int
    ):
        super().__init__()
        self.conv1 = Conv1d(n_mels, n_state, kernel_size=3, padding=1)
        self.conv2 = Conv1d(n_state, n_state, kernel_size=3, stride=2, padding=1)
        self.register_buffer("positional_embedding", sinusoids(n_ctx, n_state))

        self.blocks: Iterable[ResidualAttentionBlock] = nn.ModuleList(
            [ResidualAttentionBlock(n_state, n_head) for _ in range(n_layer)]
        )
        self.ln_post = LayerNorm(n_state)

    # * inputs 
    #   x : torch.Tensor, shape = (batch_size, n_mels, n_ctx)
    #   = (batch_size, 80, 1500)
    #
    # * x = F.gelu(self.conv1(x))
    #   x shape = (batch_size, n_state, n_ctx) = (batch_size, 384, 1500)
    #
    # * x = x.permute(0, 2, 1)
    #   x shape = (batch_size, n_ctx, n_state) = (batch_size, 1500, 384)
    #
    # * x.shape[1:] ~ (n_ctx, n_state) 
    #
    # * output
    #   (batch_size, n_ctx, n_state)
    def forward(self, x: Tensor):
        """
        x : torch.Tensor, shape = (batch_size, n_mels, n_ctx)
            the mel spectrogram of the audio
        """
        x = F.gelu(self.conv1(x))
        x = F.gelu(self.conv2(x))
        x = x.permute(0, 2, 1)

        assert x.shape[1:] == self.positional_embedding.shape, "incorrect audio shape"
        x = (x + self.positional_embedding).to(x.dtype)

        for block in self.blocks:
            x = block(x)

        x = self.ln_post(x)
        return x


class TextDecoder(nn.Module):
    # * TextDecoder 입력 (값은 tiny.pt 기준)  FIXME 
    #   n_vocab : vocab size               51865
    #   n_ctx : TextDecoder 출력 token 의 max length로 추정됨.     448
    #   n_state : TextDecoder 출력 size      384   
    #   n_head : Attention head 개수           6 
    #   n_layer : ResidualAttentionBlock 개수    4
    #
    # * self.token_embedding 
    #   vocab 의  크기는 n_vocab 이고, embedding vector 의 크기는 n_state 이다.
    #   https://pytorch.org/docs/stable/generated/torch.nn.Embedding.html
    #
    #   (*) 이 들어가면 (*, embedding_dim) 이 출력된다. 현재 embedding_dim 은 n_state 이다. 
    #
    # * self.positional_embedding 는 위치 정보를 표현하기 위한 학습 가능한 추가적인 임베딩 층
    #
    #  https://heekangpark.github.io/ml-shorts/positional-encoding-vs-positional-embedding
    # 
    # * n_ctx 는 tiny.pt 모델의 경우 448 이다. 
    #  tiny.pt 의 경우 전체 dims 는 아래와 같았다. 
    # 
    #   'dims': 
    #     {
    #       'n_mels': 80, 'n_vocab': 51865, 'n_audio_ctx': 1500, 'n_audio_state': 384, 'n_audio_head': 6, 'n_audio_layer': 4, 'n_text_ctx': 448, 'n_text_state': 384, 'n_text_head': 6, 'n_text_layer': 4
    #     }, 
    def __init__(
        self, n_vocab: int, n_ctx: int, n_state: int, n_head: int, n_layer: int
    ):
        super().__init__()

        self.token_embedding = nn.Embedding(n_vocab, n_state)
        self.positional_embedding = nn.Parameter(torch.empty(n_ctx, n_state))

        self.blocks: Iterable[ResidualAttentionBlock] = nn.ModuleList(
            [
                ResidualAttentionBlock(n_state, n_head, cross_attention=True)
                for _ in range(n_layer)
            ]
        )
        self.ln = LayerNorm(n_state)

        mask = torch.empty(n_ctx, n_ctx).fill_(-np.inf).triu_(1)
        self.register_buffer("mask", mask, persistent=False)

    # * inputs
    # - x : decoder input - the text tokens
    # - xa : encoder 에서 넘어온 encoded audio features. context vector라고도 부름. 이것은 cross attention 을 위해 필요한 듯. 
    #
    # * x 의 shape 은 (batch_size, N)  N 은 decode 된 token 개수. (N <= n_ctx)
    # self.token_embedding(x) 의 shape 은 (batch_size, N, n_state)
    #
    # * xa 의 shape = (batch_size, n_audio_ctx, n_audio_state) = (batch, 15000, 384) 
    #
    # * output
    # Logits: The raw scores produced by the output layer of a neural network before applying an activation function. Logits are real numbers, and there is no constraint on their range.
    #
    # logits 의 shape 은 (N, n_state) x (n_state, vocab_size) = (N, vocab_size)    
    def forward(self, x: Tensor, xa: Tensor, kv_cache: Optional[dict] = None):
        """
        x : torch.LongTensor, shape = (batch_size, <= n_ctx)
            the text tokens
        xa : torch.Tensor, shape = (batch_size, n_audio_ctx, n_audio_state)
            the encoded audio features to be attended on
        """
        offset = next(iter(kv_cache.values())).shape[1] if kv_cache else 0
        x = (
            self.token_embedding(x)
            + self.positional_embedding[offset : offset + x.shape[-1]]
        )
        x = x.to(xa.dtype)

        for block in self.blocks:
            x = block(x, xa, mask=self.mask, kv_cache=kv_cache)

        x = self.ln(x)
        logits = (
            x @ torch.transpose(self.token_embedding.weight.to(x.dtype), 0, 1)
        ).float()

        return logits

# * Whisper 가 최상위 모델이다.
# * dims / encoder / decoder 를 property 로 갖는다. 
# * torchscript 로 만들면 (trace / script 어떤 방식으로 하든...) 아래 함수들은 포함되려나?  아래 함수들은 복잡한 operator 들이 많이 포함되어 있다.  FIXME
#   - delete_language
#   - transcribe
#   - decode
class Whisper(nn.Module):
    def __init__(self, dims: ModelDimensions):
        super().__init__()
        self.dims = dims
        self.encoder = AudioEncoder(
            self.dims.n_mels,
            self.dims.n_audio_ctx,
            self.dims.n_audio_state,
            self.dims.n_audio_head,
            self.dims.n_audio_layer,
        )
        self.decoder = TextDecoder(
            self.dims.n_vocab,
            self.dims.n_text_ctx,
            self.dims.n_text_state,
            self.dims.n_text_head,
            self.dims.n_text_layer,
        )
        # use the last half among the decoder layers for time alignment by default;
        # to use a specific set of heads, see `set_alignment_heads()` below.
        all_heads = torch.zeros(
            self.dims.n_text_layer, self.dims.n_text_head, dtype=torch.bool
        )
        all_heads[self.dims.n_text_layer // 2 :] = True
        self.register_buffer("alignment_heads", all_heads.to_sparse(), persistent=False)

    def set_alignment_heads(self, dump: bytes):
        array = np.frombuffer(
            gzip.decompress(base64.b85decode(dump)), dtype=bool
        ).copy()
        mask = torch.from_numpy(array).reshape(
            self.dims.n_text_layer, self.dims.n_text_head
        )
        self.register_buffer("alignment_heads", mask.to_sparse(), persistent=False)

    # * mel 을 입력으로 받아서 audio_features 를 출력한다
    def embed_audio(self, mel: torch.Tensor):
        return self.encoder(mel)

    # * tokens/audio_features 를 입력으로 받아서 logits 를 출력한다.
    def logits(self, tokens: torch.Tensor, audio_features: torch.Tensor):
        return self.decoder(tokens, audio_features)

    # * mel/tokens 를 입력으로 받아서 logits 를 출력한다.
    def forward(
        self, mel: torch.Tensor, tokens: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        return self.decoder(tokens, self.encoder(mel))

    @property
    def device(self):
        return next(self.parameters()).device

    @property
    def is_multilingual(self):
        return self.dims.n_vocab >= 51865

    @property
    def num_languages(self):
        return self.dims.n_vocab - 51765 - int(self.is_multilingual)

    def install_kv_cache_hooks(self, cache: Optional[dict] = None):
        """
        The `MultiHeadAttention` module optionally accepts `kv_cache` which stores the key and value
        tensors calculated for the previous positions. This method returns a dictionary that stores
        all caches, and the necessary hooks for the key and value projection modules that save the
        intermediate tensors to be reused during later calculations.

        Returns
        -------
        cache : Dict[nn.Module, torch.Tensor]
            A dictionary object mapping the key/value projection modules to its cache
        hooks : List[RemovableHandle]
            List of PyTorch RemovableHandle objects to stop the hooks to be called
        """
        cache = {**cache} if cache is not None else {}
        hooks = []

        def save_to_cache(module, _, output):
            if module not in cache or output.shape[1] > self.dims.n_text_ctx:                   # output.shape[1] > self.dims.n_text_ctx  <= What's this?
                # save as-is, for the first token or cross attention
                cache[module] = output
            else:
                cache[module] = torch.cat([cache[module], output], dim=1).detach()              # This concatenates the tensors cache[module] and output along the second dimension (dim=1)
            return cache[module]

        def install_hooks(layer: nn.Module):
            if isinstance(layer, MultiHeadAttention):
                hooks.append(layer.key.register_forward_hook(save_to_cache))
                hooks.append(layer.value.register_forward_hook(save_to_cache))

        self.decoder.apply(install_hooks)
        return cache, hooks

    detect_language = detect_language_function
    transcribe = transcribe_function
    decode = decode_function
