import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from collections import Counter
import math
from torchtext.data.metrics import bleu_score  # sử dụng torchtext để tính BLEU
from tqdm import tqdm
import re
import os
import numpy as np
from torch.autograd import Variable
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace

# Thiết lập thiết bị
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1. Xử lý dữ liệu
class TranslationDataset(Dataset):
    def __init__(self, src_file, tgt_file, src_tokenizer=None, tgt_tokenizer=None, max_len=100, vocab_size=16000, tokenizer_path='custom_bpe_tokenizer'):
        self.src_file = src_file
        self.tgt_file = tgt_file
        self.max_len = max_len
        self.tokenizer_path = tokenizer_path

        # Đọc và tiền xử lý dữ liệu
        with open(src_file, 'r', encoding='utf-8') as f:
            self.src_sentences = [self.preprocess_sentence(line.strip()) for line in f]
        with open(tgt_file, 'r', encoding='utf-8') as f:
            self.tgt_sentences = [self.preprocess_sentence(line.strip()) for line in f]

        # Huấn luyện hoặc load tokenizer
        if src_tokenizer is not None and tgt_tokenizer is not None:
            self.src_tokenizer = src_tokenizer
            self.tgt_tokenizer = tgt_tokenizer
        else:
            self.src_tokenizer, self.tgt_tokenizer = self.train_tokenizers(vocab_size)

        # Lấy pad_idx từ tokenizer
        self.src_pad_idx = self.src_tokenizer.token_to_id('<pad>')
        self.tgt_pad_idx = self.tgt_tokenizer.token_to_id('<pad>')

        # Tokenize dữ liệu
        self.src_data = [self.tokenize_sentence(sent, self.src_tokenizer) for sent in self.src_sentences]
        self.tgt_data = [self.tokenize_sentence(sent, self.tgt_tokenizer) for sent in self.tgt_sentences]

    def preprocess_sentence(self, sentence):
        # Loại bỏ ký tự đặc biệt, chuẩn hóa khoảng trắng và dấu câu
        sentence = re.sub(r"[\*\"“”\n\\…\+\-\/=\(\)‘•:\[\]\|’\!;]", " ", str(sentence))
        sentence = re.sub(r"[ ]+", " ", sentence)
        sentence = re.sub(r"\!+", "!", sentence)
        sentence = re.sub(r"\,+", ",", sentence)
        sentence = re.sub(r"\?+", "?", sentence)
        return sentence.strip().lower()

    def train_tokenizers(self, vocab_size):
        # Tạo file tạm cho dữ liệu nguồn và đích
        src_data_path = 'src_data.txt'
        tgt_data_path = 'tgt_data.txt'

        # Ghi dữ liệu đã tiền xử lý vào file tạm
        with open(src_data_path, 'w', encoding='utf-8') as src_out:
            src_out.write('\n'.join(self.src_sentences))
        with open(tgt_data_path, 'w', encoding='utf-8') as tgt_out:
            tgt_out.write('\n'.join(self.tgt_sentences))

        # Khởi tạo tokenizer BPE cho nguồn và đích
        src_tokenizer = Tokenizer(BPE(unk_token='<unk>'))
        src_tokenizer.pre_tokenizer = Whitespace()
        tgt_tokenizer = Tokenizer(BPE(unk_token='<unk>'))
        tgt_tokenizer.pre_tokenizer = Whitespace()

        # Huấn luyện tokenizer cho nguồn
        trainer = BpeTrainer(
            vocab_size=vocab_size,
            special_tokens=["<pad>", "<sos>", "<eos>", "<unk>"],
            min_frequency=2
        )
        src_tokenizer.train(files=[src_data_path], trainer=trainer)

        # Huấn luyện tokenizer cho đích
        tgt_tokenizer.train(files=[tgt_data_path], trainer=trainer)

        # Lưu tokenizer
        os.makedirs(self.tokenizer_path, exist_ok=True)
        src_tokenizer.save(f"{self.tokenizer_path}/src_tokenizer.json")
        tgt_tokenizer.save(f"{self.tokenizer_path}/tgt_tokenizer.json")

        # Xóa file tạm
        os.remove(src_data_path)
        os.remove(tgt_data_path)

        return src_tokenizer, tgt_tokenizer

    def tokenize_sentence(self, sentence, tokenizer):
        # Tokenize và mã hóa thành token IDs
        encoding = tokenizer.encode(sentence)
        token_ids = encoding.ids

        token_ids = [tokenizer.token_to_id('<sos>')] + token_ids + [tokenizer.token_to_id('<eos>')]
        if len(token_ids) > self.max_len:
            token_ids = token_ids[:self.max_len]
        token_ids += [tokenizer.token_to_id('<pad>')] * (self.max_len - len(token_ids))

        return token_ids

    def __len__(self):
        return len(self.src_data)

    def __getitem__(self, idx):
        src = self.src_data[idx]
        tgt = self.tgt_data[idx]
        return torch.tensor(src, dtype=torch.long), torch.tensor(tgt, dtype=torch.long)

# 2. Định nghĩa mô hình Transformer



class Embedder(nn.Module):
    def __init__(self, vocab_size, d_model):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        
        self.embed = nn.Embedding(vocab_size, d_model)
        
    def forward(self, x):
        return self.embed(x)

from typing import Optional
class RotaryEmbedding(nn.Module):
    """
    This class implements Rotary Positional Embeddings (RoPE)
    proposed in https://arxiv.org/abs/2104.09864.

    Reference implementation (used for correctness verfication)
    can be found here:
    https://github.com/meta-llama/llama/blob/main/llama/model.py#L80

    In this implementation we cache the embeddings for each position upto
    ``max_seq_len`` by computing this during init.

    Args:
        dim (int): Embedding dimension. This is usually set to the dim of each
            head in the attention module computed as ``embed_dim // num_heads``
        max_seq_len (int): Maximum expected sequence length for the
            model, if exceeded the cached freqs will be recomputed
        base (int): The base for the geometric progression used to compute
            the rotation angles
    """

    def __init__(
        self,
        dim: int,
        max_seq_len: int = 4096,
        base: int = 10_000,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.base = base
        self.max_seq_len = max_seq_len
        self.rope_init()

    def rope_init(self):
        theta = 1.0 / (
            self.base
            ** (torch.arange(0, self.dim, 2)[: (self.dim // 2)].float() / self.dim)
        )
        self.register_buffer("theta", theta, persistent=False)
        self.build_rope_cache(self.max_seq_len)

    def build_rope_cache(self, max_seq_len: int = 4096) -> None:
        # Create position indexes `[0, 1, ..., max_seq_len - 1]`
        seq_idx = torch.arange(
            max_seq_len, dtype=self.theta.dtype, device=self.theta.device
        )

        # Outer product of theta and position index; output tensor has
        # a shape of [max_seq_len, dim // 2]
        idx_theta = torch.einsum("i, j -> ij", seq_idx, self.theta).float()

        # cache includes both the cos and sin components and so the output shape is
        # [max_seq_len, dim // 2, 2]
        cache = torch.stack([torch.cos(idx_theta), torch.sin(idx_theta)], dim=-1)
        self.register_buffer("cache", cache, persistent=False)

    def forward(
        self, x: torch.Tensor, *, input_pos: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): input tensor with shape
                ``[b, s, n_h, h_d]``
            input_pos (Optional[torch.Tensor]): Optional tensor which contains the position ids
                of each token. During training, this is used to indicate the positions
                of each token relative to its sample when packed, shape [b, s].
                During inference, this indicates the position of the current token.
                If none, assume the index of the token is its position id. Default is None.

        Returns:
            torch.Tensor: output tensor with shape ``[b, s, n_h, h_d]``

        Notation used for tensor shapes:
            - b: batch size
            - s: sequence length
            - n_h: num heads
            - h_d: head dim
        """
        # input tensor has shape [b, s, n_h, h_d]
        seq_len = x.size(1)

        # extract the values based on whether input_pos is set or not
        rope_cache = (
            self.cache[:seq_len] if input_pos is None else self.cache[input_pos]
        )

        # reshape input; the last dimension is used for computing the output.
        # Cast to float to match the reference implementation
        # tensor has shape [b, s, n_h, h_d // 2, 2]
        xshaped = x.float().reshape(*x.shape[:-1], -1, 2)

        # reshape the cache for broadcasting
        # tensor has shape [b, s, 1, h_d // 2, 2] if packed samples,
        # otherwise has shape [1, s, 1, h_d // 2, 2]
        rope_cache = rope_cache.view(-1, xshaped.size(1), 1, xshaped.size(3), 2)

        # tensor has shape [b, s, n_h, h_d // 2, 2]
        x_out = torch.stack(
            [
                xshaped[..., 0] * rope_cache[..., 0]
                - xshaped[..., 1] * rope_cache[..., 1],
                xshaped[..., 1] * rope_cache[..., 0]
                + xshaped[..., 0] * rope_cache[..., 1],
            ],
            -1,
        )

        # tensor has shape [b, s, n_h, h_d]
        x_out = x_out.flatten(3)
        return x_out.type_as(x)

# class PositionalEncoder(nn.Module):
#     def __init__(self, d_model, max_seq_length=200, dropout=0.1):
#         super().__init__()
        
#         self.d_model = d_model
#         self.dropout = nn.Dropout(dropout)
        
#         pe = torch.zeros(max_seq_length, d_model)
        
#         # Bảng pe mình vẽ ở trên 
#         for pos in range(max_seq_length):
#             for i in range(0, d_model, 2):
#                 pe[pos, i] = math.sin(pos/(10000**(2*i/d_model)))
#                 pe[pos, i+1] = math.cos(pos/(10000**((2*i+1)/d_model)))
#         pe = pe.unsqueeze(0)        
#         self.register_buffer('pe', pe)
    
#     def forward(self, x):
        
#         x = x*math.sqrt(self.d_model)
#         seq_length = x.size(1)
        
#         pe = Variable(self.pe[:, :seq_length], requires_grad=False)
        
#         if x.is_cuda:
#             pe.cuda()
#         # cộng embedding vector với pe 
#         x = x + pe
#         x = self.dropout(x)
        
#         return x
    
def attention(q, k, v, mask=None, dropout=None):
    """
    q: batch_size x head x seq_length x d_model
    k: batch_size x head x seq_length x d_model
    v: batch_size x head x seq_length x d_model
    mask: batch_size x 1 x 1 x seq_length
    output: batch_size x head x seq_length x d_model
    """

    # attention score được tính bằng cách nhân q với k
    d_k = q.size(-1)
    scores = torch.matmul(q, k.transpose(-2, -1))/math.sqrt(d_k)
    
    if mask is not None:
        mask = mask.unsqueeze(1)
        scores = scores.masked_fill(mask==0, -1e9)
    # xong rồi thì chuẩn hóa bằng softmax
    scores = F.softmax(scores, dim=-1)
    
    if dropout is not None:
        scores = dropout(scores)
    
    output = torch.matmul(scores, v)
    return output, scores


class MultiHeadAttention(nn.Module):
    def __init__(self, heads, d_model, dropout=0.1):
        super().__init__()
        assert d_model % heads == 0
        
        self.d_model = d_model
        self.d_k = d_model//heads
        self.h = heads
        self.attn = None

        # tạo ra 3 ma trận trọng số là q_linear, k_linear, v_linear như hình trên
        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(d_model, d_model)

        self.rope = RotaryEmbedding(self.d_k)
    
    def forward(self, q, k, v, mask=None):
        """
        q: batch_size x seq_length x d_model
        k: batch_size x seq_length x d_model
        v: batch_size x seq_length x d_model
        mask: batch_size x 1 x seq_length
        output: batch_size x seq_length x d_model
        """
        bs = q.size(0)
        # nhân ma trận trọng số q_linear, k_linear, v_linear với dữ liệu đầu vào q, k, v 
        # ở bước encode các bạn lưu ý rằng q, k, v chỉ là một (xem hình trên)
        q = self.q_linear(q).view(bs, -1, self.h, self.d_k)
        k = self.k_linear(k).view(bs, -1, self.h, self.d_k)
        v = self.v_linear(v).view(bs, -1, self.h, self.d_k)

        q = self.rope(q)
        k = self.rope(k)
        
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        # tính attention score
        scores, self.attn = attention(q, k, v, mask, self.dropout)
        
        concat = scores.transpose(1, 2).contiguous().view(bs, -1, self.d_model)
        
        output = self.out(concat)
        return output



class Norm(nn.Module):
    def __init__(self, d_model, eps = 1e-6):
        super().__init__()
    
        self.size = d_model
        
        # create two learnable parameters to calibrate normalisation
        self.alpha = nn.Parameter(torch.ones(self.size))
        self.bias = nn.Parameter(torch.zeros(self.size))
        
        self.eps = eps
    
    def forward(self, x):
        norm = self.alpha * (x - x.mean(dim=-1, keepdim=True)) \
        / (x.std(dim=-1, keepdim=True) + self.eps) + self.bias
        return norm



class FeedForward(nn.Module):
    """ Trong kiến trúc của chúng ta có tầng linear 
    """
    def __init__(self, d_model, d_ff=2048, dropout = 0.1):
        super().__init__() 
    
        # We set d_ff as a default to 2048
        self.linear_1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model)
    
    def forward(self, x):
        x = self.dropout(F.relu(self.linear_1(x)))
        x = self.linear_2(x)
        return x



class EncoderLayer(nn.Module):
    def __init__(self, d_model, heads, dropout=0.1):
        super().__init__()
        self.norm_1 = Norm(d_model)
        self.norm_2 = Norm(d_model)
        self.attn = MultiHeadAttention(heads, d_model, dropout=dropout)
        self.ff = FeedForward(d_model, dropout=dropout)
        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)
        
    def forward(self, x, mask):
        """
        x: batch_size x seq_length x d_model
        mask: batch_size x 1 x seq_length
        output: batch_size x seq_length x d_model
        """
        
        
        x2 = self.norm_1(x)
        # tính attention value, các bạn để ý q, k, v là giống nhau        
        x = x + self.dropout_1(self.attn(x2,x2,x2,mask))
        x2 = self.norm_2(x)
        x = x + self.dropout_2(self.ff(x2))
        return x
    

class DecoderLayer(nn.Module):
    def __init__(self, d_model, heads, dropout=0.1):
        super().__init__()
        self.norm_1 = Norm(d_model)
        self.norm_2 = Norm(d_model)
        self.norm_3 = Norm(d_model)
        
        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)
        self.dropout_3 = nn.Dropout(dropout)
        
        self.attn_1 = MultiHeadAttention(heads, d_model, dropout=dropout)
        self.attn_2 = MultiHeadAttention(heads, d_model, dropout=dropout)
        self.ff = FeedForward(d_model, dropout=dropout)

    def forward(self, x, e_outputs, src_mask, trg_mask):
        """
        x: batch_size x seq_length x d_model
        e_outputs: batch_size x seq_length x d_model
        src_mask: batch_size x 1 x seq_length
        trg_mask: batch_size x 1 x seq_length
        """
        # Các bạn xem hình trên, kiến trúc mình vẽ với code ở chỗ này tương đương nhau.
        x2 = self.norm_1(x)
        # multihead attention thứ nhất, chú ý các từ ở target 
        x = x + self.dropout_1(self.attn_1(x2, x2, x2, trg_mask))
        x2 = self.norm_2(x)
        # masked mulithead attention thứ 2. k, v là giá trị output của mô hình encoder
        x = x + self.dropout_2(self.attn_2(x2, e_outputs, e_outputs, src_mask))
        x2 = self.norm_3(x)
        x = x + self.dropout_3(self.ff(x2))
        return x
    


import copy

def get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

class Encoder(nn.Module):
    """Một encoder có nhiều encoder layer nhé !!!
    """
    def __init__(self, vocab_size, d_model, N, heads, dropout):
        super().__init__()
        self.N = N
        self.embed = Embedder(vocab_size, d_model)
        # self.pe = PositionalEncoder(d_model, dropout=dropout)
        self.layers = get_clones(EncoderLayer(d_model, heads, dropout), N)
        self.norm = Norm(d_model)
        
    def forward(self, src, mask):
        """
        src: batch_size x seq_length
        mask: batch_size x 1 x seq_length
        output: batch_size x seq_length x d_model
        """
        x = self.embed(src)
        # x = self.pe(x)
        for i in range(self.N):
            x = self.layers[i](x, mask)
        return self.norm(x)
    


class Decoder(nn.Module):
    """Một decoder có nhiều decoder layer nhé !!!
    """
    def __init__(self, vocab_size, d_model, N, heads, dropout):
        super().__init__()
        self.N = N
        self.embed = Embedder(vocab_size, d_model)
        # self.pe = PositionalEncoder(d_model, dropout=dropout)
        self.layers = get_clones(DecoderLayer(d_model, heads, dropout), N)
        self.norm = Norm(d_model)
    def forward(self, trg, e_outputs, src_mask, trg_mask):
        """
        trg: batch_size x seq_length
        e_outputs: batch_size x seq_length x d_model
        src_mask: batch_size x 1 x seq_length
        trg_mask: batch_size x 1 x seq_length
        output: batch_size x seq_length x d_model
        """
        x = self.embed(trg)
        # x = self.pe(x)
        for i in range(self.N):
            x = self.layers[i](x, e_outputs, src_mask, trg_mask)
        return self.norm(x)


class Transformer(nn.Module):
    """ Cuối cùng ghép chúng lại với nhau để được mô hình transformer hoàn chỉnh
    """
    def __init__(self, src_vocab, trg_vocab, d_model, N, heads, dropout):
        super().__init__()
        self.encoder = Encoder(src_vocab, d_model, N, heads, dropout)
        self.decoder = Decoder(trg_vocab, d_model, N, heads, dropout)
        self.out = nn.Linear(d_model, trg_vocab)
    def forward(self, src, trg, src_mask, trg_mask):
        """
        src: batch_size x seq_length
        trg: batch_size x seq_length
        src_mask: batch_size x 1 x seq_length
        trg_mask batch_size x 1 x seq_length
        output: batch_size x seq_length x vocab_size
        """
        e_outputs = self.encoder(src, src_mask)
        
        d_output = self.decoder(trg, e_outputs, src_mask, trg_mask)
        output = self.out(d_output)
        return output
    


def nopeak_mask(size, device):
    """Tạo mask được sử dụng trong decoder để lúc dự đoán trong quá trình huấn luyện
     mô hình không nhìn thấy được các từ ở tương lai
    """
    np_mask = np.triu(np.ones((1, size, size)),
    k=1).astype('uint8')
    np_mask =  Variable(torch.from_numpy(np_mask) == 0)
    np_mask = np_mask.to(device)
    
    return np_mask

def create_masks(src, trg, src_pad, trg_pad, device):
    """ Tạo mask cho encoder, 
    để mô hình không bỏ qua thông tin của các kí tự PAD do chúng ta thêm vào 
    """
    src_mask = (src != src_pad).unsqueeze(-2)

    if trg is not None:
        trg_mask = (trg != trg_pad).unsqueeze(-2)
        size = trg.size(1) # get seq_len for matrix
        np_mask = nopeak_mask(size, device)
        if trg.is_cuda:
            np_mask.cuda()
        trg_mask = trg_mask & np_mask
        
    else:
        trg_mask = None
    return src_mask, trg_mask


class ScheduledOptim():
    '''A simple wrapper class for learning rate scheduling'''

    def __init__(self, optimizer, init_lr, d_model, n_warmup_steps):
        self._optimizer = optimizer
        self.init_lr = init_lr
        self.d_model = d_model
        self.n_warmup_steps = n_warmup_steps
        self.n_steps = 0


    def step_and_update_lr(self):
        "Step with the inner optimizer"
        self._update_learning_rate()
        self._optimizer.step()


    def zero_grad(self):
        "Zero out the gradients with the inner optimizer"
        self._optimizer.zero_grad()


    def _get_lr_scale(self):
        d_model = self.d_model
        n_steps, n_warmup_steps = self.n_steps, self.n_warmup_steps
        return (d_model ** -0.5) * min(n_steps ** (-0.5), n_steps * n_warmup_steps ** (-1.5))

    def state_dict(self):
        optimizer_state_dict = {
            'init_lr':self.init_lr,
            'd_model':self.d_model,
            'n_warmup_steps':self.n_warmup_steps,
            'n_steps':self.n_steps,
            '_optimizer':self._optimizer.state_dict(),
        }
        
        return optimizer_state_dict
    
    def load_state_dict(self, state_dict):
        self.init_lr = state_dict['init_lr']
        self.d_model = state_dict['d_model']
        self.n_warmup_steps = state_dict['n_warmup_steps']
        self.n_steps = state_dict['n_steps']
        
        self._optimizer.load_state_dict(state_dict['_optimizer'])
        
    def _update_learning_rate(self):
        ''' Learning rate scheduling per step '''

        self.n_steps += 1
        lr = self.init_lr * self._get_lr_scale()

        for param_group in self._optimizer.param_groups:
            param_group['lr'] = lr


class LabelSmoothingLoss(nn.Module):
    def __init__(self, classes, padding_idx, smoothing=0.0, dim=-1):
        super(LabelSmoothingLoss, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.cls = classes
        self.dim = dim
        self.padding_idx = padding_idx

    def forward(self, pred, target):
        pred = pred.log_softmax(dim=self.dim)
        with torch.no_grad():
            # true_dist = pred.data.clone()
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (self.cls - 2))
            true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
            true_dist[:, self.padding_idx] = 0
            mask = torch.nonzero(target.data == self.padding_idx, as_tuple=False)
            if mask.dim() > 0:
                true_dist.index_fill_(0, mask.squeeze(), 0.0)
            
        return torch.mean(torch.sum(-true_dist * pred, dim=self.dim))


import random

def train_dev_split(dataset, dev_ratio=0.05, seed=42):
    """
    Chia TranslationDataset thành train và dev, cả hai đều giữ nguyên class và thuộc tính.

    Args:
        dataset: TranslationDataset gốc đã load dữ liệu và vocab
        dev_ratio: tỷ lệ phần trăm dùng cho dev
        seed: seed để reproducibility

    Returns:
        train_ds, dev_ds: hai TranslationDataset con
    """
    total = len(dataset)
    indices = list(range(total))
    random.seed(seed)
    random.shuffle(indices)

    dev_size = int(total * dev_ratio)
    dev_idx = set(indices[:dev_size])
    train_idx = set(indices[dev_size:])

    # Tạo đối tượng mới không gọi __init__
    def subset_ds(ds, idx_set):
        sub = ds.__class__.__new__(ds.__class__)
        # Copy các thuộc tính chung
        sub.src_file = ds.src_file
        sub.tgt_file = ds.tgt_file
        sub.max_len = ds.max_len
        sub.tokenizer_path = ds.tokenizer_path
        sub.src_tokenizer = ds.src_tokenizer
        sub.tgt_tokenizer = ds.tgt_tokenizer
        sub.src_pad_idx = ds.src_pad_idx
        sub.tgt_pad_idx = ds.tgt_pad_idx
        # sub.src_vocab_size = ds.src_vocab_size
        # sub.tgt_vocab_size = ds.tgt_vocab_size
        # Subset sentences và data
        sub.src_sentences = [ds.src_sentences[i] for i in idx_set]
        sub.tgt_sentences = [ds.tgt_sentences[i] for i in idx_set]
        sub.src_data = [ds.src_data[i] for i in idx_set]
        sub.tgt_data = [ds.tgt_data[i] for i in idx_set]
        return sub

    train_ds = subset_ds(dataset, train_idx)
    dev_ds = subset_ds(dataset, dev_idx)
    return train_ds, dev_ds

# 5. Beam search cho dịch (đã tối ưu)
from dataclasses import dataclass, field
import heapq
import torch.nn.functional as F

@dataclass(order=True)
class Beam:
    score: float
    indices: list = field(compare=False)
    tgt_tensor: torch.Tensor = field(compare=False)

def translate_sentence_beam_search(model, src_ids, dataset, beam_width=3, max_len=100):
    """
    Beam search translation optimized with batch processing and precomputation.
    """
    model.eval()
    device = next(model.parameters()).device

    # Precompute constants
    sos_idx = dataset.tgt_tokenizer.token_to_id('<sos>')
    eos_idx = dataset.tgt_tokenizer.token_to_id('<eos>')
    pad_idx = dataset.src_pad_idx

    # Prepare source tensor
    # src_ids: (1, seq_len)
    src_ids = torch.tensor(src_ids, dtype=torch.long).to(device)
    src_ids = src_ids.unsqueeze(0).to(device)
    # src_mask: (1, 1, seq_len)
    src_mask = (src_ids != pad_idx).unsqueeze(-2)
    # e_output: (1, seq_len, d_model)
    e_output = model.encoder(src_ids, src_mask)

    # Initialize beams
    beams = [Beam(0.0, [sos_idx], torch.tensor([[sos_idx]], dtype=torch.long, device=device))]
    completed = []

    with torch.no_grad():
        for step in range(2, max_len + 1):
            # Prepare batch for all beams
            # trg_mask: (1, step, step)
            trg_mask = nopeak_mask(step - 1, device)
            # tgt_tensor: (len(beams), step - 1)
            tgt_tensors = torch.cat([b.tgt_tensor for b in beams], dim=0)

            # Decoder forward pass for all beams
            decoder_output = model.decoder(
                tgt_tensors, 
                e_output.expand(len(beams), -1, -1), 
                src_mask.expand(len(beams), -1, -1), 
                trg_mask
            )
            logits = model.out(decoder_output[:, -1, :])  # Only last token
            log_probs = F.log_softmax(logits, dim=-1)

            # Get top-k candidates for each beam
            all_beams = []
            topk = log_probs.topk(beam_width, dim=1)
            for i, b in enumerate(beams):
                if b.indices[-1] == eos_idx:
                    completed.append(b)
                    continue
                for logp, idx in zip(topk.values[i], topk.indices[i]):
                    new_score = b.score + logp.item()
                    new_indices = b.indices + [idx.item()]
                    new_tgt = torch.cat([b.tgt_tensor, idx.view(1, 1)], dim=1)
                    all_beams.append(Beam(new_score, new_indices, new_tgt))

            # Select top beams with length normalization
            beams = heapq.nlargest(beam_width, all_beams, key=lambda x: x.score / len(x.indices))
            if not beams:
                break

    # Select best result
    best = max(completed or beams, key=lambda x: x.score / len(x.indices))
    toks = [dataset.tgt_tokenizer.id_to_token(i) for i in best.indices[1:] if i != eos_idx]
    return ' '.join(toks)

# 6. Đánh giá BLEU bằng torchtext

def evaluate_bleu(model, dataset):
    model.eval()
    hyps, refs = [], []
    for i in tqdm(range(len(dataset)), desc=f"[Dev BLEU]"):
        src_ids = dataset.src_data[i]
        ref_sentence = dataset.tgt_sentences[i]  # Câu tham chiếu đã tiền xử lý
        # Tokenize câu tham chiếu bằng tgt_tokenizer
        ref_tokens = dataset.tgt_tokenizer.encode(ref_sentence).tokens
        hyp_sentence = translate_sentence_beam_search(model, src_ids, dataset)
        hyp_tokens = hyp_sentence.split()
        if i < 5:
            print("src_ids: ", src_ids)
            print("hyp_tokens: ", hyp_tokens)
            print("ref_tokens: ", ref_tokens)
        hyps.append(hyp_tokens)
        refs.append([ref_tokens])  # torchtext expects list of references per hypothesis
    # torchtext BLEU trả về điểm 0-1, nhân 100 để ra phần trăm
    bleu = bleu_score(hyps, refs) * 100
    return bleu, hyps, refs

def save_checkpoint(model, optimizer, epoch, path):
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch,
    }
    torch.save(checkpoint, path)
    print(f"Saved checkpoint: {path}")

def load_checkpoint(model, optimizer, path, device):
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    
    print(f"Loaded checkpoint from {path} at epoch {epoch}")
    return epoch

# 7. Hàm chính Hàm chính
if __name__ == '__main__':
    train_src, train_tgt = 'train2023-40k.vi', 'train2023-40k.lo'
    test_src, test_tgt = 'test2023.vi', 'test2023.lo'

    # Prepare datasets
    full_dataset = TranslationDataset(train_src, train_tgt)

    # In ra 10 dòng đầu của train set (đã segment)
    print("First 10 src_data (token IDs):")
    for idx, ids in enumerate(full_dataset.src_data[:10], 1):
        print(f"{idx}: {ids}")
    print()

    print("First 10 src_sentences (tokenized):")
    for idx, tokens in enumerate(full_dataset.src_sentences[:10], 1):
        print(f"{idx}: {tokens}")
    print()

    print("First 10 tgt_sentences (tokenized):")
    for idx, tokens in enumerate(full_dataset.tgt_sentences[:10], 1):
        print(f"{idx}: {tokens}")
    print()

    print("First 10 tgt_data (token IDs):")
    for idx, ids in enumerate(full_dataset.tgt_data[:10], 1):
        print(f"{idx}: {ids}")
    print()



    opt = {
        'batchsize':64,
        'device':'cuda',
        'd_model': 512,
        'n_layers': 6,
        'heads': 8,
        'dropout': 0.1,
        'lr':0.001,
        'epochs':30,
        'printevery': 10,
        'k':1,
        'dev_ratio': 0.01,
        'seed': 42,
    }


    # Split train/dev within training function
    model = Transformer(full_dataset.src_tokenizer.get_vocab_size(), full_dataset.tgt_tokenizer.get_vocab_size(), opt['d_model'], opt['n_layers'], opt['heads'], opt['dropout']).to(device)

    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    
    train_ds, dev_ds = train_dev_split(full_dataset, dev_ratio=opt['dev_ratio'], seed=opt['seed'])
    train_loader = DataLoader(train_ds, batch_size=opt['batchsize'], shuffle=True)
    dev_loader = DataLoader(dev_ds, batch_size=opt['batchsize'])

    optimizer = ScheduledOptim(
        torch.optim.Adam(model.parameters(), betas=(0.9, 0.98), eps=1e-09),
        0.2, opt['d_model'], 4000)
    # scheduler = WarmupScheduler(optimizer, d_model=model.d_model)
    # criterion = nn.CrossEntropyLoss(ignore_index=dataset.pad_idx, label_smoothing=0.1)
    criterion = LabelSmoothingLoss(full_dataset.tgt_tokenizer.get_vocab_size(), padding_idx=full_dataset.src_pad_idx, smoothing=0.1)

    start_epoch = 1
    checkpoint_path = 'checkpoints/latest_checkpoint.pt'
    if os.path.exists(checkpoint_path):
        start_epoch = load_checkpoint(model, optimizer, checkpoint_path, device)
        start_epoch += 1

    for epoch in range(start_epoch, opt['epochs']+1):
        # Train
        model.train()
        total_train_loss = 0
        for src, tgt in tqdm(train_loader, desc=f"Epoch {epoch} [Train]"):
            src, tgt = src.to(device), tgt.to(device)
            tgt_input = tgt[:, :-1]
            tgt_output = tgt[:, 1:]
            src_mask, trg_mask = create_masks(src, tgt_input, full_dataset.src_pad_idx, full_dataset.src_pad_idx, opt['device'])
            preds = model(src, tgt_input, src_mask, trg_mask)

            ys = tgt[:, 1:].contiguous().view(-1)

            optimizer.zero_grad()
            loss = criterion(preds.view(-1, preds.size(-1)), ys)
            loss.backward()
            optimizer.step_and_update_lr()

            total_train_loss += loss.item()
        avg_train_loss = total_train_loss / len(train_loader)

        model.eval()
        total_dev_loss = 0
        with torch.no_grad():
            for src, tgt in tqdm(dev_loader, desc=f"Epoch {epoch} [Dev Loss]"):
                src, tgt = src.to(device), tgt.to(device)
                tgt_input = tgt[:, :-1]
                tgt_output = tgt[:, 1:]
                src_mask, trg_mask = create_masks(src, tgt_input, full_dataset.src_pad_idx, full_dataset.src_pad_idx, opt['device'])
                preds = model(src, tgt_input, src_mask, trg_mask)

                ys = tgt[:, 1:].contiguous().view(-1)
                loss = criterion(preds.view(-1, preds.size(-1)), ys)
                total_dev_loss += loss.item()
        avg_dev_loss = total_dev_loss / len(dev_loader)

        dev_bleu = 0.0

        if epoch % opt['printevery'] == 0:
            # os.makedirs('checkpoints', exist_ok=True)
            # save_path = f"checkpoints/model_epoch_{epoch}.pt"
            # torch.save(model.state_dict(), save_path)
            # print(f"Saved checkpoint: {save_path}")
            dev_bleu = evaluate_bleu(model, dev_ds)[0]

        os.makedirs('checkpoints', exist_ok=True)
        save_path = f"checkpoints/model_epoch_{epoch}.pt"
        save_checkpoint(model, optimizer, epoch, save_path)

        print(f"Epoch {epoch}: Train Loss={avg_train_loss:.4f}, Dev Loss={avg_dev_loss:.4f}, Dev BLEU={dev_bleu:.2f}%")

    # dev_bleu = evaluate_bleu(model, dev_ds)[0]

    # print(f"Final Dev BLEU: {dev_bleu:.2f}%")

    # Đánh giá trên test set
    test_dataset = TranslationDataset(test_src, test_tgt,
                                      src_tokenizer=full_dataset.src_tokenizer,
                                      tgt_tokenizer=full_dataset.tgt_tokenizer,)
    # test_loader = DataLoader(test_dataset, batch_size=64)

    test_bleu = evaluate_bleu(model, test_dataset)[0]

    print(f"Test BLEU: {test_bleu:.2f}%")
    



