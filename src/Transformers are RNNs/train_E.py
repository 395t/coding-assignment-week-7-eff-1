import torch
from fast_transformers.builders import TransformerEncoderBuilder, RecurrentEncoderBuilder
from fast_transformers.masking import TriangularCausalMask
from torchtext.datasets import WikiText2, PennTreebank, EnWik9
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
import torchtext
from transformers import T5Tokenizer
from torch import nn, Tensor
from torch.utils.data import dataset, DataLoader

import random
import numpy as np
import time
from typing import Tuple
import math
import copy
import os
import gzip

np.random.seed(42)
random.seed(42)
torch.manual_seed(42)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
config={
    # data processing
    'batch_size': 16,
    'eval_batch_size': 10,

    # network architecture
    'd_model': 512,
    'n_layers': 8,
    'n_heads': 8,
    'query_dimensions': 64,
    'value_dimensions': 64,
    'feed_forward_dimensions': 1024,

    # training
    'epochs': 10,
    'bptt': 250,
    'lr': 0.00005,
    'gamma': 0.95,
    'optim': 'Adam',

    # dir
    'model_dir': "train_model",
    'loss_dir': "loss_dir",

    'dataset': 'E',

    'test_model': 'train_model/dataW/d512_nl11_nh8_qd64_ffd1024_bptt35/best_model.pth'

}


# 4 - tmux 1
# 64 - tmux 2
# 256 - tmux 4


# config['exp_name'] = 'data%s/d%d_nl%d_nh%d_qd%d_ffd%d_bptt%d_ep%d'%(config['dataset'], config['d_model'], config['n_layers'], config['n_heads'], config['query_dimensions'], config['feed_forward_dimensions'], config['bptt'], config['epochs'])
# config['exp_name'] = 'data%s/b%d_ep%d_opt%s_lr%f'%(config['dataset'], config['batch_size'], config['epochs'], config['optim'], config['lr'])
config['exp_name'] = 'data%s/full'%config['dataset']

TEST=False

if not TEST:
    print('training model with exp name %s'%config['exp_name'])




def make_save_dir(save_dir):
    '''Make a directory if it does not exist.'''
    if not os.path.exists(save_dir):
        print('making directory %s...'%save_dir)
        os.makedirs(save_dir)
    return save_dir


def generate_square_subsequent_mask(sz: int) -> Tensor:
    """Generates an upper-triangular matrix of -inf, with zeros on diag."""
    return torch.triu(torch.ones(sz, sz) * float('-inf'), diagonal=1)


# helpers

def cycle(loader):
    while True:
        for data in loader:
            yield data

def decode_token(token):
    return str(chr(max(32, token)))

def decode_tokens(tokens):
    return ''.join(list(map(decode_token, tokens)))








class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

class LinearTransformer(nn.Module):

    def __init__(self, encoder: nn.Module,ntoken: int, d_model: int, dropout: float = 0.5):
        super().__init__()
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        self.transformer_encoder = encoder
        self.encoder = nn.Embedding(ntoken, d_model)
        self.d_model = d_model
        self.decoder = nn.Linear(d_model, ntoken)
        self.ntoken = ntoken

        self.init_weights()

    def init_weights(self) -> None:
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src: Tensor, src_mask: Tensor) -> Tensor:
        """
        Args:
            src: Tensor, shape [seq_len, batch_size]
            src_mask: Tensor, shape [seq_len, seq_len]

        Returns:
            output Tensor of shape [seq_len, batch_size, ntoken]
        """
        src = self.encoder(src) * math.sqrt(self.d_model)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, src_mask)
        output = self.decoder(output)
        return output


def build_model(ntoken: int, d_model: int, n_layers: int=8, n_heads: int=8, q_dim: int=64, v_dim: int=64, ff_dim: int=1024):
    # Create the builder for our transformers
    # builder = TransformerEncoderBuilder.from_kwargs(
    builder = TransformerEncoderBuilder.from_kwargs(
        n_layers=n_layers,
        n_heads=n_heads,
        query_dimensions=q_dim,
        value_dimensions=v_dim,
        feed_forward_dimensions=ff_dim
    )
    # print(builder._get_attention_builder().available_attentions)
    # Build a transformer with linear attention
    # builder.attention_type = 'causal-linear'
    builder.attention_type='full'
    linear_model = builder.get()

    return LinearTransformer(linear_model, ntoken, d_model).to(device)



class TextSamplerDataset(dataset.Dataset):
    def __init__(self, data, seq_len):
        super().__init__()
        self.data = data
        self.seq_len = seq_len
        # self.total_len = seq_len * segments

    def __getitem__(self, index):
        # rand_start = torch.randint(0, self.data.size(0) - self.total_len - 1, (1,))
        # full_seq = self.data[rand_start: rand_start + self.total_len + 1].long()
        full_seq = self.data[self.seq_len*index:self.seq_len*(index+1)].long()
        y_seq = self.data[self.seq_len*index+1:self.seq_len*(index+1)+1].long()
        return {'x':full_seq.to(device), 'y':y_seq.to(device)}

    def __len__(self):
        # return 2000
        return (self.data.size(0)-1) // self.seq_len


def train(model, train_data, optimizer, criterion, len_train):
    
    model.train()  # turn on train mode
    total_loss = 0.
    losses=[]
    log_interval = 1000
    start_time = time.time()

    num_batches = len_train
    # for batch, i in enumerate(range(0, train_data.size(0) - 1, config['bptt'])):
    for i, batch in enumerate(train_data):
        data, targets = batch['x'], batch['y']
        targets = targets.view(-1)


        # if data.size(1) != bptt:  # only on last batch
        src_mask=TriangularCausalMask(data.size(1), device=device)
            # src_mask = src_mask[:batch_size, :batch_size]
        output = model(data, src_mask)
        loss = criterion(output.view(-1, model.ntoken), targets)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()

        total_loss += loss.item()
        losses.append(loss.item())
        if i % log_interval == 0 and i > 0:
            lr = scheduler.get_last_lr()[0]
            ms_per_batch = (time.time() - start_time) * 1000 / log_interval
            cur_loss = total_loss / log_interval
            ppl = math.exp(cur_loss)
            print(f'| epoch {epoch:3d} | {i:5d}/{num_batches:5d} batches | '
                  f'lr {lr:02.2f} | ms/batch {ms_per_batch:5.2f} | '
                  f'loss {cur_loss:5.2f} | ppl {ppl:8.2f}')
            total_loss = 0
            start_time = time.time()

    return losses


def evaluate(model: nn.Module, eval_data: Tensor, criterion: nn.Module) -> float:
    model.eval()  # turn on evaluation mode
    total_loss = 0.
    total=0
    with torch.no_grad():
        # for i in range(0, eval_data.size(0) - 1, config['bptt']):
        for i, batch in enumerate(eval_data):
            data, targets = batch['x'], batch['y']
            targets = targets.view(-1)
            src_mask=TriangularCausalMask(data.size(1), device=device)

            output = model(data, src_mask)
            output_flat = output.view(-1, model.ntoken)
            # total_loss += (data.size(0) * criterion(output_flat, targets).item())
            total_loss += criterion(output_flat, targets).item()
            total += 1
    # return total_loss / (len(eval_data) - 1)
    return total_loss/ total


file = gzip.open('./data/enwik8.gz')
X = np.fromstring(file.read(int(95e6)), dtype=np.uint8)
trX, vaX = np.split(X, [int(90e6)])
data_train, data_val = torch.from_numpy(trX), torch.from_numpy(vaX)
print('data', data_train.size())

train_dataset = TextSamplerDataset(data_train, config['bptt'])
val_dataset   = TextSamplerDataset(data_val, config['bptt'])
len_train     = len(train_dataset)//config['batch_size']
len_val       = len(val_dataset) // config['batch_size']
train_loader  = DataLoader(train_dataset, batch_size = config['batch_size'])
val_loader    = DataLoader(val_dataset, batch_size = config['batch_size'])

print(f'Data Size: Train: {len(train_dataset)} Val Data: {len(val_dataset)}')


# ntoken=torch.max(data_train.long())
ntoken=256
print(ntoken)
# exit(0)


# some init 

model=build_model(ntoken, 
                  config['d_model'],
                  n_layers=config['n_layers'], 
                  n_heads=config['n_heads'], 
                  q_dim=config['query_dimensions'], 
                  v_dim=config['value_dimensions'], 
                  ff_dim=config['feed_forward_dimensions']
                  )
if config['optim'] == 'Adam':
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])
elif config['optim'] == 'SGD':
    optimizer = torch.optim.SGD(model.parameters(), lr=config['lr'])
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=config['gamma'])

best_val_loss = float('inf')
best_model = None
criterion=nn.CrossEntropyLoss()


if TEST:
    print('loading from %s'%config['test_model'])
    model.load_state_dict(torch.load(config['test_model']))
    loss = evaluate(model, val_data, criterion)
    perplexity=math.exp(loss)
    print(f'loss: {loss} | exp: {perplexity}')
    print('finish testing...')
    exit(0)


### training ###
train_losses=[]
val_losses=[]
for epoch in range(1, config['epochs'] + 1):
    epoch_start_time = time.time()
    train_loss = train(model, train_loader, optimizer, criterion, len_train)
    val_loss = evaluate(model, val_loader, criterion)
    train_losses += train_loss
    val_losses.append(val_loss)
    

    val_ppl = math.exp(val_loss)
    elapsed = time.time() - epoch_start_time
    print('-' * 89)
    print(f'| end of epoch {epoch:3d} | time: {elapsed:5.2f}s | '
          f'valid loss {val_loss:5.2f} | valid ppl {val_ppl:8.2f}')
    print('-' * 89)

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_model = copy.deepcopy(model)
        make_save_dir(os.path.join(config['model_dir'], config['exp_name']))
        print('saving!!!! To %s...'%os.path.join(config['model_dir'], config['exp_name'], 'best_model.pth'))
        torch.save(best_model.state_dict(), os.path.join(config['model_dir'], config['exp_name'], 'best_model.pth'))
    scheduler.step()

make_save_dir(os.path.join(config['loss_dir'], config['exp_name']))
np.save(os.path.join(config['loss_dir'], config['exp_name'], 'training_loss.npy'), train_losses)
np.save(os.path.join(config['loss_dir'], config['exp_name'], 'valid_loss.npy'), val_losses)