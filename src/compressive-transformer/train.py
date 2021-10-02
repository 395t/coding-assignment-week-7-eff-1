from compressive_transformer_pytorch import CompressiveTransformer
from compressive_transformer_pytorch.autoregressive_wrapper import AutoregressiveWrapper

import random
import tqdm
import zipfile
import numpy as np
import os
import requests
import torch
import torch.optim as optim
from torch import Tensor
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset, dataset
from torchtext import datasets
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator

# constants

NUM_BATCHES = int(1e5)
BATCH_SIZE = 16
MAX_BATCH_SIZE = 4
LEARNING_RATE = 1e-4
VALIDATE_EVERY  = 100

GENERATE_EVERY  = 500
PRIME_LENGTH    = 512
GENERATE_LENGTH = 1024

SEQ_LEN = 512
NUM_SEGMENTS = 4

# helpers

def cycle(loader):
    while True:
        for data in loader:
            yield data

def decode_token(token):
    return str(chr(max(32, token)))

def decode_tokens(tokens):
    return ''.join(list(map(decode_token, tokens)))

# prepare data
class TextSamplerDataset(Dataset):
    def __init__(self, data, seq_len, segments):
        super().__init__()
        self.data = data
        self.seq_len = seq_len
        self.segments = segments
        self.total_len = seq_len * segments

    def __getitem__(self, index):
        rand_start = torch.randint(0, self.data.size(0) - self.total_len - 1, (1,))
        full_seq = self.data[rand_start: rand_start + self.total_len + 1].long()
        return full_seq.cuda()

    def __len__(self):
        return self.data.size(0) // self.total_len

dataset_name = 'enwik8'

if dataset_name == 'enwik8':
    enwik8_path = "./data/enwik8.zip"
    os.makedirs("data", exist_ok=True)
    if not os.path.exists(enwik8_path):
        response = requests.get("https://data.deepai.org/enwik8.zip")
        with open(enwik8_path, 'wb') as f:
            f.write(response.content)

    with zipfile.ZipFile(enwik8_path, 'r') as archive:
        X = np.fromstring(archive.read('enwik8')[:int(95e6)], dtype=np.uint8)
        trX, vaX = np.split(X, [int(90e6)])
        train_data, val_data = torch.from_numpy(trX), torch.from_numpy(vaX)
    num_tokens = 256
    lookup_fun = decode_tokens

else:
    if dataset_name == 'WikiText-2':
        dataset_obj = datasets.WikiText2
    else:
        dataset_obj = datasets.PennTreebank
    train_iter = dataset_obj(root='data', split='train')
    tokenizer = get_tokenizer('basic_english')
    vocab = build_vocab_from_iterator(map(tokenizer, train_iter), specials=['<unk>'])
    vocab.set_default_index(vocab['<unk>'])

    def data_process(raw_text_iter: dataset.IterableDataset) -> Tensor:
        """Converts raw text into a flat Tensor."""
        data = [torch.tensor(vocab(tokenizer(item)), dtype=torch.long) for item in raw_text_iter]
        return torch.cat(tuple(filter(lambda t: t.numel() > 0, data)))

    train_iter, val_iter, _ = dataset_obj(root='data')
    train_data = data_process(train_iter)
    val_data = data_process(val_iter)
    num_tokens = len(vocab)
    lookup_fun = vocab.lookup_tokens

train_dataset = TextSamplerDataset(train_data, SEQ_LEN, NUM_SEGMENTS)
val_dataset   = TextSamplerDataset(val_data, SEQ_LEN, NUM_SEGMENTS)
train_loader  = cycle(DataLoader(train_dataset, batch_size = BATCH_SIZE))
val_loader    = cycle(DataLoader(val_dataset, batch_size = BATCH_SIZE))

# instantiate model

model = CompressiveTransformer(
    num_tokens = num_tokens,
    dim = 512,
    depth = 8,
    seq_len = SEQ_LEN,
    mem_len = SEQ_LEN,
    cmem_len = SEQ_LEN // 4,
    heads = 8,
    gru_gated_residual = False,
    one_kv_head = False,
    memory_layers = None
)

model = AutoregressiveWrapper(model)
model.cuda()

# optimizer

optim = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

# training

for i in tqdm.tqdm(range(NUM_BATCHES), mininterval=10., desc='training'):
    model.train()

    grad_accum_every = BATCH_SIZE / MAX_BATCH_SIZE

    for mlm_loss, aux_loss, is_last in model(next(train_loader), max_batch_size = MAX_BATCH_SIZE, return_loss = True):
        loss = mlm_loss + aux_loss
        (loss / grad_accum_every).backward()

        print(f'training loss: {mlm_loss.item():.4f} | aux_loss: {aux_loss.item():.4f}')

        if is_last:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optim.step()
            optim.zero_grad()

    if (i+1) % VALIDATE_EVERY == 0:
        model.eval()
        with torch.no_grad():
            for loss, aux_loss, _ in model(next(val_loader), return_loss = True):
                print(f'validation loss: {loss.item():.4f}')

    if (i+1) % GENERATE_EVERY == 0:
        model.eval()
        inp = random.choice(val_dataset)[:-1]
        inp = inp[:PRIME_LENGTH]
        prime = lookup_fun(inp.cpu().numpy())
        print(f'%s \n\n %s', (prime, '*' * 100))

        sample = model.generate(inp, GENERATE_LENGTH)
        output_str = lookup_fun(sample.cpu().numpy())
        print(output_str)
