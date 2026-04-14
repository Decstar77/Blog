import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import math
from tqdm import tqdm

# Problems I encountered:
#   - CrossEntropyLoss sizing mismatch, expects B * T, V
#   - mask_fill is not inplace 
#   - Dont need softmax on last layer because the CrossEntropyLoss applies it. 
#       - I was doing something like softmax(softmax(x))
#   - Dataset was wrong because __len__ was just returning len(data)
#       - Would lead to crashes when reading off the edge of the list
#       - Fix is to make sure that len is less the context window
#   - Learnt about projection, taking a 4 * d_model and scaling it down back to d_model
#   - Learning about multi-head attention 
#   - Learnt about drop out 
#   - Learnt about weight decay
#   - Learnt about grad clipping normalization 
#   - Learnt about GELU

batch_size      = 128
context_window  = 128
model_size      = 256
model_heads     = 8
checkpoint_path = "project5/model"
data_path       = "data/garethdeclan/clean_data.txt" 
inputprompt     = "Declan: How goes it ?"
load_checkpoint      = "project6-tinytransformer/checkpoints/checkpoint_9.pth"
torch.manual_seed(5)

with open(data_path, "r") as file:
    content = file.read()

unique_chars = sorted(set(content))

vocab = {}
vocab_inv = {}

for i, char in enumerate(unique_chars):
    vocab[char] = i
    vocab_inv[i] = char

assert len(vocab) == len(vocab_inv)
vocab_size = len(vocab)
print(f"Vocab size {vocab_size}")

content_vocab       = [ vocab[char] for char in content ]
raw_training_data   = content_vocab[:int(0.8 * len(content_vocab)) ]
raw_validation_data = content_vocab[ int(0.8 * len(content_vocab)):]

class TinyDataset(Dataset):
    def __init__(self, raw_data):
        self.data = raw_data

    def __len__(self):
        return len(self.data) - context_window

    def __getitem__(self, idx):
        inputs  = self.data[ idx     : idx + context_window ]
        targets = self.data[ idx + 1 : idx + context_window + 1 ]
        inputs  = torch.tensor(inputs,  dtype=torch.long)
        targets = torch.tensor(targets, dtype=torch.long)
        return (inputs, targets)

training_dataloader   = DataLoader( TinyDataset(raw_training_data),   batch_size=batch_size, shuffle=True )
validation_dataloader = DataLoader( TinyDataset(raw_validation_data), batch_size=batch_size )

class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, device):
        super(TransformerBlock, self).__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.device = device
        self.Q = nn.Linear(d_model, d_model)
        self.K = nn.Linear(d_model, d_model)
        self.V = nn.Linear(d_model, d_model)
        self.softmax = nn.Softmax(-1)
        self.proj = nn.Linear(d_model, d_model)
        self.layernorm1 = nn.LayerNorm(d_model)
        self.layernorm2 = nn.LayerNorm(d_model)
        self.ffn1 = nn.Linear(d_model, 4 * d_model)
        self.ffn2 = nn.Linear( 4 * d_model, d_model)
        self.ffnA = nn.GELU()
        self.attn_drop  = nn.Dropout(0.2)
        self.ffn_drop   = nn.Dropout(0.2)

    def forward(self, x):
        B = x.shape[0]
        T = x.shape[1]
        norm = self.layernorm1(x)

        dims = int(self.d_model / self.n_heads)
        
        q = self.Q(norm)
        k = self.K(norm)
        v = self.V(norm)
        
        q = q.reshape(B, T, self.n_heads, dims).transpose(1, 2) # [B, Heads, T, Dims]
        k = k.reshape(B, T, self.n_heads, dims).transpose(1, 2) # [B, Heads, T, Dims]
        v = v.reshape(B, T, self.n_heads, dims).transpose(1, 2) # [B, Heads, T, Dims]

        a = q @ k.transpose(-2, -1) / math.sqrt(dims)
        
        mask = ~torch.tril( torch.ones( T, T, dtype=torch.bool, device=self.device ) )
        a = a.masked_fill(mask, float("-inf"))
        a = self.attn_drop( self.softmax(a) ) @ v

        a = a.transpose(1, 2)                       # (B, T, H, DIMS)
        a = a.reshape(B, T, self.d_model)

        a_out = self.proj( a )

        ar  = x + a_out
        out = ar + self.ffn_drop( self.ffn2( self.ffnA( self.ffn1( self.layernorm2(ar) ) )  ) )
        return out


class TinyDecoder(nn.Module):
    def __init__( self, d_model, n_heads, device ):
        super(TinyDecoder, self).__init__()
        self.device = device
        self.d_model = d_model
        self.token_embed        = nn.Embedding(vocab_size, d_model)
        self.pos_embed          = nn.Embedding(context_window, d_model)
        self.embed_drop         = nn.Dropout(0.2)
        self.transformer1       = TransformerBlock(d_model=d_model, n_heads=n_heads, device=device)
        self.transformer2       = TransformerBlock(d_model=d_model, n_heads=n_heads, device=device)
        self.transformer3       = TransformerBlock(d_model=d_model, n_heads=n_heads, device=device)
        self.transformer4       = TransformerBlock(d_model=d_model, n_heads=n_heads, device=device)
        self.transformer5       = TransformerBlock(d_model=d_model, n_heads=n_heads, device=device)
        self.transformer6       = TransformerBlock(d_model=d_model, n_heads=n_heads, device=device)
        self.outln              = nn.LayerNorm(d_model)
        self.outffn             = nn.Linear(d_model, vocab_size)
        
    def forward(self, x):
        t = x.shape[1]
        te = self.token_embed(x)
        pe = self.pos_embed( torch.arange( 0, t, dtype=torch.long, device=self.device ) )
        e = self.embed_drop( te + pe )
        t1 = self.transformer1(e)
        t2 = self.transformer2(t1)
        t3 = self.transformer3(t2)
        t4 = self.transformer4(t3)
        t5 = self.transformer5(t4)
        t6 = self.transformer6(t5)
        out = self.outffn( self.outln( t6 ) )
        return out


if torch.cuda.is_available():
    print(torch.cuda.get_device_name(0))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#device = torch.device("cpu")

model     = TinyDecoder(d_model=model_size, n_heads=model_heads, device=device).to(device)
losser    = nn.CrossEntropyLoss()
optimizer = optim.AdamW( model.parameters(), lr=0.3e-4, weight_decay=0.1 )

def prompt( text, max_new_tokens=200 ):
    print(text)
    with torch.no_grad():
        tokens = torch.tensor( [ vocab[char] for char in text ], dtype=torch.long, device=device )
        tokens = tokens.unsqueeze(0)  # [1, T]
        model.eval()
        for _ in range(max_new_tokens):
            context = tokens[:, -context_window:]          # crop if too long
            logits  = model(context)                       # [1, T, vocab_size]
            next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)  # [1, 1]
            tokens = torch.cat([tokens, next_token], dim=1)
        generated = tokens[0, len(text):].tolist()
        print("".join([ vocab_inv[i] for i in generated ]))

total_params = sum(p.numel() for p in model.parameters())
print(f"Training tiny transformer! {(total_params / 1000000):.2f} million parameters ")

if load_checkpoint == "":
    for epoch in range(10):
        counter = 0
        training_loss = 0
        pbar = tqdm(training_dataloader, desc="Training")
        model.train()
        for i, (inputs, targets) in enumerate(pbar):
            inputs = inputs.to(device)
            targets = targets.to(device)
            optimizer.zero_grad()
            logits = model(inputs)
            logits = logits.reshape(-1, vocab_size)
            targets = targets.reshape(-1)
            loss = losser(logits, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            training_loss += loss.item()
            counter += 1
            pbar.set_postfix(loss=f"{(training_loss / counter):.4f} | {torch.cuda.memory_reserved()  / 1024**2:.1f}MB")

        counter = 0
        validation_loss = 0
        pbar = tqdm(validation_dataloader, desc="Validating")
        model.eval()
        for i, (inputs, targets) in enumerate(pbar):
            inputs = inputs.to(device)
            targets = targets.to(device)
            with torch.no_grad():
                logits = model(inputs)
            logits = logits.reshape(-1, vocab_size)
            targets = targets.reshape(-1)
            loss = losser(logits, targets)
            validation_loss += loss.item()
            counter += 1
            pbar.set_postfix(loss=f"{(validation_loss / counter):.4f}")

        prompt(inputprompt)

        torch.save(model.state_dict(), f"{checkpoint_path}/checkpoint_{epoch}.pth")
else:
    state = torch.load(load_checkpoint, weights_only=True)
    model.load_state_dict( state )
    model.eval()
    
    print("=======================")
    prompt("Declan: Henlo boyo")

