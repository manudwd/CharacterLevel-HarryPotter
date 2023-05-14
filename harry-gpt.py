#!/usr/bin/env python
# coding: utf-8

# In[75]:


import torch
import torch.nn as nn
from torch.nn import functional as F
import os
torch.manual_seed(42)

# hyperparameters
batch_size = 32 # how many independent sequences will we process in parallel?
block_size = 256 # what is the maximum context length for predictions?
max_iters = 10000
eval_interval = 500
learning_rate = 3e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 400
n_embd = 480
n_head = 8
n_layer = 8
dropout = 0.2
# ------------

with open('combined_books.txt', encoding='utf8') as fp:
    books = fp.read()

print('length of data is ',len(books))


# In[76]:


#first 1000 characters
print(books[:1000])
vocab = sorted(list(set(books)))
vocab_size = len(vocab)
vcti = {v:i for i,v in enumerate(vocab)}
itvc = {i:v for i,v in enumerate(vocab)}

encode = lambda s: [vcti[c] for c in s]
decode = lambda i: [itvc[c] for c in i]


# In[77]:


data = torch.tensor(encode(books),dtype=torch.long)

print(f'The data shape {data.shape} and type is {data.dtype}')


# In[78]:


n = int(0.9*len(data))
train_data = data[:n]
test_data = data[n:]
block_size=8
train_data[:block_size+1]


# In[79]:



def get_batch(split):
    #generate batches of input and target 
    data = train_data if split == 'train' else test_data
    ix = torch.randint(len(data)-block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    return x.to(device),y.to(device)

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out
class Head(nn.Module):
    
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)
        
        pass
    
    def forward(self, x):
        #input size is (Batch, Time, channels) 
        #output size is (Batch, time, head_size)
        
        B,T,C = x.shape
        k = self.key(x)
        q = self.query(x)
        v = self.value(x)
        
        #compute attention using key and query
        W = q@k.transpose(-2, -1) * k.shape[-1]**-0.5 #(B,T,Hs)@(B,Hs,T) -> (B,T,T)
        W = W.masked_fill(self.tril[:T,:T] ==0, float('-inf')) #to make sure only T dimensions are captured, its :T
        W = F.softmax(W, dim=-1) #apply to the last dimension which is channel
        W = self.dropout(W)
        #perform weighted aggregation of the values with the dot product
        out = W @(v) #(B,T,T)@(B,T,hs)->(B,T,Hs)
        return out
        
class MultiHeadAttention(nn.Module):
    """Multiple heads of self-attention in parallel"""
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList(Head(head_size) for _ in range(num_heads))
        self.dropout = nn.Dropout(dropout)
        self.proj = nn.Linear(head_size*num_heads, n_embd)
    
    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(out)
        return out

    
class FeedForwardNetwork(nn.Module):
    
    def __init__(self, n_embd):
        super().__init__()
        
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4*n_embd),
            nn.ReLU(),
            nn.Linear(4*n_embd, n_embd),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd//n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForwardNetwork(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
        
    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x
    
class BigramLanguageModel(nn.Module):
    
    def __init__(self):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding = nn.Embedding(block_size, n_embd)
        self.block = nn.Sequential(*[Block(n_embd, n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)
        
        #weight initialization
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    
    def forward(self, idx, targets=None):
        B,T = idx.shape
        
        #idx and target are both (B,T) tensor of integers
        tok_emb = self.token_embedding_table(idx) #(B,T,C)
        pos_emb = self.position_embedding(torch.arange(T, device=device)) #(T,C)
        x = tok_emb + pos_emb # (B,T,C)
        x = self.block(x) #(B,T,C)
        x = self.ln_f(x) #(B,T,C)
        logits = self.lm_head(x) #(B,T,Vocab_size)
        
        if targets is None:
            loss = None
        else:
            B,T,C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)
            
        return logits, loss
    
    def generate(self, idx, max_new_tokens):
        
        # idx is (B,T) array of indices in the current context
        for _ in range(max_new_tokens):
            # crop idx to the last block_size tokens
            idx_cond = idx[:, -block_size:]
            #get the predictions
            logits, loss = self(idx_cond)
            #focus only on the last time step
            logits = logits[:, -1, :] #becomes (B,C)
            #apply softmax to get probablities
            probs = F.softmax(logits, dim=-1) #(B,C)
            #sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) #(B,1)
            # append the sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) #(B,T+1)
        return idx

model = BigramLanguageModel()
m  = model.to(device)

#print the number of parameters in the model
print(sum(p.numel() for p in m.parameters())/1e6, 'M parameters')

# Create a PyTorch optimizer
optimizer = torch.optim.AdamW(m.parameters(), lr=learning_rate)

for itr in range(max_iters):
    
    #every once in a while evaluate the loss on train and val sets
    if itr % eval_interval == 0 or iter == max_iters -1:
        losses = estimate_loss()
        print(f"step {itr}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
        
    # sample a batch of data
    xb, yb = get_batch('train')
    
    # evaluate the loss
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    
#generate from the model
context = torch.zeros((1,1), device=device, dtype=torch.long)
m.eval()
#print(decode(m.generate(context, max_new_tokens=500)[0].tolist()))
#open('harry_potter_by_me.txt', 'w').write(decode(m.generate(context, max_new_tokens=10000)[0].tolist()))


# In[82]:


content = decode(m.generate(context, max_new_tokens=10000)[0].tolist())
my_harry_potter = "".join(content)


# In[84]:


open('harry_potter_by_me.txt', 'w', encoding='utf8').write(my_harry_potter)

