import torch 
 import torch.nn as nn
 from torch.utils.data import DataLoader, Dataset
 import torch.optim as optim
 import math
 from tqdm import tqdm
 
 with open('learning/project3/data/shakesspeare.txt', 'r', encoding='utf-8') as file:
     content = file.read()
 
 unique_chars  = sorted(set(content))
 
 vocab = {}
 vocab_inv = {}
 
 i = 0
 for c in unique_chars:
     vocab[c] = i
     vocab_inv[i] = c
     i+=1
 
 vocab_len = len(vocab)
 print(f"Vocab Size={vocab_len}")
 
 tokens_list = [ vocab[x] for x in content ]
 tokens_size= len(tokens_list)
 
 training_tokens = torch.tensor( tokens_list[:int(0.8*tokens_size)],  dtype=torch.long )
 validation_tokens = torch.tensor( tokens_list[int(0.8*tokens_size):], dtype=torch.long )
 
 class TokenDataset(Dataset):
     def __init__(self, data_tensor, window_size):
         self.data = data_tensor
         self.window_size = window_size
 
     def __len__(self):
         return len(self.data) - self.window_size
 
     def __getitem__(self, idx):
         inputs = self.data[idx : idx + self.window_size]
         targets = self.data[idx + 1: idx + self.window_size + 1]
         return (inputs, targets)
 
 batch_size = 64
 window_size = 128
 
 training_dataset = TokenDataset(training_tokens, window_size)
 training_dataloader = DataLoader(training_dataset, batch_size, shuffle=True)
 
 validation_dataset = TokenDataset(validation_tokens, window_size)
 validation_dataloader = DataLoader(validation_dataset, batch_size)
 
 
 class TinyCharRNN(nn.Module):
     def __init__(self):
         super(TinyCharRNN, self).__init__()
         self.embed = nn.Embedding(vocab_len, 256)
         self.rnn1 = nn.GRU(256, 256, batch_first=True)
         self.rnn2 = nn.GRU(256, 256)
         self.linear = nn.Linear(256, vocab_len)
 
     def forward(self, x):
         l1 = self.embed(x)
         rnn_out1, h = self.rnn1(l1)
         rnn_out2, h = self.rnn2(rnn_out1)
         l3 = self.linear(rnn_out2)
         return l3
 
 model = TinyCharRNN()
 loss = nn.CrossEntropyLoss()
 optimizer = optim.AdamW( model.parameters(), lr=0.001 )
 
 def run_test():
     test_input = """General:
 Are we blind !?
 Deploy the garrison!
 Move!!! 
     """
     print(len(test_input))
     test_list = [ vocab[x] for x in test_input ]
     test_tokens = torch.tensor( test_list, dtype=torch.long )
     output = model(test_tokens)
     output = torch.argmax( output, dim=1)
     output = output.tolist()
     output = [ vocab_inv[i] for i in output ] 
     output = "".join(output)
     result = test_input + output
     print("---Test---")
     print(result)
 
 run_test()
 
 for epoch in range(20):
     pbar = tqdm(training_dataloader, desc="Training")
 
     model.train()
     training_loss = 0
     training_steps = 0
     for i, (training_inputs, training_targets) in enumerate(pbar):
         optimizer.zero_grad()
         training_preds = model(training_inputs)
         training_preds = torch.flatten(training_preds, 0, 1)
         training_targets = torch.flatten(training_targets, 0)
         current_loss = loss(training_preds, training_targets)
         current_loss.backward()
         optimizer.step()
         training_loss += current_loss.item()
         training_steps += 1
         loss_str = f"{current_loss.item():.2f}"
         pbar.set_postfix({"loss":loss_str})
  
     print("Validating...")
     validation_loss = 0
     validation_steps = 0
     model.eval()
     with torch.no_grad():
         for i, (validation_inputs, validation_targets) in enumerate(validation_dataloader):
             validation_preds = model(validation_inputs)
             validation_preds = torch.flatten(validation_preds, 0, 1)
             validation_targets = torch.flatten(validation_targets, 0)
             current_loss = loss(validation_preds, validation_targets)
             validation_loss += current_loss.item()
             validation_steps += 1
 
     avg_training_loss = training_loss / training_steps
     avg_validation_loss = validation_loss / validation_steps
     perplexity = math.exp(avg_validation_loss)
     print(f"================={epoch}==============")
     print(f"avg_t_loss={avg_training_loss:.4f} | avg_v_loss={avg_validation_loss} | perplexity={perplexity}")
     run_test()
     print(f"======================================")
 
     torch.save(model.state_dict(), f'learning/project3/model/checkpoint_{epoch}.pth')