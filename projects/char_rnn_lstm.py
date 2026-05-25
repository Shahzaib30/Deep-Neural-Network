"""Character-level LSTM (toy example)

This small script implements a character-level LSTM suitable for experimenting
with sequence modeling. It includes helper functions for fetching a tiny
dataset, batching, a compact `LSTM` model, and a sampling helper to generate
text from a trained model.

The file is intentionally lightweight and intended for experimentation and
education. For production use, split training, model, and sampling logic into
separate modules and add a CLI or configuration management.
"""

import torch
import torch.nn as nn
import requests
import torch.optim as optim

def get_data():
    url = 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'
    response = requests.get(url)
    text = response.text
    return text

def batch_loader(data, batch_size, seq_length):

    ix = torch.randint(len(data) - seq_length, (batch_size,))
    x = torch.stack([data[i:i+seq_length] for i in ix])
    y = torch.stack([data[i+1:i+seq_length+1] for i in ix])
    return x, y

class LSTM(nn.Module):
    def __init__ (self, vocab_size, hidden_size, embedding_dim):
        super(LSTM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)
    def forward(self, x, hidden=None):
        x = self.embedding(x)
        out, hidden = self.lstm(x, hidden)
        out = out.reshape(-1, out.shape[2]) 
        out = self.fc(out)
        return out, hidden
    
    
def generate_text(model, start_char, char_to_idx, idx_to_char, length=200):
    model.eval()
    hidden =None
    input_ids = [char_to_idx[c] for c in start_char]
    input_char = torch.tensor([input_ids], dtype=torch.long)
    hidden = None

    _, hidden = model(input_char, hidden)
    last_char_idx = input_ids[-1]
    input_char = torch.tensor([[last_char_idx]], dtype=torch.long)

    generated_text = start_char
    with torch.no_grad():
        for _ in range(length):
            output, hidden = model(input_char, hidden)
            probs = torch.softmax(output, dim=1).squeeze()
            next_char_idx = torch.multinomial(probs, num_samples=1).item()
            next_char = idx_to_char[next_char_idx]
            generated_text += next_char
            input_char = torch.tensor([[next_char_idx]], dtype=torch.long)

    return generated_text


if __name__ == "__main__":
    text = get_data()
    print(f"Length of text: {len(text)} characters")
    print(f"Unique characters: {len(set(text))}")

    batch_size = 64
    seq_length = 100
    char = sorted(list(set(text)))
    char_to_idx = {ch: idx for idx, ch in enumerate(char)}

    data = torch.tensor([char_to_idx[ch] for ch in text], dtype=torch.long)
    
    hidden_size = 128
    embedding_dim = 64
    vocab_size = len(set(text))
    Lstm_model = LSTM(vocab_size, hidden_size, embedding_dim)

    optimizer = optim.Adam(Lstm_model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    num_epochs = 1000

    for epoch in range(num_epochs):
        Lstm_model.train()

        x_batch, y_batch = batch_loader(data, batch_size, seq_length)

        optimizer.zero_grad()
        output, _ = Lstm_model(x_batch)
        loss = criterion(output, y_batch.view(-1))
        loss.backward()
        optimizer.step()
        
        if (epoch+1) % 100 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")
    

    print("Training complete. Now let's generate some text.")

    start_str = 'Romeo'
    char_to_ix = {ch: idx for idx, ch in enumerate(sorted(set(text)))}
    idx_to_char = {idx: ch for ch, idx in char_to_ix.items()}
    
    generated_text = generate_text(Lstm_model, start_str, char_to_ix, idx_to_char)
    print(f"Generated Text: {generated_text}")
