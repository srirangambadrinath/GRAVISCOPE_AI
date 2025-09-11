import torch
import torch.nn as nn
import torch.nn.functional as F

class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.attn = nn.Linear(hidden_dim * 2, 1)

    def forward(self, rnn_output):
        weights = F.softmax(self.attn(rnn_output), dim=1)  # (batch, seq_len, 1)
        context = torch.sum(weights * rnn_output, dim=1)   # (batch, hidden_dim*2)
        return context

class GraviscopeModel(nn.Module):
    def __init__(self, input_length, cnn_out=64, gru_hidden=128, num_classes=2):
        super().__init__()

        self.cnn = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=7, padding=3),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(32, cnn_out, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(2),
        )

        self.gru = nn.GRU(input_size=cnn_out, hidden_size=gru_hidden, batch_first=True, bidirectional=True)
        self.attention = Attention(gru_hidden)
        self.classifier = nn.Linear(gru_hidden * 2, num_classes)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.cnn(x)
        x = x.permute(0, 2, 1)
        gru_out, _ = self.gru(x)
        attn_out = self.attention(gru_out)
        out = self.classifier(attn_out)
        return out

    def forward_with_attention(self, x):
        x = x.unsqueeze(1)
        x = self.cnn(x)
        x = x.permute(0, 2, 1)
        gru_out, _ = self.gru(x)
        weights = torch.softmax(self.attention.attn(gru_out), dim=1)
        attn_out = torch.sum(weights * gru_out, dim=1)
        logits = self.classifier(attn_out)
        return logits, weights.squeeze(-1)
