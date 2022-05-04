import sys
sys.path.append('./python')
import needle as ndl
import needle.nn as nn
import math
import numpy as np
np.random.seed(0)


class ResNet9(ndl.nn.Module):
    def __init__(self, device=None, dtype="float32"):
        super().__init__()
        self.device = device
        self.dtype = dtype

        self.features = nn.Sequential(
            self._make_convbn(3, 16, 7, 4),
            self._make_convbn(16, 32, 3, 2),
            nn.Residual(
                nn.Sequential(
                    self._make_convbn(32, 32, 3, 1),
                    self._make_convbn(32, 32, 3, 1),
                )
            ),
            self._make_convbn(32, 64, 3, 2),
            self._make_convbn(64, 128, 3, 2),
            nn.Residual(
                nn.Sequential(
                    self._make_convbn(128, 128, 3, 1),
                    self._make_convbn(128, 128, 3, 1),
                )
            ),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 128, device=device, dtype=dtype),
            nn.ReLU(),
            nn.Linear(128, 10, device=device, dtype=dtype)
        )

    def forward(self, x):
        features = self.features(x)
        return self.classifier(features)

    def _make_convbn(self, in_chan, out_chan, kernel_size, stride):
        return nn.Sequential(
            nn.Conv(in_chan, out_chan, kernel_size, stride,
                    device=self.device, dtype=self.dtype),
            nn.BatchNorm(out_chan, device=self.device, dtype=self.dtype),
            nn.ReLU()
        )


class LanguageModel(nn.Module):
    def __init__(self, embedding_size, output_size, hidden_size, num_layers=1,
                 seq_model='rnn', device=None, dtype="float32"):
        """
        Consists of an embedding layer, a sequence model (either RNN or LSTM), and a
        linear layer.

        Parameters:
        output_size: Size of dictionary
        embedding_size: Size of embeddings
        hidden_size: The number of features in the hidden state of LSTM or RNN
        seq_model: 'rnn' or 'lstm', whether to use RNN or LSTM
        num_layers: Number of layers in RNN or LSTM
        """
        super(LanguageModel, self).__init__()

        self.embedding_size = embedding_size
        self.output_size = output_size
        self.hidden_size = hidden_size

        _model_by_name = {"rnn": nn.RNN, "lstm": nn.LSTM}

        self.embedding = nn.Embedding(output_size, embedding_size, device=device, dtype=dtype)
        self.model = _model_by_name[seq_model](embedding_size, hidden_size, num_layers, device=device, dtype=dtype)
        self.proba = nn.Linear(hidden_size, output_size, device=device, dtype=dtype)

    def forward(self, x, h=None):
        """
        Given sequence (and the previous hidden state if given), returns probabilities of next word
        (along with the last hidden state from the sequence model).

        Inputs:
        x of shape (seq_len, bs)
        h of shape (num_layers, bs, hidden_size) if using RNN,
            else h is tuple of (h0, c0), each of shape (num_layers, bs, hidden_size)

        Returns (out, h)
        out of shape (seq_len*bs, output_size)
        h of shape (num_layers, bs, hidden_size) if using RNN,
            else h is tuple of (h0, c0), each of shape (num_layers, bs, hidden_size)
        """
        seq_len, bs = x.shape
        emb = self.embedding(x)
        fea, hid_st = self.model(emb, h)
        return self.proba(fea.reshape((seq_len * bs, self.hidden_size))), hid_st


if __name__ == "__main__":
    model = ResNet9()
    x = ndl.ops.randu((1, 32, 32, 3), requires_grad=True)
    model(x)
    cifar10_train_dataset = ndl.data.CIFAR10Dataset("data/cifar-10-batches-py", train=True)
    train_loader = ndl.data.DataLoader(cifar10_train_dataset, 128, ndl.cpu(), dtype="float32")
    print(dataset[1][0].shape)
