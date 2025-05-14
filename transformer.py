##01.03.2025
import numpy as np
from matplotlib import pyplot as plt
from scipy.ndimage import gaussian_filter1d
import torch
import torch.nn as nn
import torch.nn.functional as F
class MultiHeadAttentionBlock(nn.Module):
    def __init__(self, dim_val, dim_attn, n_heads):
        super(MultiHeadAttentionBlock, self).__init__()
        self.attn = nn.MultiheadAttention(dim_val, n_heads, batch_first=True)

    def forward(self, x, kv=None):

        return self.attn(x, kv, kv)[0] if kv is not None else self.attn(x, x, x)[0]

class EncoderLayer(nn.Module):
    def __init__(self, dim_val, dim_attn, n_heads=1):
        super(EncoderLayer, self).__init__()
        self.attn = MultiHeadAttentionBlock(dim_val, dim_attn, n_heads)
        self.fc1 = nn.Linear(dim_val, dim_val)
        self.fc2 = nn.Linear(dim_val, dim_val)
        self.norm1 = nn.LayerNorm(dim_val)
        self.norm2 = nn.LayerNorm(dim_val)

    def forward(self, x):
        a = self.attn(x)
        x = self.norm1(a + x)
        a = self.fc1(F.elu(self.fc2(x)))
        x = self.norm2(x + a)
        return x
class DecoderLayer(nn.Module):
    def __init__(self, dim_val, dim_attn, n_heads=1):
        super(DecoderLayer, self).__init__()
        self.attn1 = MultiHeadAttentionBlock(dim_val, dim_attn, n_heads)
        self.attn2 = MultiHeadAttentionBlock(dim_val, dim_attn, n_heads)
        self.fc1 = nn.Linear(dim_val, dim_val)
        self.fc2 = nn.Linear(dim_val, dim_val)

        self.norm1 = nn.LayerNorm(dim_val)
        self.norm2 = nn.LayerNorm(dim_val)
        self.norm3 = nn.LayerNorm(dim_val)

    def forward(self, x, enc):
        a = self.attn1(x)
        x = self.norm1(a + x)
        a = self.attn2(x, kv=enc)
        x = self.norm2(a + x)
        a = self.fc1(F.elu(self.fc2(x)))
        x = self.norm3(x + a)
        return x
def generate_lookahead_mask(size):

    mask = torch.triu(torch.ones(size, size), diagonal=1).bool()
    return mask
class Transformer(nn.Module):
    def __init__(self, dim_val, dim_attn, input_size, dec_seq_len, out_seq_len, n_decoder_layers=1, n_encoder_layers=1, n_heads=1):
        super(Transformer, self).__init__()
        self.dec_seq_len = dec_seq_len
        self.T = out_seq_len

        self.encs = nn.ModuleList([EncoderLayer(dim_val, dim_attn, n_heads) for _ in range(n_encoder_layers)])

        self.decs = nn.ModuleList([DecoderLayer(dim_val, dim_attn, n_heads) for _ in range(n_decoder_layers)])

        self.fc_out = nn.Linear(dim_val, out_seq_len)

    def forward(self, x):

        for enc in self.encs:
            x = enc(x)

        for dec in self.decs:
            x = dec(x, x)


        output = self.fc_out(x)
        return output

SMOOTHING_SIGMA = 20

def power_spectrum_error(x_gen, x_true):
    pse_errors_per_dim = power_spectrum_error_per_dim(x_gen, x_true)
    return np.array(pse_errors_per_dim).mean(axis=0)

def compute_power_spectrum(x):
    fft_real = np.fft.rfft(x)
    ps = np.abs(fft_real)**2
    ps_smoothed = gaussian_filter1d(ps, SMOOTHING_SIGMA)
    return ps_smoothed

def get_average_spectrum(x):
    x_ = (x - x.mean()) / x.std()
    spectrum = compute_power_spectrum(x_)
    return spectrum / spectrum.sum()

def power_spectrum_error_per_dim(x_gen, x_true):
    assert x_true.shape[1] == x_gen.shape[1]
    assert x_true.shape[2] == x_gen.shape[2]
    dim_x = x_gen.shape[2]
    pse_per_dim = []
    for dim in range(dim_x):
        spectrum_true = get_average_spectrum(x_true[:, :, dim])
        spectrum_gen = get_average_spectrum(x_gen[:, :, dim])
        hd = hellinger_distance(spectrum_true, spectrum_gen)
        pse_per_dim.append(hd)
    return pse_per_dim
def hellinger_distance(p, q):
    return 1 / np.sqrt(2) * np.sqrt(np.sum((np.sqrt(p) - np.sqrt(q))**2))

def kernel_smoothen(data, kernel_sigma=1):
    """
    Smoothen data with Gaussian kernel
    @param kernel_sigma: standard deviation of gaussian, kernel_size is adapted to that
    @return: internal data is modified but nothing returned
    """
    kernel = get_kernel(kernel_sigma)
    data_final = data.copy()
    data_conv = np.convolve(data[:], kernel)
    pad = int(len(kernel) / 2)
    data_final[:] = data_conv[pad:-pad]
    data = data_final
    return data

def gauss(x, sigma=1):
    return 1 / np.sqrt(2 * np.pi * sigma ** 2) * np.exp(-1 / 2 * (x / sigma) ** 2)

def get_kernel(sigma):
    size = sigma * 10 + 1
    kernel = list(range(size))
    kernel = [float(k) - int(size / 2) for k in kernel]
    kernel = [gauss(k, sigma) for k in kernel]
    kernel = [k / np.sum(kernel) for k in kernel]
    return kernel
batch_size = 32
sequence_length = 1000
dim_val = 64
dim_attn = 64
n_heads = 4
input_size = 64
out_seq_len = 500

x = torch.randn(batch_size, sequence_length, dim_val)

model = Transformer(dim_val, dim_attn, input_size, sequence_length, out_seq_len, n_decoder_layers=2, n_encoder_layers=2, n_heads=n_heads)

generated_time_series = model(x).detach().numpy()

x_true = np.random.randn(batch_size, sequence_length, out_seq_len)
error = power_spectrum_error(generated_time_series, x_true)
print("Power Spectrum Error: ", error)

import numpy as np

data1 = np.load('/Users/sjm_/lorenz/lorenz63_test.npy')
data2 = np.load('/Users/sjm_/Desktop/lorenz63_on0.05_train.npy')
data3 = np.load('/Users/sjm_/Desktop/lorenz96_test.npy')
data4 = np.load('/Users/sjm_/Desktop/lorenz96_on0.05_train.npy')
print(data1.shape)
print(data2.shape)
print(data3.shape)
print(data4.shape)
data2 = data2[:1000].reshape(1, 1000, 3)
data4 = data4[:1000].reshape(1, 1000, 20)
data1 = data1[:1000].reshape(1, 1000, 3)
data3 = data3[:1000].reshape(1, 1000, 20)

model = Transformer(dim_val, dim_attn, input_size, sequence_length, out_seq_len, n_decoder_layers=2, n_encoder_layers=2, n_heads=n_heads)
dim_val_data1 = 3
model_data1 = Transformer(dim_val_data1, dim_attn=3, input_size=3, dec_seq_len=1000, out_seq_len=3)

dim_val_data3 = 20
model_data3 = Transformer(dim_val_data3, dim_attn=20, input_size=20, dec_seq_len=1000, out_seq_len=20)

class MultiHeadAttentionBlock(nn.Module):
    def __init__(self, dim_val, dim_attn, n_heads):
        super(MultiHeadAttentionBlock, self).__init__()
        self.attn = nn.MultiheadAttention(dim_val, n_heads, batch_first=True)

    def forward(self, x, kv=None):
        return self.attn(x, x, x)[0] if kv is not None else self.attn(x, x, x)[0]
class Transformer(nn.Module):
    def __init__(self, dim_val, dim_attn, input_size, dec_seq_len, out_seq_len, n_decoder_layers=1, n_encoder_layers=1, n_heads=1):
        super(Transformer, self).__init__()
        self.dec_seq_len = dec_seq_len
        self.T = out_seq_len
        self.encs = nn.ModuleList([EncoderLayer(dim_val, dim_attn, n_heads) for _ in range(n_encoder_layers)])
        self.decs = nn.ModuleList([DecoderLayer(dim_val, dim_attn, n_heads) for _ in range(n_decoder_layers)])
        self.fc_out = nn.Linear(dim_val, out_seq_len)

    def forward(self, x):
        for enc in self.encs:
            x = enc(x)
        for dec in self.decs:
            x = dec(x)
        output = self.fc_out(x)
        return output
def power_spectrum_error(x_gen, x_true):
    if x_gen.ndim != x_true.ndim:
        raise ValueError(f"Input dimensions do not match: {x_gen.ndim} vs {x_true.ndim}")
    pse_errors_per_dim = power_spectrum_error_per_dim(x_gen, x_true)
    pse_errors = np.array(pse_errors_per_dim).mean(axis=0)
    return torch.tensor(pse_errors, dtype=torch.float32, requires_grad=True)
def power_spectrum_error_per_dim(x_gen, x_true):
    assert x_true.shape[1] == x_gen.shape[1], f"Shape mismatch: {x_true.shape[1]} != {x_gen.shape[1]}"
    assert x_true.shape[2] == x_gen.shape[2], f"Shape mismatch: {x_true.shape[2]} != {x_gen.shape[2]}"
    dim_x = x_gen.shape[2]
    pse_per_dim = []
    for dim in range(dim_x):
        spectrum_true = get_average_spectrum(x_true[:, :, dim])
        spectrum_gen = get_average_spectrum(x_gen[:, :, dim])
        hd = hellinger_distance(spectrum_true, spectrum_gen)
        pse_per_dim.append(hd)
    return pse_per_dim

def train_model(model, train_data, epochs=50, batch_size=32, lr=1e-4):
    train_data = torch.tensor(train_data, dtype=torch.float32)

    dataset = TensorDataset(train_data)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    optimizer = optim.Adam(model.parameters(), lr=lr)

    model.train()
    for epoch in range(epochs):
        epoch_loss = 0
        for batch_idx, (data_batch,) in enumerate(data_loader):
            optimizer.zero_grad()

            inputs = data_batch

            outputs = model(inputs)

            loss = power_spectrum_error(outputs.detach().numpy(), inputs.detach().numpy())
            loss.backward()

            optimizer.step()

            epoch_loss += loss.item()

        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Loss: {epoch_loss / len(data_loader)}")

def test_model(model, test_data):
    test_data = torch.tensor(test_data, dtype=torch.float32)
    model.eval()
    with torch.no_grad():
        output = model(test_data)
        error = power_spectrum_error(output.detach().numpy(), test_data.detach().numpy())
        print(f"Test Power Spectrum Error: {error}")

test_model(model_data1, data1)  # Lorenz 63
test_model(model_data3, data3)  # Lorenz 96
