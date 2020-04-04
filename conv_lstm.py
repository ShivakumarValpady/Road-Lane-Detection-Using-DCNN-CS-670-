import torch
import torch.nn as nn
import segnet.vgg16_decoder as decoder
import segnet.vgg_16_encoder as encoder
from lstm import convlstm

class Conv_LSTM(nn.Module):
    def __init__(self, lstm_hidden_dim:list, lstm_nlayers:int=2, decoder_out_channels:int=1,
                 vgg_decoder_config:list=None):

        super(Conv_LSTM, self).__init__()
        assert lstm_nlayers == len(lstm_hidden_dim)

        self.classes = decoder_out_channels
 
        self.encoder = encoder.VGGencoder()
        self.decoder = decoder.VGGDecoder(decoder_out_channels, config=vgg_decoder_config)
        self.lstm = convlstm.ConvLSTM(input_size=(4, 8), input_dim=512, hidden_dim=lstm_hidden_dim,
                                      kernel_size=(3, 3), num_layers=lstm_nlayers, batch_first=True)

    def forward(self, x:list):
        y = []
        for i, batched_samples in enumerate(x):
            encoded, unpool_indices, unpool_sizes = self.encoder(batched_samples)
            y.append(encoded)
        batched_code_sequence = torch.stack(y, dim=1)
        output, _ = self.lstm(batched_code_sequence)
        output = output[0][:, -1, :, :, :]  # batch size must be first!
        decoded = self.decoder(output, unpool_indices, unpool_sizes)
        return decoded 

