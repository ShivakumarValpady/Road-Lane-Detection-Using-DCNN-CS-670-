import torch
import torch.nn as nn
import training as tu
from conv_lstm import Conv_LSTM
from setup import device
from setup import Configs
import setup
from data import dataset
from torch.utils.data import DataLoader
from data import show_plain_images


if __name__ == '__main__':
    configs = Configs()
    print("Loading model")
    model = Conv_LSTM(configs.hidden_dims, decoder_out_channels=2, lstm_nlayers=len(configs.hidden_dims), vgg_decoder_config=configs.decoder_config)
    tu.load_model_checkpoint(model, 'train-results/model.torch', inference=False, map_location=device)
    print("Loading Completed Succesfully")
    tu_test_dataset = dataset(setup.test_dir,setup.test_sub,setup.test_labels)
    tu_dataloader = DataLoader(tu_test_dataset, batch_size=2, shuffle=True)
    model.train()
    with torch.no_grad():

        for batchno, (frames, targets) in enumerate(tu_dataloader):
            output = model(frames)
            targets_ = targets.squeeze(1).long()

            print("Loss:", nn.CrossEntropyLoss(weight=torch.FloatTensor(configs.loss_weights))(output, targets_))
            output = (torch.sigmoid(output[:, 1, :, :]) > .5).float()
            print("Output max:", output.max().item(), "Output mean", output.mean().item())
            print("Pixel lane points:", targets.sum().item(), output.sum().item())
            for i in range(2):
                samples = []
                for j in range(len(frames)):
                    a, b = torch.chunk(frames[j], 2, dim=0)
                    if i == 0:
                        samples.append(a.squeeze())
                    else:
                        samples.append(b.squeeze())

                o = samples[0].permute(1, 2, 0).numpy() + targets[i].permute(1, 2, 0).numpy()
                show_plain_images(samples + [targets[i]] + [output[i].unsqueeze(0)], len(samples) + 2, save=True,
                                  fname=f'visual_{batchno}-{i}.png')
