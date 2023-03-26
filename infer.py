import os
import yaml
import argparse
import numpy as np
from pathlib import Path

from torch.nn.functional import interpolate

from models import *
from experiment import VAEXperiment
import torch.backends.cudnn as cudnn
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.utilities.seed import seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from dataset import VAEDataset
from pytorch_lightning.plugins import DDPPlugin

from torch import optim, nn, utils, Tensor
from torchvision.datasets.folder import default_loader
from torchvision import transforms
import torchvision.utils as vutils

parser = argparse.ArgumentParser(description='Generic runner for VAE models')
parser.add_argument('--config',  '-c',
                    dest="filename",
                    metavar='FILE',
                    help =  'path to the config file',
                    default='configs/vae.yaml')

args = parser.parse_args()
with open(args.filename, 'r') as file:
    try:
        config = yaml.safe_load(file)
    except yaml.YAMLError as exc:
        print(exc)


tb_logger =  TensorBoardLogger(save_dir=config['logging_params']['save_dir'],
                               name=config['model_params']['name'],)

# For reproducibility
seed_everything(config['exp_params']['manual_seed'], True)
model = vae_models[config['model_params']['name']](**config['model_params'])
# experiment = VAEXperiment(model,
#                           config['exp_params'])

# load checkpoint
#checkpoint = "/root/autodl-tmp/hlhdata/CelebA/log/VanillaVAE/version_4/checkpoints/last.ckpt"
checkpoint = "/root/autodl-tmp/hlhdata/CelebA/log/VanillaVAE/version_12/checkpoints/epoch=177-step=76539.ckpt"
autoencoder = VAEXperiment.load_from_checkpoint(checkpoint, vae_model=model,
                          params=config['exp_params'])

# choose your trained nn.Module
encoder = autoencoder.model.encoder
encoder.eval()

#unic = "4E00"#一
#unic = "4E8C"#二
#unic = "4E09"#三
# embed 1 fake pre-processed images!
# img = default_loader("/home/ubuntu/code/ttf_vae/all_pngs/uni6728_simfang.png") # mu
img = default_loader("/root/autodl-tmp/hlhdata/ttftopng/simfang_png/uni4E00_simfang.png") # yi
img1 = default_loader("/root/autodl-tmp/hlhdata/ttftopng/uni4E00_middle.png")
#img1 = default_loader("/root/autodl-tmp/hlhdata/ttftopng/STXINGKA_png/uni4E00_STXINGKA.png") # yi

img_transforms = transforms.Compose([
                                              transforms.CenterCrop(148),
                                              transforms.Resize(config['data_params']['patch_size']),
                                              transforms.Grayscale(num_output_channels=1),
                                              transforms.ToTensor(),]) #(H*W*C)[0,255] -> (C*H*W)[0.0,1.0]
img = img_transforms(img)#torch.Size([1, 64, 64])
img1 = img_transforms(img1)#torch.Size([1, 64, 64])
imgs = torch.stack((img, img1))#torch.Size([2,1, 64, 64])

#白色1.0
same_elements = (img == 1.0) & (img1 == 1.0)
same_indices = torch.nonzero(same_elements)

fake_image_batch = imgs
# fake_image_batch = torch.rand(1, 1, 64, 64)
[mu, log_var] = autoencoder.model.encode(fake_image_batch)

#print(mu.size(),log_var.size());#torch.Size([2, 128])

embeddings = autoencoder.model.reparameterize(mu, log_var)
print("⚡" * 20, "\nPredictions (32 image embeddings):\n", embeddings.shape, "\n", "⚡" * 20)#[2,128]

embeddings = embeddings.clone().detach().reshape(1, 2, 128)#[1,2,128]
# embeddings = embeddings.permute(0, 2, 1)
# embeddings = torch.nn.functional.interpolate(embeddings, 10, mode='linear')
# embeddings = embeddings.permute(0, 2, 1)
# embeddings = embeddings.clone().detach().reshape(10, 128)

final_latent = embeddings[0][0]
latents_arr = [final_latent]

for i in range(1, 9):
    insert_latent = torch.lerp(embeddings[0][0], embeddings[0][1], 0.1*i)
    latents_arr.append(insert_latent)
latents_arr.append(embeddings[0][1])

embeddings = torch.stack(latents_arr)

img = img.clone().detach().reshape(1, 1, 64, 64)
img1 = img1.clone().detach().reshape(1, 1, 64, 64)
decoded_imgs = autoencoder.model.decode(embeddings)
decoded_imgs = torch.cat((img, decoded_imgs, img1))

'''folder_path = "yi1"
if not os.path.exists(folder_path):
    os.makedirs(folder_path)

for i in range(decoded_imgs.shape[0]):
    for index in same_indices:
        decoded_imgs[i][tuple(index)] = 1.0
    file_name = f"yi_test_{i}.png"
    file_path = os.path.join(folder_path, file_name)
    vutils.save_image(decoded_imgs[i], file_path, normalize=True)'''

for i in range(decoded_imgs.shape[0]):
    for index in same_indices:
        decoded_imgs[i][tuple(index)] = 1.0

for j in range(decoded_imgs.shape[0]):
    decoded_imgs[i][(decoded_imgs[i] > 0) & (decoded_imgs[i] < 1)]=0

vutils.save_image(decoded_imgs.data,
                  "yimiddle.png",
                  normalize=True,
                  nrow=12)

