import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
import segmentation_models_pytorch as smp
from catalyst.dl.runner.supervised import SupervisedRunner
from catalyst.dl.callbacks import DiceCallback, EarlyStoppingCallback, InferCallback, CheckpointCallback

from dataset import get_datloader

root_dir = './data'
path2train_csv = './data/train.csv'
path2val_csv = './data/val.csv'
path2test_csv = './data/test.csv'

num_cls2name = {1: 'car'}

encoder = 'se_resnext50_32x4d'
encoder_weights = 'imagenet'
activation = 'sigmoid'

train_dataloader = get_datloader(
  path2train_csv=path2train_csv,
  num_cls2name=num_cls2name,
  encoder=encoder,
  encoder_weigchts=encoder_weigchts,
  root_dir=root_dir,
  is_train=True,
  batch_size=8,
  num_workers=12,
)
val_dataloader = (
  path2val_csv=path2val_csv,
  num_cls2name=num_cls2name,
  encoder=encoder,
  encoder_weigchts=encoder_weigchts,
  root_dir='./data',
  is_train=False,
  batch_size=1,
  shuffle=False,
  num_workers=4,
)
test_dataloader = (
  path2val_csv=path2test_csv,
  num_cls2name=num_cls2name,
  encoder=encoder,
  encoder_weigchts=encoder_weigchts,
  root_dir='./data',
  is_train=False,
  batch_size=1,
  shuffle=False,
  num_workers=4,
)

loaders = {
  "train": train_dataloader,
  "valid": val_dataloader,
  "test": test_dataloader,
}

model = smp.FPN(
    encoder_name=encoder, 
    encoder_weights=encoder_weights, 
    classes=len(num_cls2name), 
    activation=activation,
)

num_epochs = 19
logdir = "./logs/segmentation"

optimizer = torch.optim.Adam([
    {'params': model.decoder.parameters(), 'lr': 1e-2}, 
    {'params': model.encoder.parameters(), 'lr': 1e-3},  
])
scheduler = ReduceLROnPlateau(optimizer, factor=0.15, patience=2)
criterion = smp.utils.losses.BCEDiceLoss(eps=1.)
runner = SupervisedRunner()

runner.train(
    model=model,
    criterion=criterion,
    optimizer=optimizer,
    scheduler=scheduler,
    loaders=loaders,
    callbacks=[DiceCallback(), EarlyStoppingCallback(patience=5, min_delta=0.001)],
    logdir=logdir,
    num_epochs=num_epochs,
    verbose=True
)


