import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
import segmentation_models_pytorch as smp
from catalyst.dl.runner.supervised import SupervisedRunner
from catalyst.dl.callbacks import DiceCallback, EarlyStoppingCallback, InferCallback, CheckpointCallback

from dataset import get_dataloader

path2train_csv = 'data/CamVid/train.txt'
path2val_csv = 'data/CamVid/val.txt'
path2test_csv = 'data/CamVid/test.txt'

num_cls = 12

encoder = 'se_resnext50_32x4d'
encoder_weights = 'imagenet'
activation = 'sigmoid'

train_dataloader = get_dataloader(
  path2csv_dataset=path2train_csv,
  num_cls=num_cls,
  encoder=encoder,
  encoder_weights=encoder_weights,
  is_train=True,
  batch_size=8,
  num_workers=12,
)
val_dataloader = get_dataloader(
  path2csv_dataset=path2val_csv,
  num_cls=num_cls,
  encoder=encoder,
  encoder_weights=encoder_weights,
  is_train=False,
  batch_size=1,
  shuffle=False,
  num_workers=4,
)
test_dataloader = get_dataloader(
  path2csv_dataset=path2test_csv,
  num_cls=num_cls,
  encoder=encoder,
  encoder_weights=encoder_weights,
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
    classes=num_cls, 
    activation=activation,
)

num_epochs = 19
logdir = "./logs/segmentation"

optimizer = torch.optim.Adam([
    {'params': model.decoder.parameters(), 'lr': 1e-2}, 
    {'params': model.encoder.parameters(), 'lr': 1e-3},  
])
scheduler = ReduceLROnPlateau(optimizer, factor=0.15, patience=2)
criterion = smp.utils.losses.DiceLoss(eps=1.)
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
