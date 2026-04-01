"""
train.py

Script for training ASTHMClassifier.py with EmbeddingData.py.

Author: Olivia Zahn

!!! ATTENTION !!! 27 march 2026: Please modify the model definition (eg models/ASTMHClassifier.py)
to ensure that the prediction classes of the model are alphabetically ordered. Torch does not do this
automatically.

"""
import os
import random
import argparse

from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

from models.ASTMHClassifier import ASTMHClassifier
from datamodules.EmbeddingData import EmbeddingData

#%% 

cli_description = "Training code for ASTMH abstract classifier."
parser = argparse.ArgumentParser(description=cli_description)

# 17 general classes and 51 classes 

# Data, LLM, and augmentation arguments
parser.add_argument('--data_split', type=str, default='1_41824')
parser.add_argument('--LLM_name', type=str, default='all-mpnet-base-v2')
parser.add_argument('--augmentations', help = '', default = ['synonym_replacement', 'random_insertion', 'random_swap', 'random_deletion', 'random_masking'])
parser.add_argument('--num_aug_words', type=int, default=0)

# Trainer and model arguments
parser.add_argument('-bs', '--batch_size', type=int, default=1024)
parser.add_argument('-lr', '--learning_rate', type=float, default=10e-5)
parser.add_argument('-t', '--temperature', type=float, default=1.0)
parser.add_argument('--layer_dims', nargs='+', help='<Required> Set flag', required=True)
parser.add_argument('--classes', type=int, default=17) #17 or 51
parser.add_argument('--gpus', type=str, help="specify the gpus being used. ", default="0,1") #"0" or "0,1"
parser.add_argument('--dropout', type=float, default=0.4)
parser.add_argument('--alpha', type=float, default=0.0)


# Other arguments
parser.add_argument('--wandb_name', type=str)

# Extract command line arguments
args = parser.parse_args()

# Assgign GPUs
os.environ["CUDA_VISIBLE_DEVICES"]= random.choice(["0","1"])

wandb_logger = WandbLogger(project="LLM4ASTMH",
                           name=args.wandb_name
                           )
dm = EmbeddingData(batch_size=args.batch_size,
                   data_split=args.data_split,
                   LLM_name=args.LLM_name,
                   augmentations=args.augmentations,
                   num_aug_words=args.num_aug_words,
                   num_classes=args.classes,
                   )
model = ASTMHClassifier(layer_dims=list(map(int,args.layer_dims[0].split())),
                        lr=args.learning_rate,
                        temperature=args.temperature,
                        num_classes=args.classes,
                        dropout = args.dropout,
                        alpha=args.alpha)

wandb_logger.watch(model=model, log_freq=30, log='all')
print(model)

trainer = Trainer(max_epochs=2000,
                  accelerator='gpu',
                  devices=[0],
                  logger=wandb_logger,
                  callbacks=[EarlyStopping(monitor="val_loss", mode="min", patience=30)],
                  )

trainer.fit(model=model, datamodule=dm)
