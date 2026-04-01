"""
train.py

Script for training ASTHMClassifier.py with EmbeddingData.py.

Author: Olivia Zahn
"""
import argparse

from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

from models.ASTMHClassifierwKeywords import ASTMHClassifierwKeywords
from datamodules.EmbeddingData import EmbeddingData

cli_description = "Training code for ASTMH abstract classifier."
parser = argparse.ArgumentParser(description=cli_description)

# 17 general classes and 51 classes 

# Data, LLM, and augmentation arguments
parser.add_argument('--data_split', type=str, default='merged_1_32624')
parser.add_argument('--LLM_name', type=str, default='all-mpnet-base-v2')
#parser.add_argument('--augmentation', type=str, default='None')
parser.add_argument('--num_aug_words', type=int, default=0)
parser.add_argument('--w0', type=int, default=1)
parser.add_argument('--w1', type=int, default=0)

# Trainer and model arguments
parser.add_argument('-bs', '--batch_size', type=int, default=1024)
parser.add_argument('-lr', '--learning_rate', type=float, default=10e-5)
parser.add_argument('-t', '--temperature', type=float, default=0.0)
parser.add_argument('--embed_layer_dims', nargs='+', help='<Required> Set flag', required=True)
parser.add_argument('--kw_layer_dims', nargs='+', help='<Required> Set flag', required=True)
parser.add_argument('--comb_layer_dims', nargs='+', help='<Required> Set flag', required=True)
parser.add_argument('--classes', type=int, default=17) #17 or 51

# Other arguments
parser.add_argument('--wandb_name', type=str)

# Extract command line arguments
args = parser.parse_args()

wandb_logger = WandbLogger(project="LLM4ASTMH",
                           name=args.wandb_name
                           )

dm = EmbeddingData(batch_size=args.batch_size,
                   data_split=args.data_split,
                   LLM_name=args.LLM_name,
                   #augmentation=args.augmentation,
                   num_aug_words=args.num_aug_words,
                   num_classes=args.classes,
                   w0=args.w0,
                   w1=args.w1,
                   )

model = ASTMHClassifierwKeywords(embed_layer_dims=list(map(int,args.embed_layer_dims[0].split())),
                                 kw_layer_dims=list(map(int,args.kw_layer_dims[0].split())),
                                 comb_layer_dims=list(map(int,args.comb_layer_dims[0].split())),
                                 lr=args.learning_rate,
                                 temperature=args.temperature,
                                 num_classes=args.classes,
                                 )
#test
trainer = Trainer(max_epochs=1000,
                  accelerator='gpu',
                  devices=[1],
                  logger=wandb_logger,
                  callbacks=[EarlyStopping(monitor="val_loss", 
                                           mode="min",
                                           patience=10)],
                  )

trainer.fit(model=model, 
            datamodule=dm)
