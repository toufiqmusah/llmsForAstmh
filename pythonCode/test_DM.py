import torch

from pytorch_lightning import Trainer

from datamodules.EmbeddingData import EmbeddingData
from models.ASTMHClassifier import ASTMHClassifier

dm = EmbeddingData(num_classes=51) #num_classes=17
dm.setup(stage='eval')
print(dm.val_dataset.__len__())

#model = ASTMHClassifier(layer_dims=[1000, 500, 100])
#pred = model.forward(emb=emb, kw=kw)
#print(pred)
#trainer = Trainer()

#print(model)

#preds = trainer.fit(model=model, datamodule=dm)
#print(len(preds))
#torch.save(preds[0][2], '/workspace/code/pythonCode/loss/sample_model_output/sample_true_classes.pt')
#torch.save(preds[0][0], '/workspace/code/pythonCode/loss/sample_model_output/sample.pt')