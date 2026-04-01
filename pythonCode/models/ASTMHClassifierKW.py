"""
ASTMHClassifier.py

nn.Module for training a 51-class classifier on ASTMH embedded abstracts. 

Author: Olivia Zahn 
"""

import torch
from torch import nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchmetrics import Accuracy

from pytorch_lightning import LightningModule
from sentence_transformers import SentenceTransformer

from loss.lossOverImportantClassesOnly_fn_9april2024 import GreenCatNLLLoss

class ASTMHClassifierKW(LightningModule):
    """
    TO DO: 
        2. Add dropout layers
        3. add functionality to include encoded keywords
    """
    def __init__(self,
                 layer_dims: list,
                 LLM_name: str = 'all-mpnet-base-v1',
                 act: nn.Module = nn.ReLU(),
                 loss: nn.Module = nn.NLLLoss(),
                 lr: float = 10e-5,
                 temperature: float = 1.0,
                 num_classes: int = 51
                 ):
        super(ASTMHClassifierKW, self).__init__()

        self.loss = loss
        self.act = act
        self.lr = lr
        self.t = temperature
        self.num_classes = num_classes
        self.accuracy = Accuracy(task="multiclass", 
                                 num_classes=self.num_classes)

        self.layer_dims = layer_dims
        self.num_hidden_layers = len(self.layer_dims)
        self.gh_fp_fn_list = []
        self.ic_fp_fn_list = []
        self.mal_fp_fn_list = []

        # Import the keywords file and read as list of strings. 
        kws_file = open('/workspace/code/data/keywords_8april2024.txt', 'r')
        kws = kws_file.read()
        kws_list = kws.split(',')
        input_dim = len(kws_list)

        layers = []
        # Add input layer
        layers.append(nn.Linear(input_dim, self.layer_dims[0]))
        layers.append(self.act)
        # Add hidden layers
        for i in range(self.num_hidden_layers - 1):
            layers.append(nn.Linear(self.layer_dims[i],
                                    self.layer_dims[i+1]))
            layers.append(self.act)

        # Add final layer
        layers.append(nn.Linear(self.layer_dims[-1],
                                self.num_classes))

        self.model = nn.Sequential(*layers)

        self.save_hyperparameters()

    def forward(self, x):
        """
        Performs forward pass. 

        Parameters:
            x: torch.Tensor - embedded abstract with dimensions (768,).

        Returns:
            y: torch.Tensor - encoded classification output with dimensions (51,).
        """
        x = self.model(x)
        y = self.logsoftmaxtemp(x)
        return y

    def training_step(self, batch, batch_idx):
        """
        Performs training step. Passes a batch of data through the model, computes the loss, 
        and reports loss using log_trainval_metrics function.

        Parameters:
            batch: tuple - (x,y) pair of data (x) and label (y)
            batch_idx: int - index value corresponding to (x,y) in the Dataset.

        Returns:
            loss: float - Loss value computed using the model nn.Module Loss.
        """
        # Use _shared_eval_step to run data through the model and compute the loss
        loss, acc, _, _, _ = self._shared_eval_step(batch, batch_idx)
        # Report the loss as train_loss metric
        metrics = {"train_loss": loss,
                   "train_accuracy": acc}
        self.log_trainval_metrics(metrics)
        # Return the loss
        return loss

    def validation_step(self, batch, batch_idx):
        """
        Performs validation step. Passes a batch of data through the model, computes the loss, 
        and reports loss using log_trainval_metrics function.

        Parameters:
            batch: tuple - (x,y) pair of data (x) and label (y)
            batch_idx: int - index value corresponding to (x,y) in the Dataset.

        Returns:
            loss: float - Loss value computed using the model nn.Module Loss.
        """
        loss, acc, gh, ic, m = self._shared_eval_step(batch, batch_idx)
        metrics = {"val_loss": loss,
                   "val_accuracy": acc}
        self.log_trainval_metrics(metrics)
        return loss

    def test_step(self, batch, batch_idx):
        """
        Performs evaluation step. Passes a batch of data through the model, computes the loss, 
        and reports loss using log_trainval_metrics function.

        Parameters:
            batch: tuple - (x,y) pair of data (x) and label (y)
            batch_idx: int - index value corresponding to (x,y) in the Dataset.

        Returns:
            loss: float - Loss value computed using the model nn.Module Loss.
        """
        loss, acc, _, _, _ = self._shared_eval_step(batch, batch_idx)
        metrics = {"test_loss": loss,
                   "test_accuracy": acc}
        self.log_dict(metrics)
        return loss
    
    def predict_step(self, batch, batch_idx):
        emb, kw, emb_kw, label = batch # Extract x and y from batch
        preds = self.forward(kw) # Pass x through model
        preds = torch.exp(preds) # Undo log to get the probabilities 
        preds_classes = torch.argmax(preds, axis=-1) #Return predicted classes
        return preds, preds_classes

    def _shared_eval_step(self, batch, batch_idx):
        """
        Common evaluation step. Passes data through model then computes and return 
            loss (CrossEntropyLoss). 

        Parameters:
            batch: tuple - (x,y) pair of data (x) and label (y)
            batch_idx: int - index value corresponding to (x,y) in the Dataset.

        Returns:
            loss: float - Loss value computed using the model nn.Module Loss.
            acc: float - Accuracy computed using torchmetric Accuracy class. 
        """
        emb, kw, emb_kw, label = batch # Extract x and y from batch
        preds = self.forward(kw) # Pass x through model
        probs = torch.exp(preds) # Undo the log to get the probabilities

        loss = self.loss(preds, label)
        acc = self.accuracy(probs, label)

        preds_classes = torch.argmax(preds, axis=-1)

        label_idx = torch.where(label == 5, 1, 0) # Find where there are green class labels
        pred_idx = torch.where(preds_classes == 5, 1, 0) # Find where there are green class predictions
        globalhealth_FNplusFP = torch.sum(torch.where(pred_idx != label_idx, 1, 0)) # Find FN plus FP 

        label_idx = torch.where(label == 8, 1, 0) # Find where there are green class labels
        pred_idx = torch.where(preds_classes == 8, 1, 0) # Find where there are green class predictions
        integratedcontrol_FNplusFP = torch.sum(torch.where(pred_idx != label_idx, 1, 0)) # Find FN plus FP 

        label_idx = torch.where(label == 10, 1, 0) # Find where there are green class labels
        pred_idx = torch.where(preds_classes == 10, 1, 0) # Find where there are green class predictions
        malaria_FNplusFP = torch.sum(torch.where(pred_idx != label_idx, 1, 0)) # Find FN plus FP 

        return loss, acc, globalhealth_FNplusFP, integratedcontrol_FNplusFP, malaria_FNplusFP

    def on_validation_epoch_end(self):
        metrics = {"Global Health FP+FN": sum(self.gh_fp_fn_list),
                   "Integrated Control FP+FN": sum(self.ic_fp_fn_list),
                   "Malaria FP+FN": sum(self.mal_fp_fn_list)}
        self.log_dict(metrics)
        self.gh_fp_fn_list = []
        self.ic_fp_fn_list = []
        self.mal_fp_fn_list = []

    def configure_optimizers(self):
        """
        Configures and returns optimizer used for training.. 
        """
        optimizer = Adam(params=self.model.parameters(), lr=self.lr)
        scheduler = ReduceLROnPlateau(optimizer=optimizer)
        return {"optimizer": optimizer,
                "lr_scheduler": scheduler,
                "monitor": "val_loss"}

    def logsoftmaxtemp(self, x):
        """
        Log-softmax activation function with temperature.
        """
        ex = torch.exp(x/self.t)
        inv_summ = 1/torch.sum(ex, axis=1)
        prod = inv_summ*ex.transpose(0,1)
        return torch.log(prod.transpose(0,1))

    def log_trainval_metrics(self, metrics):
        """
        Logs training time metrics from train DataLoader and validation DataLoader.

        Parameters:
            metrics: dictionary - metrics to log to Lightning Logger (by default, TensorBoard)
        """
        # Note the frequency of metrics reported (epoch, not step)
        # prog_bar: print on progress bar
        # sync_dist: for syncing output across distributed training
        kwargs = {
            'on_step': False, 
            'on_epoch': True, 
            'prog_bar': True, 
            'sync_dist': True,
        }
        self.log_dict(metrics, **kwargs)
    