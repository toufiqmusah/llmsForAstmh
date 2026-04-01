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

class ASTMHClassifierwKeywords(LightningModule):
    """
    TO DO: 
        2. Add dropout layers
        3. add functionality to include encoded keywords
    """
    def __init__(self,
                 embed_layer_dims: list,
                 kw_layer_dims: list,
                 comb_layer_dims: list,
                 LLM_name: str = 'all-mpnet-base-v1',
                 act: nn.Module = nn.ReLU(),
                 loss: nn.Module = nn.NLLLoss(),
                 lr: float = 10e-5,
                 temperature: float = 1.0,
                 num_classes: int = 51,
                 alpha: float = 0.0
                 ):
        super(ASTMHClassifierwKeywords, self).__init__()

        self.loss = loss
        self.act = act
        self.lr = lr
        self.t = temperature
        self.num_classes = num_classes
        self.accuracy = Accuracy(task="multiclass", 
                                 num_classes=self.num_classes)
        self.alpha = alpha
        
        self.gh_fp_fn_list = []
        self.ic_fp_fn_list = []
        self.mal_fp_fn_list = []

        # Import the keywords file and read as list of strings. 
        kws_file = open('/workspace/code/data/keywords_8april2024.txt', 'r')
        kws = kws_file.read()
        kws_list = kws.split(',')
        num_kws = len(kws_list)

        # Network structure parameters
        embed_num_hidden_layers = len(embed_layer_dims)
        llm = SentenceTransformer(f'sentence-transformers/{LLM_name}')
        embed_input_dim = llm.get_sentence_embedding_dimension()

        kw_num_hidden_layers = len(kw_layer_dims)
        kw_input_dim = num_kws

        comb_num_hidden_layers = len(comb_layer_dims)

    
        embed_layers = []
        # Add input layer
        embed_layers.append(nn.Linear(embed_input_dim, embed_layer_dims[0]))
        embed_layers.append(self.act)
        # Add hidden layers
        for i in range(embed_num_hidden_layers - 1):
            embed_layers.append(nn.Linear(embed_layer_dims[i],
                                    embed_layer_dims[i+1]))
            embed_layers.append(self.act)
        embed_layers.append(nn.Linear(embed_layer_dims[-1],comb_layer_dims[0]))
        embed_layers.append(self.act)

        kw_layers = []
        # Add input layer
        kw_layers.append(nn.Linear(kw_input_dim, kw_layer_dims[0]))
        kw_layers.append(self.act)
        # Add hidden layers
        for i in range(kw_num_hidden_layers - 1):
            kw_layers.append(nn.Linear(kw_layer_dims[i],
                                       kw_layer_dims[i+1]))
            kw_layers.append(self.act)
        kw_layers.append(nn.Linear(kw_layer_dims[-1],comb_layer_dims[0]))
        kw_layers.append(self.act)

        comb_layers = []
        # Add hidden layers
        for i in range(comb_num_hidden_layers - 1):
            comb_layers.append(nn.Linear(comb_layer_dims[i],
                                         comb_layer_dims[i+1]))
            comb_layers.append(self.act)

        # Add output layer
        comb_layers.append(nn.Linear(comb_layer_dims[-1],
                                     self.num_classes))

        self.embed_model = nn.Sequential(*embed_layers)
        self.kw_model = nn.Sequential(*kw_layers)
        self.comb_model = nn.Sequential(*comb_layers)

        self.save_hyperparameters()

    def forward(self, emb, kw):
        """
        Performs forward pass. 

        Parameters:
            x: torch.Tensor - embedded abstract with dimensions (768,).

        Returns:
            y: torch.Tensor - encoded classification output with dimensions (51,).
        """
        # Feed embeddings into embedding model.
        emb_out = self.embed_model(emb)
        # Feed keyword encodings into keyword model. 
        kw_out = self.kw_model(kw)

        # Add the outputs of the embedding and keyword models.
        comb_input = emb_out + kw_out
        # Feed the combined outputs into the combined model. 
        y = self.comb_model(comb_input)

        # Log-softmax activation function with temperature. 
        y = self.logsoftmaxtemp(y)
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
        self.gh_fp_fn_list.append(gh)
        self.ic_fp_fn_list.append(ic)
        self.mal_fp_fn_list.append(m)

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
        preds = self.forward(emb, kw) # Pass x through model
        probs = torch.exp(preds) # Undo log to get the probabilities 
        preds_classes = torch.argmax(probs, axis=-1) #Return predicted classes
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
        preds = self.forward(emb, kw) # Pass x through model
        probs = torch.exp(preds) # Undo the log to get the probabilities

        gcl = self.green_cat_loss(probs, label) 
        loss = self.loss(preds, label)

        total_loss = self.alpha * gcl + loss

        acc = self.accuracy(probs, label)

        preds_classes = torch.argmax(probs, axis=-1)

        label_idx = torch.where(label == 5, 1, 0) # Find where there are green class labels
        pred_idx = torch.where(preds_classes == 5, 1, 0) # Find where there are green class predictions
        globalhealth_FNplusFP = torch.sum(torch.where(pred_idx != label_idx, 1, 0)) # Find FN plus FP 

        label_idx = torch.where(label == 8, 1, 0) # Find where there are green class labels
        pred_idx = torch.where(preds_classes == 8, 1, 0) # Find where there are green class predictions
        integratedcontrol_FNplusFP = torch.sum(torch.where(pred_idx != label_idx, 1, 0)) # Find FN plus FP 

        label_idx = torch.where(label == 10, 1, 0) # Find where there are green class labels
        pred_idx = torch.where(preds_classes == 10, 1, 0) # Find where there are green class predictions
        malaria_FNplusFP = torch.sum(torch.where(pred_idx != label_idx, 1, 0)) # Find FN plus FP 

        return  loss, acc, globalhealth_FNplusFP, integratedcontrol_FNplusFP, malaria_FNplusFP

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
        Configures and returns optimizer used for training.
        """
        optimizer = Adam([{'params': self.embed_model.parameters()},
                          {'params': self.kw_model.parameters()},
                          {'params': self.comb_model.parameters()}],
                          lr=self.lr)
            #params=self.model.parameters(), lr=self.lr)
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
    