import torch

from loss.lossOverImportantClassesOnly_fn_9april2024 import GreenCatNLLLoss

outputs = torch.load('/workspace/code/pythonCode/loss/sample_model_output/sample.pt')
targets = torch.load('/workspace/code/pythonCode/loss/sample_model_output/sample_true_classes.pt')

print(outputs.shape)
print(targets.shape)

lossfn = GreenCatNLLLoss()
loss = lossfn(outputs, targets)

print(loss)