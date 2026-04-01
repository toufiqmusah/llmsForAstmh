"""
Given llm-generated embeddings of the abstracts to be predicted (see "USER ENTRIES" below) and 
authors' categories:
    1. run the embeddings through the classifier. 
    2. Generate a result that has author category, model's 1st and 2nd categories, and model 
    scores.
    3. Plot a confusion matrix
    4. Put the relevant results per abstract in a dataframe and save as xlsx spreadsheet

!!! ATTENTION !!! Note the crucial assumption about the order of categories of the model (in its 
final prediction layer) at line 108.
The catch is that torch.unique() might not sort these alphabetically, but we assume that they
are sorted. This means we must add an argin to the model definition to order the classes correctly. 
 
"""
import os
import numpy as np
import torch 
import pandas as pd
from matplotlib import pyplot as plt, rc

from tqdm import tqdm
from pytorch_lightning import Trainer

from models.ASTMHClassifier import ASTMHClassifier

# Changed for 2025 on this local copy, since we're using pre-built embeddings
# from datamodules.EmbeddingDataForPredictOnly_23apr2025 import EmbeddingData 
# For 2026:
from datamodules.EmbeddingDataForPredictOnly_27mar2026 import EmbeddingData


# Comment: "usePreBuiltEmbeddings" is hard-coded in line 57 of "ASTMHClassifier.py" 

#%% Constants for 2025:
# CHKPT_PATH = '/workspace/code/pythonCode/LLM4ASTMH/088dva8c/checkpoints/epoch=778-step=1558.ckpt'

# # For 2025 version:
# embeddingsFolder = r"V:\FAMLI\Results\Olivia\astmh\dataToRunPredictOneTime"
# CHKPT_PATH = os.path.join(embeddingsFolder, "epoch-epoch=03-accuracy-val_accuracy=0.5924.ckpt") 

# saveFolder = r"C:\Users\CharlesDelahunt\GitHubOnHardDrive\LLMForASTMH\results\for2025"
# numClasses = 46  # 2025 has 46 merged categories  (vs 51 in 2024)


# For 2026 version:
embeddingsFolder = "TO BE DEFINED" # r"V:\FAMLI\Results\Olivia\astmh\dataToRunPredictOneTime"
CHKPT_PATH = os.path.join(embeddingsFolder, "TO BE DEFINED")
                          #"epoch-epoch=03-accuracy-val_accuracy=0.5924.ckpt") 

saveFolder = r"C:\Users\CharlesDelahunt\GitHubOnHardDrive\LLMForASTMH\results\for2025"
numClasses = 54  # 2026 has 54 merged categories  

#%% USER ENTRIES !!!!

# ATTENTION!! 2 Things you MUST do!
# 1. Copy the .npy you wish to use into 'embeddingsFolder', then rename it to 
#    'test_embeddings.npy'.
#     Because: the dataset to be predicted is called, canonically, "test_embeddings.npy". 
#     This is defined in "EmbeddingDataset.py" (not to be confused with "EmbeddingData.py").

# 2. Insert the 'testedFilename' into line 85 of 'EmbeddingDataForPredictOnly_27mar2026.py'  

# # For 2025:
# testedFilename = 'embedded_test_split_merged_1_18apr2025.xlsx'
# saveFilename = 'test_split_merged_1.xlsx'
# titleStr = 'val_split_1'

# testedFilename = 'embedded_train_split_merged_1_18apr2025.xlsx'
# saveFilename = 'train_split_merged_1.xlsx'
# titleStr = 'train_split_1'
 
# testedFilename = 'embedded_newAbstractsToReclassify_merged_18apr2025.xlsx'
# saveFilename = 'newAbstracts2025_merged_v2.xlsx'
# titleStr = '2025_abstracts'

# # For 2026: 
testedFilename = 'embedded_newAbstractsToReclassify_2026.xlsx'
saveFilename = 'newAbstracts2026_mergedCats.xlsx'
titleStr = '2026_abstracts'

# # To assess results on a single split:
# # Train results:
# testedFilename = 'embedded_test_split_merged_1_2026.xlsx'
# saveFilename = 'test_split_merged_1.xlsx'
# titleStr = 'val_split_1'

# # Val results:
# testedFilename = 'embedded_train_split_merged_1_2026.xlsx'
# saveFilename = 'train_split_merged_1.xlsx'
# titleStr = 'train_split_1'

# ---------------------------------------------------------------

#%% Load the excel file to get the order of mergedCategories used by the model, and to convert
# the true mergedCategories into these same numbers:
testedData = pd.read_excel(os.path.join(embeddingsFolder, testedFilename))

# trueCats = np.array(testedData['mergedCategory'].values)  # 2025
trueCats = np.array(testedData['shortMergedCat'].values)  # 2026

# To get full list of cats used by the model, we must use train or test, not new to-be-predicted:
    
# 2026:
df = pd.read_excel(r"data\materialsFor2026Version" + 
               r"\combinedAbstractContents_2023_2024_2025_19mar2026.xlsx")
allShortMergedCatsAlphabetical = np.unique(df['shortMergedCat'].values)
catsInModelOrder = allShortMergedCatsAlphabetical  # !!! ATTENTION: WE ASSUME THAT THE MODEL IS 
# SORTING THE SHORT MERGED CATEGORIES ALPHABETICALLY !!! 


# # 2025: 
# temp = pd.read_excel(os.path.join(r"V:\FAMLI\Results\Olivia\astmh",
#                             "embedded_test_split_merged_1_18apr2025.xlsx"))
# catsInModelOrder = temp['mergedCategory'].unique()  # cannot use np.unique(), because torch.unique 
# # might not sort.

# Now give the true cat vals as ints:
trueCatsAsInt = -1 * np.ones(len(testedData), dtype=int)
for i in range(len(catsInModelOrder)):
    inds = np.where(trueCats == catsInModelOrder[i])[0]
    trueCatsAsInt[inds] = i


#%% Run the model:
model = ASTMHClassifier(layer_dims=[500,500,500, 500])
trained_model = ASTMHClassifier.load_from_checkpoint(checkpoint_path=CHKPT_PATH)

# trained_model.eval()
trainer = Trainer()

datamodule = EmbeddingData(num_classes=numClasses)  #  51)  # 46 for 2025 version
datamodule.setup(stage='eval')  # 'fit')
# print(datamodule.val_dataset.__len__())
 
# preds = trainer.predict(model=trained_model, datamodule=datamodule)

# predicted_classes = []
# true_labels = []
# for b in range(len(preds)):
#     predicted_classes.extend(preds[b][2].tolist())
#     true_labels.extend(preds[b][3].tolist())
  
#%% Results:
    
# train_dataset = datamodule.train_dataset   # 23 april 2025
val_dataset = datamodule.val_dataset

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

# train_df = train_dataset.data_df
 
firstPredCatsAsInt = []
firstPredCats = []    # Save 1st and 2nd choices
firstScores = []
secondPredCats = []
secondScores = []
#labels = []
allScores = []

for i in tqdm(range(val_dataset.__len__())):
    emb, kw, emb_kw, label = val_dataset.__getitem__(i)
    pred = trained_model.forward(
        torch.Tensor(emb).to(device))
    prob = torch.exp(pred)
    theseScores = prob.detach().numpy()
    allScores.append(theseScores)
    # Record the first and second choices, plus their scores:
    thisPredCat1AsInt = np.argmax(theseScores)
    firstPredCatsAsInt.append(thisPredCat1AsInt) 
    firstPredCats.append(catsInModelOrder[thisPredCat1AsInt])
    firstScores.append(max(theseScores))
    theseScores[np.argmax(theseScores)] = -1
    thisPredCat2AsInt = np.argmax(theseScores)
    secondPredCats.append(catsInModelOrder[thisPredCat2AsInt])
    secondScores.append(max(theseScores))
    
    thisPredAsInt = torch.argmax(prob, axis=-1).cpu().detach().numpy()
    #labels.append(label)

firstPredCats = np.array(firstPredCats)  # so that indexed replacement works
firstPredCatsAsInt = np.array(firstPredCatsAsInt)
secondPredCats = np.array(secondPredCats)
trueCats = np.array(trueCats)
trueCatsAsInt = np.array(trueCatsAsInt)

# # 2025 only, special fussing: 
# # Because of two kinds of hyphens, there are two malaria-surveillance cats. Combine these:
# old1 = 'Malaria – Surveillance and Data Utilization'  # 44 = index
# new1 = 'Malaria - Surveillance and Data Utilization'  # 24 = index
# old2 = 'Integrated Control Measures for Neglected Tropical Diseases (NTDs)'
# new2 = 'Measures for Control and Elimination of Neglected Tropical Diseases (NTDs)'
# firstPredCats[firstPredCats == old1] = new1
# secondPredCats[secondPredCats == old1] = new1
# trueCats[trueCats == old1] = new1
# firstPredCatsAsInt[firstPredCatsAsInt == 44] = 24 
# trueCatsAsInt[trueCatsAsInt == 44] = 24
# firstPredCats[firstPredCats == old2] = new2
# secondPredCats[secondPredCats == old2] = new2
# trueCats[trueCats == old2] = new2
# firstPredCatsAsInt[firstPredCatsAsInt == 13] = 45 
# trueCatsAsInt[trueCatsAsInt == 13] = 45 

#%% Print a confusion scatterplot:
# 2026: Red lines delineate blocks of priority categories:
redLineLocations = []
jitX = -0.3 + 0.6 * np.random.random(len(trueCats))
jitY = -0.3 + 0.6 * np.random.random(len(trueCats))
yTicks = [catsInModelOrder[i] + '_' + str(i) for i in range(len(catsInModelOrder))]
 
relevantInts = np.unique(trueCatsAsInt)

# # 2025 only:
# yTicks[13]= ''   
# yTicks[44] = '' 
# maxVal = 46
# redLineLocations = [14.5, 26.5, 31,5, 34,5, 42]

# 2026:
maxVal = 55
# priority indices = [6,11,21,22,23,24,25,26,27,28,29,30,31,32,38,39,44,45,46,47,48,49,50,51,52]
# !!! CAUTION !!! We assume that the model output classes are the sorted (ie alphabetical) list of
# shortMergedCats from column "shortMergedCat" of spreadsheet 
# "data\materialsFor2026Version\combinedAbstractContents_2023_2024_2025_19mar2026.xlsx"
redLineLocations = [5.5, 6.5, 10.5, 11.5, 20.5, 32.5, 37.5, 39.5, 43.5]


plt.figure()
#plt.axis('equal')
plt.hlines(y=redLineLocations, xmin=-1, xmax=maxVal, colors='r')
plt.vlines(x=redLineLocations, ymin=-1, ymax=maxVal, colors='r')
plt.plot([0, maxVal], [0, maxVal], 'g:')
plt.plot(relevantInts, -1 * np.ones(len(relevantInts)), color='r', marker='s', linestyle='')
plt.plot( -1 * np.ones(len(relevantInts)), relevantInts, color='r', marker='s', linestyle='')

plt.plot(trueCatsAsInt + jitX,firstPredCatsAsInt + jitY, '*')
plt.xlabel('true Categories',fontweight='bold')
plt.ylabel('predicted Categories', fontweight='bold')
plt.yticks(ticks=range(maxVal), labels=yTicks)
plt.grid()
plt.xlim((-3, maxVal + 1))
plt.ylim((-2, maxVal + 1))
plt.title(titleStr, fontweight='bold')

#%% Create a dataframe with relevant results:
val_df = pd.DataFrame() # val_dataset.data_df
val_df['abstract Id'] = testedData['abstractId'].values
val_df['given Cat'] = trueCats 
val_df['first Pred Cat'] = firstPredCats
val_df['second Pred Cat'] = secondPredCats
val_df['first Score'] = np.round(np.array(firstScores) * 100)
val_df['second Score'] = np.round(np.array(secondScores) * 100)
val_df['agreement'] = val_df['first Pred Cat']==val_df['given Cat']
val_df['score Vector'] = allScores
val_df['abstract Title'] = testedData['title']
val_df['abstract Text'] = testedData['abstractText']

#%% Save df as spreadsheet: 
val_df.to_excel(os.path.join(saveFolder, saveFilename))
# preds = []
# labels = []

# Commented out for 2025 (prediction on test only)
# for i in tqdm(range(train_dataset.__len__())):
#     emb, kw, emb_kw, label = train_dataset.__getitem__(i)
#     pred = trained_model.forward(
#         torch.Tensor(emb).to(device))
#     prob = torch.exp(pred)
#     preds.append(torch.argmax(prob, axis=-1).cpu().detach().numpy())
#     labels.append(label)

# train_df['prediction'] = preds
# train_df['label'] = labels

# train_df['correct'] = train_df['prediction']==train_df['label']

# train_df.to_excel('/workspace/code/results/088dva8c/train_predictions.xlsx')

