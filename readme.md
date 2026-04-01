README for llmsForAstmh repo           18 March 2026

TO DO: add sample data filenames.

Background:
When authors submit abstracts to ASTMH, they choose a category (there are about 50 categories).
Because there is overlap among the category topics, some author-chosen categories are incorrect. Because abstracts are sent out to the various scientific subcommittees based on category, an incorrect category creates a logistic nightmare because the abstract goes to the wrong subcommittee for review.

Our goal is to train a model, based on past years' abstracts, to predict the correct category. Then when new abstracts come in, they are run through the model. If a predicted and author's categories are discrepant, the abstract can be manually checked and the category possibly changed before being sent out to the various scientific subcommittees for review.

This model has delivered high value to the scientific committee chairs for the past 2 years. But to deliver this value, turnaround time is very short because the classifier is the first step in the review pipeline. We have about a week from when the new abstracts are delivered. This should be fine if the model has been trained and verified beforehand.

-----------------------

Training:

Input: An excel spreadsheet, where each row is a sample that has a title plus abstract (= string of combined length < 3300), and a category (ie ground truth label)

The model steps are:
Cleaning: 
	New data needs to be put into xlsx format
	Categories need to be merged. This means combining a few low-population categories so that each class has a decent number of samples.
	Category names need to be updated from previous years, since these change a little year to year.
	Double-check the categories to catch slight typos, eg "NTD_elimination" vs "NTD-elimination"
	Scripts: 
		parseAbstractContentsIntoDataframeFor2025_18april2025.py
		prepareTrainAndTestDataFor2025Run_18april2025.py
		
Feature generation:
	Run each combined title + abstract through an LLM to generate a feature vector.
	Scripts:
		abstracts2vec.py
	
Train an NN classifier:
	Generate splits.
	NN is not fancy. Found in ASTMHClassifier.py
	Scripts:
		generate_splits.py
		convertSpreadSheetsWithEmbeddingsToNpyFiles_23april2025.py
		train.py 
			ASTMHClassifier.py
				lossOverImportantClassesOnly_fn_9april2024.py
 
Generate results confusion matrix and spreadsheet:
	The results from the splits are combined and plotted as a scatterplot confusion matrix.
	The results can be saved as a spreadsheet.
	Scripts:
		The confusion matrix and spreadsheet blocks are in predict.py  (TO DO: pull these into separate functions)
	
--------------------------------

Prediction:

The new abstracts to be checked arrive. 
Cleaning:
	New data needs to be put into xlsx format
	Categories need to be merged. This means combining a few low-population categories so that each class has a decent number of samples.
	Double-check the categories to see that they all match legit categories. Correct any typos.
	Scripts: 
		parseAbstractContentsIntoDataframeFor2025_18april2025.py
		prepareTrainAndTestDataFor2025Run_18april2025.py
		
Generate features as above.
	Scripts:
		abstracts2vec.py

Run feature vectors through the NN.
	Scripts:
		convertSpreadSheetsWithEmbeddingsToNpyFiles_23april2025.py
		predict.py

Generate results confusion matrix and spreadsheet:
	The new results plotted as a scatterplot confusion matrix.
	The results are saved as a spreadsheet, with author's category and model's 1st and 2nd category and scores.
	Scripts:
		predict.py
		
	The spreadsheet is sent to ASTMH for review.
