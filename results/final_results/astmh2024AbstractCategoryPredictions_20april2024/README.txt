READ ME

ASTMH 2024 abstract category prediction model    26 April 2024

For each abstract, the predictive model returns scores (high means most likely) for each of the 16 general categories, with the scores summing to 100 (as we have formatted them here).

In the two spreadsheets: 
	Column H contains the top-scoring category prediction
	Column I contains the second highest-scoring prediction
	(The remainder, if they don't sum to 100, is the sum of the other scores)
	Column J contains the difference between the top and second scores. This is a measure of the model's certainty. For example, if column J is above 90, the second choice is a distant second. If column J is under 10, the model ranks the top and second predictions about equally.
	
The "toReview" spreadsheet contains only those abstracts for which the model's top prediction differed from the submitted category. 
	The abstracts are ranked by decreasing order of column J, ie the abstracts on which the model is most "certain" are at the top, and the (very interesting) abstracts for which the model returned two strong candidates are at the bottom. 
	We grayed out the second predictions when the score difference was greater than 95, to indicate that the second choice is very noisy, though they may still convey useful categorizing information.
	
The two confusion matrices show the spread of predictions for each of the 4 general categories. One confusion matrix shows points, one per abstract. The other gives the counts in each box.

ML team, Global Health Labs, Bellevue, WA
Lead modeler:  Olivia Zahn
Support:	   Sourabh Kulhare, Ishan Shah, Charles Delahunt

