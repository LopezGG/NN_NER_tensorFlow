# NN_NER_TF
TensorFlow Implementation of "End-to-end Sequence Labeling via Bi-directional LSTM-CNNs-CRF"

# Summary/notes for this project
If you would want to know my understanding of this paper , please have a look at Notes.pdf. It gives background info and summarizes the whole project. My Prof. wants us to use ACL format. The paper is a course project and not for ACL submission

This works on 
- TensorFlow 1.1
- Python 3
- uses glove.6B.100d.gz from https://nlp.stanford.edu/projects/glove/

# Sample Output:
You may look at the sample output of the final model in the file Sample_Data/test_Predictions_41000.txt

# Training
The Training can be run with

python BasicTextPreprocessing_CNN_CRF.py

The training File looks like the files provided in Sample_Data folder(same as the conll 2003 format , with just Doc Start lines removed. each sentence is separated with a blank line in between)

Set the Flags to update paths for training , test or to change any parameters. 

The code logs a lot of things. Feel free to
comment those parts especially "predictAccuracyAndWrite" function

# Prediction
The prediction can be run as 

"python test_NER.py --PathToConfig Train_Results/ --modelName model-41000 --TestFilePath Sample_Data/eng.testb.iobes.act_part"

Sample input and predictions for the test run is provided in results folder. Update the path based on the flags in the code
Sample prediction is Sample_Data/test_Predictions_41000.txt

# P & R , F1 scores
The Evaluation code to calculate Precision and recall per category is in Eval.py. 

in the Eval Change the prediction file name to the one from your test step. the precision and recall were calculated as described by  
Tjong Kim Sang, Erik. F. 2002. Introduction to the CoNLL-2003 Shared Task: Language Independent Named Entity Recognition. In Proc. Conference on Natural Language Learning

