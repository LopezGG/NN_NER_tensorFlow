# NN_NER_TF
TensorFlow Implementation of "End-to-end Sequence Labeling via Bi-directional LSTM-CNNs-CRF"

The Training can be run with

python BasicTextPreprocessing_CNN_CRF.py

This works on 
- TensorFlow 1.1
- Python 3
- uses glove.6B.100d.gz from https://nlp.stanford.edu/projects/glove/

The training File looks like the files provided in Sample_Data folder(same as the conll 2003 format , with just Doc Start lines removed. each sentence is separated with a blank line in between)

Set the Flags to update paths for training , test or to chnage any parameters. 

The code logs a lot of things. Feel free to
comment those parts especially "predictAccuracyAndWrite" function

The prediction can be run as 

"python test_NER.py --PathToConfig Train_Results/ --modelName model-41000 --TestFilePath Sample_Data/eng.testb.iobes.act_part"

Sample input and predictions for the test run is provided in results folder. Update the path based on the flags in the code
Sample prediction is Sample_Data/test_Predictions_41000.txt

The Evaluation code to calculate Precision and recall per category is in Eval.py. 

in the Eval Change the prediction file name to the one from your test step. the precision and recall were calculated as described 
Tjong Kim Sang, Erik. F. 2002. Introduction to the CoNLL-2003 Shared Task: Language Independent Named Entity Recognition. In Proc. Conference on Natural Language Learning
