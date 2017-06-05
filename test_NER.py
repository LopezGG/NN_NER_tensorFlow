
#usage python test_NER.py --PathToConfig /data/gilopez/Tf_LSTM_CRF/runs/1494895894/ --modelName model-41000
import tensorflow as tf
import utils as utils
import data_processor as dp

#Alphabet maps objects to integer ids
from alphabet import Alphabet
import network as network
import aux_network_func as af
#import pickle
import dill

import numpy as np
import os
import time
import datetime

cwd = os.getcwd()

tf.flags.DEFINE_string("modelName", 'model', "Name of model (default: model)")
tf.flags.DEFINE_string("PathToConfig", cwd, "Path to the directory where config file is stored (default: model)")
tf.flags.DEFINE_string("TestFilePath", "eng.testa.iobes.act.part", "Path to the directory where config file is stored (default: eng.testa.iobes.act)")
FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()

configFile = os.path.abspath(os.path.join(FLAGS.PathToConfig ,"config.pkl"))
print(configFile)
FlagsDict = dill.load(open(configFile,'rb'))

modelName = FLAGS.modelName

path_to_models = FlagsDict['checkpoint_dir']
#path_to_models = "/data/gilopez/Tf_LSTM_CRF/runs/1494820638/checkpoints"
logger = utils.get_logger("EvalCode")

word_alphabet = dill.load(open(os.path.abspath(os.path.join(FLAGS.PathToConfig ,'word_alphabet.pkl')),'rb')) 
char_alphabet = dill.load(open(os.path.abspath(os.path.join(FLAGS.PathToConfig ,'char_alphabet.pkl')),'rb'))  
label_alphabet = dill.load(open(os.path.abspath(os.path.join(FLAGS.PathToConfig ,'label_alphabet.pkl')),'rb')) 

test_path = FLAGS.TestFilePath
timestamp = str(int(time.time()))
out_dir = os.path.abspath(os.path.join(os.path.curdir, "test", timestamp))
if not os.path.exists(out_dir):
        os.makedirs(out_dir)


with tf.Session() as sess:
	sess.run(tf.global_variables_initializer()) #always have this before you load , else it reinitializes the  model to zero state. 
	new_saver  = tf.train.import_meta_graph(os.path.abspath(os.path.join(path_to_models, modelName+".meta")))
	new_saver.restore(sess, os.path.abspath(os.path.join(path_to_models, modelName)))
	
	# Access saved Variables directly. Get a list of variables by using #to get all keys
	#All_varaibles = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
	#Below code verifies that the model was loaded correctly
	#print(sess.run('output/b_out:0'))
	# to save any variable do :
	#Embedding = sess.run(graph.get_tensor_by_name('word_embedding/W_word:0'))
	logger.info("Running Test ....")
	af.test_step_report(logger,sess,
		PadZeroBegin=FlagsDict['PadZeroBegin'],max_length = FlagsDict['sequence_length'],
		test_path=test_path, dropout_keep_prob=FlagsDict['dropout_keep_prob'],step=1,
		out_dir=out_dir,char_alphabet=char_alphabet,label_alphabet=label_alphabet,
		word_alphabet=word_alphabet,word_column=FlagsDict['word_col'], label_column=FlagsDict['label_col'], char_embedd_dim=FlagsDict['char_embedd_dim'],max_char_per_word=FlagsDict['max_char_per_word'])

	''' This is for interpretability
	new_accuracy_test,accuracy_low_classes_test= af.test_step_eval(logger,sess,
		PadZeroBegin=FlagsDict['PadZeroBegin'],max_length = FlagsDict['sequence_length'],
		test_path=test_path, dropout_keep_prob=FlagsDict['dropout_keep_prob'],step=1,
		out_dir=out_dir,char_alphabet=char_alphabet,label_alphabet=label_alphabet,
		word_alphabet=word_alphabet,word_column=FlagsDict['word_col'], label_column=FlagsDict['label_col'], char_embedd_dim=FlagsDict['char_embedd_dim'],max_char_per_word=FlagsDict['max_char_per_word'])

