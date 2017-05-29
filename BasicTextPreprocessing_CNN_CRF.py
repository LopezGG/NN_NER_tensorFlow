
# coding: utf-8

# In[1]:

import tensorflow as tf
import utils as utils
import aux_network_func as af
import data_processor as dp
#Alphabet maps objects to integer ids
from alphabet import Alphabet
import network as network

import dill

import numpy as np
import os
import time
import datetime
from tensorflow.python import debug as tf_debug
# In[2]:

tf.__version__

#usage : python BasicTextPreprocessing_CNN_CRF.py 
#here 'word' is the name of the alphabet class instance
print("Loading data...")
word_alphabet = Alphabet('word')
#'label_name' is 'pos' or 'ner'
label_name ="ner"
label_alphabet = Alphabet(label_name)
logger = utils.get_logger("MainCode")
embedding = "glove"
embedding_path = "glove.6B.100d.gz"



oov = 'embedding'
fine_tune = True
# Model Hyperparameters
#tf.flags.DEFINE_integer("embedding_dim", 128, "Dimensionality of character embedding (default: 128)") #not used 
tf.flags.DEFINE_string("train_path", "eng.train.iobes.act", "Train Path")
tf.flags.DEFINE_string("test_path", "eng.testa.iobes.act", "Test Path")
tf.flags.DEFINE_string("dev_path", "eng.testb.iobes.act", "dev Path")
tf.flags.DEFINE_float("dropout_keep_prob", 0.5, "Dropout keep probability (default: 0.5)")
tf.flags.DEFINE_float("grad_clip", 5, "value for gradient clipping to avoid exploding/vanishing gradient(default: 5.0) in LSTM")
tf.flags.DEFINE_float("max_global_clip", 5.0, "value for gradient clipping to avoid exploding/vanishing gradient overall(default: 1.0)")


# Training parameters
tf.flags.DEFINE_integer("batch_size", 10, "Batch Size (default: 64)")
tf.flags.DEFINE_integer("word_col", 0, "position of the word in input file (default: 0)")
tf.flags.DEFINE_integer("label_col", 3, "position of the label in input file (default: 3)")
tf.flags.DEFINE_integer("n_hidden_LSTM", 200, "Number of hidden units in LSTM (default: 200)")
tf.flags.DEFINE_integer("num_epochs", 50, "Number of training epochs (default: 200)")
tf.flags.DEFINE_integer("num_filters", 30, "Number of filters to apply for char CNN (default: 30)") 
tf.flags.DEFINE_integer("filter_size", 3, "filter_size (default: 3 )")
tf.flags.DEFINE_integer("evaluate_every", 1000, "Evaluate model on dev set after this many steps (default: 100)")
tf.flags.DEFINE_integer("char_embedd_dim", 30, "char_embedd_dim(default: 30)")
tf.flags.DEFINE_integer("Optimizer", 1, "Adam : 1 , SGD:2")
tf.flags.DEFINE_integer("num_checkpoints", 5, "Number of checkpoints to store (default: 5)")
tf.flags.DEFINE_float("starter_learning_rate", 0.015, "Initial learning rate for the optimizer. (default: 1e-3)")
tf.flags.DEFINE_float("decay_rate", 0.05, "How much to decay the learning rate. (default: 0.015)")

# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")
tf.flags.DEFINE_boolean("PadZeroBegin", False, "where to pad zero in the input")
FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()
Flags_Dict= utils.print_FLAGS(FLAGS,logger)

train_path = Flags_Dict.train_path
test_path = Flags_Dict.test_path
dev_path = Flags_Dict.dev_path

word_column = FLAGS.word_col
label_column = FLAGS.label_col
# Output directory for models and summaries
timestamp = str(int(time.time()))
out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
print("Writing to {}\n".format(out_dir))
# read training data
logger.info("Reading data from training set...")
word_sentences_train, _, word_index_sentences_train, label_index_sentences_train = dp.read_conll_sequence_labeling(
    train_path, word_alphabet, label_alphabet, word_column, label_column,out_dir=out_dir)

# if oov is "random" and do not fine tune, close word_alphabet
if oov == "random" and not fine_tune:
    logger.info("Close word alphabet.")
    word_alphabet.close()
    

# read dev data
logger.info("Reading data from dev set...")
word_sentences_dev, _, word_index_sentences_dev, label_index_sentences_dev = dp.read_conll_sequence_labeling(
    dev_path, word_alphabet, label_alphabet, word_column, label_column)

# close alphabets : by close we mean we cannot add any more words to the word vocabulary. 
#To DO :change to close this after train set alone
word_alphabet.close()
label_alphabet.close()


# we are doing a -1 because we did not use the zer index. I believe this is to account for unknown word
logger.info("word alphabet size: %d" % (word_alphabet.size() - 1))
logger.info("label alphabet size: %d" % (label_alphabet.size() - 1))
# get maximum length : this is mainly for padding. 
max_length_train = utils.get_max_length(word_sentences_train)
max_length_dev = utils.get_max_length(word_sentences_dev)
#max_length_test = utils.get_max_length(word_sentences_test)
max_length = min(dp.MAX_LENGTH, max(max_length_train, max_length_dev))
logger.info("Maximum length (i.e max words ) of training set is %d" % max_length_train)
logger.info("Maximum length (i.e max words ) of dev set is %d" % max_length_dev)
#logger.info("Maximum length (i.e max words ) of test set is %d" % max_length_test)
logger.info("Maximum length (i.e max words ) used for training is %d" % max_length)

logger.info("Padding training text and lables ...")
word_index_sentences_train_pad,train_seq_length = utils.padSequence(word_index_sentences_train,max_length, beginZero=FLAGS.PadZeroBegin)
label_index_sentences_train_pad,_= utils.padSequence(label_index_sentences_train,max_length, beginZero=FLAGS.PadZeroBegin)

logger.info("Padding dev text and lables ...")
word_index_sentences_dev_pad,dev_seq_length = utils.padSequence(word_index_sentences_dev,max_length, beginZero=FLAGS.PadZeroBegin)
label_index_sentences_dev_pad,_= utils.padSequence(label_index_sentences_dev,max_length, beginZero=FLAGS.PadZeroBegin)

logger.info("Creating character set FROM training set ...")
char_alphabet = Alphabet('character')
char_index_train,max_char_per_word_train= dp.generate_character_data(word_sentences_train,  
                                    char_alphabet=char_alphabet,setType="Train")
# close character alphabet. WE close it because the embed table is goign to be random
char_alphabet.close()

logger.info("Creating character set FROM dev set ...")
char_index_dev,max_char_per_word_dev= dp.generate_character_data(word_sentences_dev, 
                                    char_alphabet=char_alphabet, setType="Dev")


logger.info("character alphabet size: %d" % (char_alphabet.size() - 1))
max_char_per_word = min(dp.MAX_CHAR_PER_WORD, max_char_per_word_train,max_char_per_word_dev)
logger.info("Maximum character length is %d" %max_char_per_word)
logger.info("Constructing embedding table ...")
#TODO : modify network to use this
char_embedd_table = dp.build_char_embedd_table(char_alphabet,char_embedd_dim=FLAGS.char_embedd_dim)

logger.info("Padding Training set ...")
char_index_train_pad = dp.construct_padded_char(char_index_train, char_alphabet, max_sent_length=max_length,max_char_per_word=max_char_per_word)
logger.info("Padding Dev set ...")
char_index_dev_pad = dp.construct_padded_char(char_index_dev, char_alphabet, max_sent_length=max_length,max_char_per_word=max_char_per_word)

#logger.info("Generating data with fine tuning...")
embedd_dict, embedd_dim, caseless = utils.load_word_embedding_dict(embedding, embedding_path,logger)
logger.info("Dimension of embedding is %d, Caseless: %d" % (embedd_dim, caseless))
#Create an embedding table where if the word from training/train/dev set is in glove , then assign glove values else assign random values
embedd_table = dp.build_embedd_table(word_alphabet, embedd_dict, embedd_dim, caseless)
word_vocab = word_alphabet.instances
word_vocab_size = len(word_vocab)
char_vocab = char_alphabet.instances
char_vocab_size = len(char_vocab)
num_classes = len(label_alphabet.instances) + 1 #to account for zero index we dont use
#logger.info("length of the embedding table is  %d" , embedd_table.shape[0])

#Store the parameters for loading in test set
Flags_Dict['sequence_length']=max_length
Flags_Dict['num_classes']=num_classes
Flags_Dict['word_vocab_size']=word_vocab_size
Flags_Dict['char_vocab_size']=char_vocab_size
Flags_Dict['max_char_per_word']=max_char_per_word
Flags_Dict['embedd_dim']=embedd_dim
Flags_Dict['out_dir']=out_dir
Flags_Dict['model_path']=out_dir
# Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
Flags_Dict['checkpoint_dir']=checkpoint_dir
dill.dump(Flags_Dict,open(os.path.join(out_dir, "config.pkl"),'wb')) 
dill.dump(char_alphabet,open(os.path.join(out_dir, "char_alphabet.pkl"),'wb')) 
dill.dump(word_alphabet,open(os.path.join(out_dir, "word_alphabet.pkl"),'wb')) 
dill.dump(label_alphabet,open(os.path.join(out_dir, "label_alphabet.pkl"),'wb')) 
tf.reset_default_graph()

session_conf = tf.ConfigProto(
  allow_soft_placement=FLAGS.allow_soft_placement,
  log_device_placement=FLAGS.log_device_placement)
with tf.Session(config=session_conf) as sess:
    best_accuracy = 0 
    best_overall_accuracy = 0
    best_accuracy_test = 0 
    best_overall_accuracy_test = 0
    best_step = 0
    BiLSTM = network.textBiLSTM(sequence_length=max_length, num_classes=num_classes, word_vocab_size=word_vocab_size,
      word_embedd_dim=embedd_dim,n_hidden_LSTM =FLAGS.n_hidden_LSTM,max_char_per_word=max_char_per_word,
      char_vocab_size=char_vocab_size,char_embedd_dim = FLAGS.char_embedd_dim,grad_clip=FLAGS.grad_clip,num_filters=FLAGS.num_filters,filter_size= FLAGS.filter_size)


    # Define Training procedure
    global_step = tf.Variable(0, name="global_step", trainable=False)
    decay_step = int(len(word_index_sentences_train_pad)/FLAGS.batch_size) #we want to decay per epoch. Comes to around 1444 for batch of 100
    #print("decay_step :",decay_step)
    learning_rate = tf.train.exponential_decay(FLAGS.starter_learning_rate, global_step,decay_step, FLAGS.decay_rate, staircase=True)
    if(FLAGS.Optimizer==2):
        optimizer = tf.train.GradientDescentOptimizer(learning_rate) #also try GradientDescentOptimizer , AdamOptimizer
    elif(FLAGS.Optimizer==1):
        optimizer = tf.train.AdamOptimizer(learning_rate)
    
    #This is the first part of minimize()
    grads_and_vars = optimizer.compute_gradients(BiLSTM.loss)
    #clipped_grads_and_vars = [(tf.clip_by_norm(grad, FLAGS.max_global_clip), var) for grad, var in grads_and_vars]


    #we will do grad_clipping for LSTM only
    #capped_gvs = [(tf.clip_by_value(grad, -FLAGS.max_global_clip, FLAGS.max_global_clip), var) for grad, var in grads_and_vars]

    # the following bloack is a hack for clip by norm
    #grad_list = [grad for grad, var in grads_and_vars]
    #var_list =  [var for grad, var in grads_and_vars]
    #capped_gvs = tf.clip_by_global_norm(grad_list, clip_norm=FLAGS.max_global_norm)
    #grads_and_vars_pair = zip(capped_gvs,var)


    #This is the second part of minimize()
    train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)
    # Keep track of gradient values and sparsity (optional)
    grad_summaries = []
    for g, v in grads_and_vars:
        if g is not None:
            grad_hist_summary = tf.summary.histogram("{}/grad/hist".format(v.name), g)
            sparsity_summary = tf.summary.scalar("{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))
            grad_summaries.append(grad_hist_summary)
            grad_summaries.append(sparsity_summary)
    grad_summaries_merged = tf.summary.merge(grad_summaries)
    

    
    # Summaries for loss and accuracy
    loss_summary = tf.summary.scalar("loss", BiLSTM.loss)
    #acc_summary = tf.summary.scalar("accuracy", BiLSTM.accuracy)  

    # Train Summaries
    train_summary_op = tf.summary.merge([loss_summary, grad_summaries_merged])
    train_summary_dir = os.path.join(out_dir, "summaries", "train")
    train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)

    # Dev summaries
    dev_summary_op = tf.summary.merge([loss_summary])
    dev_summary_dir = os.path.join(out_dir, "summaries", "dev")
    dev_summary_writer = tf.summary.FileWriter(dev_summary_dir, sess.graph)



    checkpoint_prefix = os.path.join(checkpoint_dir, "model")
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    saver = tf.train.Saver(tf.global_variables(), max_to_keep=FLAGS.num_checkpoints)
    
    # variables need to be initialized before we can use them
    sess.run(tf.global_variables_initializer())

    #debug block
    #sess = tf_debug.LocalCLIDebugWrapperSession(sess)
    #sess.add_tensor_filter("has_inf_or_nan", tf_debug.has_inf_or_nan)
        

    
    def dev_step (session,BiLSTM,PadZeroBegin,max_length,x_batch,y_batch,act_seq_lengths,
        dropout_keep_prob,embedd_table,step,char_batch,char_embedd_table,writer= None):
        feed_dict=af.create_feed_Dict(BiLSTM,PadZeroBegin,max_length,x_batch,y_batch,act_seq_lengths,dropout_keep_prob,embedd_table,char_batch,char_embedd_table)
        logits, transition_params,summaries = session.run([BiLSTM.logits, BiLSTM.transition_params,dev_summary_op],feed_dict=feed_dict)
        accuracy,accuracy_low_classes = af.predictAccuracyAndWrite(logits,transition_params,act_seq_lengths,y_batch,step,x_batch,word_alphabet,label_alphabet,beginZero=FLAGS.PadZeroBegin)

        time_str = datetime.datetime.now().isoformat()
        print("{}: step {},  accuracy on set {:g}, accuracy for classes except Others: {:g}".format(time_str, step,accuracy,accuracy_low_classes))
        if writer:
                writer.add_summary(summaries, step)
        return accuracy,accuracy_low_classes

    def train_step(session,BiLSTM,PadZeroBegin,max_length,x_batch, y_batch,act_seq_lengths,dropout_keep_prob,embedd_table,char_batch,char_embedd_table):
        """
        A single training step
        """
        feed_dict=af.create_feed_Dict(BiLSTM,PadZeroBegin,max_length,x_batch,y_batch,act_seq_lengths,dropout_keep_prob,embedd_table,char_batch,char_embedd_table)
        
        _, step, summaries, loss,logits,transition_params = session.run(
            [train_op, global_step, train_summary_op, BiLSTM.loss,BiLSTM.logits,BiLSTM.transition_params],
            feed_dict)

        time_str = datetime.datetime.now().isoformat()
        print("{}: step {}, loss {:g}".format(time_str, step, loss))
        train_summary_writer.add_summary(summaries, step)

    # Generate batches
    batches = utils.batch_iter(
        list(zip(word_index_sentences_train_pad, label_index_sentences_train_pad ,train_seq_length,char_index_train_pad)), FLAGS.batch_size, FLAGS.num_epochs)
    
    # Training loop. For each batch...
    for batch in batches:
        x_batch, y_batch,act_seq_lengths,char_batch = zip(*batch)
        train_step(sess,BiLSTM,FLAGS.PadZeroBegin,max_length,x_batch, y_batch,act_seq_lengths,FLAGS.dropout_keep_prob,
            embedd_table,char_batch,char_embedd_table)
        current_step = tf.train.global_step(sess, global_step)
        if current_step % FLAGS.evaluate_every == 0:
            print("\nEvaluation:")
            new_accuracy,accuracy_low_classes=dev_step(sess,BiLSTM,FLAGS.PadZeroBegin,max_length,word_index_sentences_dev_pad,
                                            label_index_sentences_dev_pad ,dev_seq_length,FLAGS.dropout_keep_prob,
                                            embedd_table,current_step,char_index_dev_pad,char_embedd_table, writer=dev_summary_writer)
            print("")
            if (accuracy_low_classes > best_accuracy):
                
                path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                best_accuracy = accuracy_low_classes
                best_step = current_step
                best_overall_accuracy = new_accuracy
                print("Saved model checkpoint to {}\n".format(path))
                #run test data
                new_accuracy_test,accuracy_low_classes_test = af.test_step(logger= logger,session=sess,BiLSTM=BiLSTM,PadZeroBegin=FLAGS.PadZeroBegin,max_length=max_length,
                    test_path=test_path,dropout_keep_prob=FLAGS.dropout_keep_prob,step=current_step,out_dir=out_dir,char_alphabet=char_alphabet,
                    label_alphabet=label_alphabet,word_alphabet=word_alphabet,word_column=word_column,label_column=label_column,
                    char_embedd_dim=FLAGS.char_embedd_dim,max_char_per_word=max_char_per_word)
                if (accuracy_low_classes_test > best_accuracy_test):
                    best_accuracy_test = accuracy_low_classes_test
                    best_step_test = current_step
                    best_overall_accuracy_test = new_accuracy_test

    print("DEV: best_accuracy on NER : %f best_step: %d best_overall_accuracy: %d" %(best_accuracy,best_step,best_overall_accuracy))
    print("TEST : best_accuracy on NER : %f best_step: %d best_overall_accuracy: %d" %(best_accuracy_test,best_step_test,best_overall_accuracy_test))



