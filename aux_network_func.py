import tensorflow as tf
import numpy as np
import utils as utils
import os
import time
import datetime
import data_processor as dp
import dill

def create_feed_Dict(BiLSTM,PadZeroBegin,max_length,x_batch,y_batch,act_seq_lengths,dropout_keep_prob,embedd_table,char_batch,char_embedd_table):
        if PadZeroBegin:
            cur_batch_size = len(x_batch)
            sequence_length_batch= np.full((cur_batch_size), max_length, dtype=int)
            feed_dict = {
              BiLSTM.input_x: x_batch,
              BiLSTM.input_y: y_batch,
              BiLSTM.dropout_keep_prob: dropout_keep_prob,
              BiLSTM.word_embedding_placeholder: embedd_table,
              BiLSTM.sequence_lengths : sequence_length_batch, #NOTES:sadly giving hte actual seq length gives None all the time when sequence is padded in begining
              #BiLSTM.sequence_lengths : seq_length
              BiLSTM.input_x_char : char_batch,
              BiLSTM.char_embedding_placeholder : char_embedd_table

            }
        else:
            feed_dict = {
              BiLSTM.input_x: x_batch,
              BiLSTM.input_y: y_batch,
              BiLSTM.dropout_keep_prob: dropout_keep_prob,
              BiLSTM.word_embedding_placeholder: embedd_table,
              #BiLSTM.sequence_lengths : sequence_length_batch, #NOTES:sadly giving hte actual seq length gives None all the time when sequence is padded in begining
              BiLSTM.sequence_lengths : act_seq_lengths,
              BiLSTM.input_x_char : char_batch,
              BiLSTM.char_embedding_placeholder : char_embedd_table


            }
        return feed_dict

def create_feed_Dict_Test(BiLSTM,PadZeroBegin,max_length,x_batch, y_batch,act_seq_lengths, dropout_keep_prob,char_batch):
        if PadZeroBegin:
            cur_batch_size = len(x_batch)
            sequence_length_batch= np.full((cur_batch_size), max_length, dtype=int)
            feed_dict = {
              BiLSTM.input_x: x_batch,
              BiLSTM.input_y: y_batch,
              BiLSTM.dropout_keep_prob: dropout_keep_prob,
              BiLSTM.sequence_lengths : sequence_length_batch, #NOTES:sadly giving hte actual seq length gives None all the time when sequence is padded in begining
              #BiLSTM.sequence_lengths : seq_length
              BiLSTM.input_x_char : char_batch,

            }
        else:
            feed_dict = {
              BiLSTM.input_x: x_batch,
              BiLSTM.input_y: y_batch,
              BiLSTM.dropout_keep_prob: dropout_keep_prob,
              #BiLSTM.sequence_lengths : sequence_length_batch, #NOTES:sadly giving hte actual seq length gives None all the time when sequence is padded in begining
              BiLSTM.sequence_lengths : act_seq_lengths,
              BiLSTM.input_x_char : char_batch,


            }
        return feed_dict


def predictAccuracyAndWrite(logits,transition_params,seq_length,y_batch,step,x_batch,word_alphabet,label_alphabet,prefix_filename="Dev",beginZero=True):
        correct_labels = 0
        total_labels = 0
        correct_labels_low_classes = 0
        total_labels_low_classes = 0
        fname = prefix_filename + "_Predictions_" +str(step)+".txt"
        
        with open(fname, 'w') as outfile:
            outfile.write("word\ty_label\tpred_label\n")
            for tf_unary_scores_, y_, sequence_length_, x_ in zip(logits, y_batch,seq_length,x_batch):
                # Remove padding from the scores and tag sequence.
                tf_unary_scores_ = tf_unary_scores_[-sequence_length_:] if beginZero else tf_unary_scores_[:sequence_length_] 
                #for writing to file
                y_ = y_[-sequence_length_:] if beginZero else y_[:sequence_length_]
                x_ = x_[-sequence_length_:] if beginZero else x_[:sequence_length_]
                # Compute the highest scoring sequence.
                viterbi_sequence, viterbi_score = tf.contrib.crf.viterbi_decode(
                      tf_unary_scores_, transition_params)
                for xi,yi,vi in zip(x_,y_,viterbi_sequence):
                    x_word = word_alphabet.get_instance(xi)
                    y_label = label_alphabet.get_instance(yi)
                    pred_label = label_alphabet.get_instance(vi)
                    outfile.write(str(x_word) + "\t"+str(y_label)+"\t"+str(pred_label)+"\n")
                    if(y_label != "O"):  
                        total_labels_low_classes = total_labels_low_classes + 1
                        if (y_label == pred_label):
                            correct_labels_low_classes = correct_labels_low_classes +1 
                outfile.write("\n")    
                # Evaluate word-level accuracy.
                correct_labels += np.sum(np.equal(viterbi_sequence, y_))
                total_labels += sequence_length_  
            accuracy = 100.0 * correct_labels / float(total_labels)
            accuracy_low_classes = 100.0 * correct_labels_low_classes / float(total_labels_low_classes)
            outfile.write("accuracy: " + str(accuracy))
            outfile.write("\naccuracy for classes except other : " + str(accuracy_low_classes))
            outfile.write("\ntotal other classes : {}, correctly predicted : {}  ".format(total_labels_low_classes,correct_labels_low_classes ))
            outfile.write("\ntotal : {}, correctly predicted : {}  ".format(total_labels,correct_labels ))
        return accuracy,accuracy_low_classes

def test_step(logger,session,BiLSTM,PadZeroBegin,max_length,test_path,
    dropout_keep_prob,step,out_dir,char_alphabet,label_alphabet,word_alphabet,
     word_column, label_column,char_embedd_dim,max_char_per_word):
        # read test data
    logger.info("Reading data from test set...")
    word_sentences_test, _, word_index_sentences_test, label_index_sentences_test = dp.read_conll_sequence_labeling(
        test_path, word_alphabet, label_alphabet, word_column, label_column)
    logger.info("Padding test text and lables ...")
    word_index_sentences_test_pad,test_seq_length = utils.padSequence(word_index_sentences_test,max_length, beginZero=PadZeroBegin)
    label_index_sentences_test_pad,_= utils.padSequence(label_index_sentences_test,max_length, beginZero=PadZeroBegin)
    logger.info("Creating character set FROM test set ...")
    char_index_test,_= dp.generate_character_data(word_sentences_test, 
                                char_alphabet=char_alphabet, setType="Test")

    logger.info("Padding Test set ...")
    char_index_test_pad = dp.construct_padded_char(char_index_test, char_alphabet, max_sent_length=max_length,max_char_per_word=max_char_per_word)
    
    # test summaries
    #test_summary_op = tf.summary.merge([loss_summary])
    #test_summary_dir = os.path.join(out_dir, "summaries", "test")
    #test_summary_writer = tf.summary.FileWriter(test_summary_dir, sess.graph)

    feed_dict=create_feed_Dict_Test(BiLSTM,PadZeroBegin=PadZeroBegin,max_length=max_length,
        x_batch=word_index_sentences_test_pad, y_batch=label_index_sentences_test_pad,
         act_seq_lengths= test_seq_length, dropout_keep_prob=dropout_keep_prob,
         char_batch=char_index_test_pad)
    '''#tf.Print(feed_dict,feed_dict)
    logits, transition_params = session.run([BiLSTM.logits, BiLSTM.transition_params],feed_dict)
    #logits is a list of numpy.ndarray
    #transition_params : ndarray'''
    
    logits, transition_params,embedded_char,embedded_words,char_pool_flat,input_x_test = session.run([BiLSTM.logits, BiLSTM.transition_params,
        BiLSTM.W_char,BiLSTM.W_word,BiLSTM.char_pool_flat,BiLSTM.input_x],feed_dict)
    
    accuracy,accuracy_low_classes = predictAccuracyAndWrite(logits,transition_params,test_seq_length,
        label_index_sentences_test_pad,step,word_index_sentences_test_pad,word_alphabet,label_alphabet,prefix_filename="test",beginZero=PadZeroBegin)

    #test_summary_writer.add_summary(summaries, step)
    print("step {},  accuracy on test set {:g}, accuracy for classes except Others: {:g}".format(step,accuracy,accuracy_low_classes))

    checkpoint_dir_test = os.path.abspath(os.path.join(out_dir, "checkpoints_test"))

    if not os.path.exists(checkpoint_dir_test):
        os.makedirs(checkpoint_dir_test)
    fname_data = "input_x_test_"+str(step)+".pkl"
    fname_conv_out = "char_pool_flat_"+str(step)+".pkl" 
    fname_seqLength = "act_seq_len_"+str(step)+".pkl" 
    fname_embedded_char = "embedded_char_"+str(step)+".pkl" 
    fname_embedded_words = "embedded_words_"+str(step)+".pkl" 
    dill.dump(input_x_test,open(os.path.join(checkpoint_dir_test, fname_data),'wb'))
    dill.dump(char_pool_flat,open(os.path.join(checkpoint_dir_test, fname_conv_out),'wb'))
    dill.dump(test_seq_length,open(os.path.join(checkpoint_dir_test, fname_seqLength),'wb'))
    dill.dump(embedded_char,open(os.path.join(checkpoint_dir_test, fname_embedded_char),'wb'))
    dill.dump(embedded_words,open(os.path.join(checkpoint_dir_test, fname_embedded_words),'wb'))
    print("Saved test data checkpoint to {}\n".format(checkpoint_dir_test))
    return accuracy,accuracy_low_classes



def create_feed_Dict_Eval(graph,PadZeroBegin,max_length,x_batch,act_seq_lengths,dropout_keep_prob,
    char_batch):
    
    if PadZeroBegin:
        cur_batch_size = len(x_batch)
        sequence_length_batch= np.full((cur_batch_size), max_length, dtype=int)
        feed_dict = {
          graph.get_tensor_by_name('input_x:0'): x_batch, 
          graph.get_tensor_by_name('dropout_keep_prob:0'): dropout_keep_prob,
          graph.get_tensor_by_name('sequence_lengths:0') : sequence_length_batch, #NOTES:sadly giving hte actual seq length gives None all the time when sequence is padded in begining
          graph.get_tensor_by_name('input_x_char:0') : char_batch,
        }
    else:
        feed_dict = {
          graph.get_tensor_by_name('input_x:0'): x_batch,
          graph.get_tensor_by_name('dropout_keep_prob:0'): dropout_keep_prob,
          graph.get_tensor_by_name('sequence_lengths:0') : act_seq_lengths,
          graph.get_tensor_by_name('input_x_char:0') : char_batch,

        }
        return feed_dict

# This function is just to understand the network for debugging purposes
def viterbi_decode(score, transition_params, targetWordIndex):
  """Decode the highest scoring sequence of tags outside of TensorFlow.
  This should only be used at test time.
  Args:
    score: A [seq_len, num_tags] matrix of unary potentials.
    transition_params: A [num_tags, num_tags] matrix of binary potentials.
  Returns:
    viterbi: A [seq_len] list of integers containing the highest scoring tag
        indicies.
    viterbi_score: A float containing the score for the Viterbi sequence.
  """
  trellis = np.zeros_like(score)
  backpointers = np.zeros_like(score, dtype=np.int32)
  trellis[0] = score[0]

  v_target = np.zeros_like(transition_params)
  for t in range(1, score.shape[0]):
    v = np.expand_dims(trellis[t - 1], 1) + transition_params
    if(t==targetWordIndex):
      v_target = v
    trellis[t] = score[t] + np.max(v, 0)
    backpointers[t] = np.argmax(v, 0)
  

  viterbi = [np.argmax(trellis[-1])]
  for bp in reversed(backpointers[1:]):
    viterbi.append(bp[viterbi[-1]])
  viterbi.reverse()
  if(targetWordIndex == 0):
    total = float(np.sum([i if i > 0 else 0 for i in score[0]]))
    prob = [i/total if i > 0 else 0  for i in score[0]]
  else:
    total = float(np.sum([i if i > 0 else 0 for i in v_target[viterbi[targetWordIndex]]]))
    prob = [i/total if i > 0 else 0  for i in v_target[viterbi[targetWordIndex]]]
  dill.dump(prob,open("prob.dill",'wb'))
  '''dill.dump(trellis,open("trellis.dill",'wb'))
  dill.dump(score,open("score.dill",'wb'))
  dill.dump(transition_params,open("transition_params.dill",'wb'))'''
  viterbi_score = np.max(trellis[-1])
  return viterbi, viterbi_score,prob

def debug(logits,transition_params,seq_length,x_batch,word_alphabet,label_alphabet, targetWordIndexArray,prefix_filename="Dev", beginZero=True):

  for tf_unary_scores_,sequence_length_, x_,targetWordIndex in zip(logits, seq_length,x_batch,targetWordIndexArray):
      # Remove padding from the scores and tag sequence.
      tf_unary_scores_ = tf_unary_scores_[-sequence_length_:] if beginZero else tf_unary_scores_[:sequence_length_] 
      x_ = x_[-sequence_length_:] if beginZero else x_[:sequence_length_]

      # Compute the highest scoring sequence.
      viterbi_sequence, viterbi_score,prob = viterbi_decode(
            tf_unary_scores_, transition_params, targetWordIndex)

      return
def test_step_eval(logger,session,PadZeroBegin,max_length,test_path,
    dropout_keep_prob,step,out_dir,char_alphabet,label_alphabet,word_alphabet,
     word_column, label_column,char_embedd_dim,max_char_per_word):
        # read test data
    graph = tf.get_default_graph()
    logger.info("Reading data from test set...")
    word_sentences_test, _, word_index_sentences_test, label_index_sentences_test = dp.read_conll_sequence_labeling(
        test_path, word_alphabet, label_alphabet, word_column, label_column)
    logger.info("Padding test text and lables ...")
    word_index_sentences_test_pad,test_seq_length = utils.padSequence(word_index_sentences_test,max_length, beginZero=PadZeroBegin)
    label_index_sentences_test_pad,_= utils.padSequence(label_index_sentences_test,max_length, beginZero=PadZeroBegin)
    logger.info("Creating character set FROM test set ...")
    char_index_test,_= dp.generate_character_data(word_sentences_test, 
                                char_alphabet=char_alphabet, setType="Test")

    logger.info("Padding Test set ...")
    char_index_test_pad = dp.construct_padded_char(char_index_test, char_alphabet, max_sent_length=max_length,max_char_per_word=max_char_per_word)
    print(type(char_index_test_pad))
    print(type(word_index_sentences_test_pad))
    
    feed_dict=create_feed_Dict_Eval(graph,PadZeroBegin=PadZeroBegin,max_length=max_length,
        x_batch=word_index_sentences_test_pad,
         act_seq_lengths= test_seq_length, dropout_keep_prob=dropout_keep_prob,
         char_batch=char_index_test_pad)
    #tf.Print(feed_dict,feed_dict)
    logit_op = graph.get_tensor_by_name('output/logits:0')
    transition_params_op = graph.get_tensor_by_name('transitions:0')
    logits,transition_params = session.run([logit_op, transition_params_op],feed_dict)
    print(logits.shape)
    targetWordIndexArray = np.asarray([0])
    debug(logits=logits,transition_params=transition_params,
      seq_length=test_seq_length,x_batch=word_index_sentences_test_pad,word_alphabet=word_alphabet,label_alphabet=label_alphabet, 
      targetWordIndexArray=targetWordIndexArray,prefix_filename="test",beginZero=PadZeroBegin)
    return 0,0
   
    '''accuracy,accuracy_low_classes = predictAccuracyAndWrite(logits,transition_params,test_seq_length,step,word_index_sentences_test_pad,word_alphabet,label_alphabet,prefix_filename="test",beginZero=PadZeroBegin)

    #test_summary_writer.add_summary(summaries, step)
    print("step {},  accuracy on test set {:g}, accuracy for classes except Others: {:g}".format(step,accuracy,accuracy_low_classes))

    return accuracy,accuracy_low_classes'''
def viterbi_decode(logits,transition_params,seq_length,x_batch,word_alphabet,label_alphabet, prefix_filename="Test", beginZero=True):
  fname = prefix_filename + "_Predictions.txt"
  with open(fname, 'w') as outfile:
    outfile.write("word\ty_label\tpred_label\n")
    for tf_unary_scores_,sequence_length_, x_,targetWordIndex in zip(logits, seq_length,x_batch,targetWordIndexArray):
        # Remove padding from the scores and tag sequence.
        tf_unary_scores_ = tf_unary_scores_[-sequence_length_:] if beginZero else tf_unary_scores_[:sequence_length_] 
        x_ = x_[-sequence_length_:] if beginZero else x_[:sequence_length_]
        # Compute the highest scoring sequence.
        viterbi_sequence, viterbi_score = tf.contrib.crf.viterbi_decode(
              tf_unary_scores_, transition_params)
        for xi,yi,vi in zip(x_,viterbi_sequence):
            x_word = word_alphabet.get_instance(xi)
            pred_label = label_alphabet.get_instance(vi)
            outfile.write(str(x_word) + "\t"+str(pred_label)+"\n")
        outfile.write("\n")   

  return

def test_step_report(logger,session,PadZeroBegin,max_length,test_path,
    dropout_keep_prob,step,out_dir,char_alphabet,label_alphabet,word_alphabet,
     word_column, label_column,char_embedd_dim,max_char_per_word):
        # read test data
    graph = tf.get_default_graph()
    logger.info("Reading data from test set...")
    word_sentences_test, _, word_index_sentences_test, label_index_sentences_test = dp.read_conll_sequence_labeling(
        test_path, word_alphabet, label_alphabet, word_column, label_column)
    logger.info("Padding test text and lables ...")
    word_index_sentences_test_pad,test_seq_length = utils.padSequence(word_index_sentences_test,max_length, beginZero=PadZeroBegin)
    label_index_sentences_test_pad,_= utils.padSequence(label_index_sentences_test,max_length, beginZero=PadZeroBegin)
    logger.info("Creating character set FROM test set ...")
    char_index_test,_= dp.generate_character_data(word_sentences_test, 
                                char_alphabet=char_alphabet, setType="Test")

    logger.info("Padding Test set ...")
    char_index_test_pad = dp.construct_padded_char(char_index_test, char_alphabet, max_sent_length=max_length,max_char_per_word=max_char_per_word)
    print(type(char_index_test_pad))
    print(type(word_index_sentences_test_pad))
    
    feed_dict=create_feed_Dict_Eval(graph,PadZeroBegin=PadZeroBegin,max_length=max_length,
        x_batch=word_index_sentences_test_pad,
         act_seq_lengths= test_seq_length, dropout_keep_prob=dropout_keep_prob,
         char_batch=char_index_test_pad)
    #tf.Print(feed_dict,feed_dict)
    logit_op = graph.get_tensor_by_name('output/logits:0')
    transition_params_op = graph.get_tensor_by_name('transitions:0')
    logits,transition_params = session.run([logit_op, transition_params_op],feed_dict)
    viterbi_decode(logits=logits,transition_params=transition_params,
      seq_length=test_seq_length,x_batch=word_index_sentences_test_pad,word_alphabet=word_alphabet,label_alphabet=label_alphabet, 
      prefix_filename="test",beginZero=PadZeroBegin)
    return