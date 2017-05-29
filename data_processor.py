import utils as utils
from alphabet import Alphabet
import numpy as np
import pickle
import os
MAX_LENGTH = 120
MAX_CHAR_PER_WORD = 45
root_symbol = "##ROOT##"
root_label = "<ROOT>"
word_end = "##WE##"
logger = utils.get_logger("LoadData")

def read_conll_sequence_labeling(path, word_alphabet, label_alphabet, word_column=1, label_column=3,out_dir=None):
    """
    read data from file in conll format
    :param path: file path
    :param word_column: the column index of word (start from 0)
    :param label_column: the column of label (start from 0)
    :param word_alphabet: alphabet of words
    :param label_alphabet: alphabet -f labels
    :return: sentences of words and labels, sentences of indexes of words and labels.
    """
    word_sentences = []
    label_sentences = []

    word_index_sentences = []
    label_index_sentences = []
    if(out_dir !=None):
        vocab = set()
        #print(out_dir = os.path.abspath(os.path.join(os.path.curdir, "vocab", timestamp)))
        vocab_save_path = os.path.join(out_dir, "vocab.pkl")
    words = []
    labels = []

    word_ids = []
    label_ids = []

    num_tokens = 0
    with open(path) as file:
        for line in file:
            #line.decode('utf-8')
            if line.strip() == "":#this means we have the entire sentence
                if 0 < len(words) <= MAX_LENGTH:
                    word_sentences.append(words[:])
                    label_sentences.append(labels[:])

                    word_index_sentences.append(word_ids[:])
                    label_index_sentences.append(label_ids[:])

                    num_tokens += len(words)
                else:
                    if len(words) != 0:
                        logger.info("ignore sentence with length %d" % (len(words)))

                words = []
                labels = []

                word_ids = []
                label_ids = []
            else:
                tokens = line.strip().split()
                word = tokens[word_column]
                label = tokens[label_column]

                words.append(word)
                if(out_dir !=None):
                    vocab.add(word)
                labels.append(label)
                word_id = word_alphabet.get_index(word)
                label_id = label_alphabet.get_index(label)

                word_ids.append(word_id)
                label_ids.append(label_id)
    #this is for the last sentence            
    if 0 < len(words) <= MAX_LENGTH:
        word_sentences.append(words[:])
        label_sentences.append(labels[:])

        word_index_sentences.append(word_ids[:])
        label_index_sentences.append(label_ids[:])

        num_tokens += len(words)
    else:
        if len(words) != 0:
            logger.info("ignore sentence with length %d" % (len(words)))

    if(out_dir !=None):
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        with open(vocab_save_path, 'wb') as handle:
            pickle.dump(vocab, handle)  
        logger.info("vocab written to %s" % (vocab_save_path))     
    logger.info("#sentences: %d, #tokens: %d" % (len(word_sentences), num_tokens))
    
    return word_sentences, label_sentences, word_index_sentences, label_index_sentences

def build_embedd_table(word_alphabet, embedd_dict, embedd_dim, caseless):
    scale = np.sqrt(3.0 / embedd_dim)
    #TODO:should we build an embedding table with words in our training/dev/test plus glove .
    # the extra words in glove will not be trained but can help with UNK 
    embedd_table = np.empty([word_alphabet.size(), embedd_dim], dtype=np.float64)
    embedd_table[word_alphabet.default_index, :] = np.random.uniform(-scale, scale, [1, embedd_dim])
    for word, index in word_alphabet.items():
        ww = word.lower() if caseless else word
        embedd = embedd_dict[ww] if ww in embedd_dict else np.random.uniform(-scale, scale, [1, embedd_dim])
        embedd_table[index, :] = embedd
    return embedd_table

def construct_padded_char(index_sentences,char_alphabet,max_sent_length,max_char_per_word):
    C = np.empty([len(index_sentences), max_sent_length, max_char_per_word], dtype=np.int32)
    # this is to mark space at the end of the words
    word_end_id = char_alphabet.get_index(word_end)

    for i in range(len(index_sentences)):
        words = index_sentences[i]
        sent_length = len(words)
        for j in range(min(sent_length,max_sent_length)):
            chars = words[j]
            char_length = len(chars)
            for k in range(min (char_length,max_char_per_word)):
                cid = chars[k]
                C[i, j, k] = cid
            # fill index of word end after the end of word
            C[i, j, char_length:] = word_end_id
        # Zero out C after the end of the sentence
        C[i, sent_length:, :] = 0
    return C


def build_char_embedd_table(char_alphabet,char_embedd_dim=30):
    scale = np.sqrt(3.0 / char_embedd_dim)
    char_embedd_table = np.random.uniform(-scale, scale, [char_alphabet.size(), char_embedd_dim]).astype(
        np.float64)
    return char_embedd_table


def generate_character_data(sentences_list,char_alphabet, setType="Train", char_embedd_dim=30):
    """
    generate data for charaters
    :param sentences_train:
    :param sentences_train:
    :param max_sent_length: zero for trainset:
    :return: char_index_set_pad,max_char_per_word, char_embedd_table,char_alphabet
    """

    def get_character_indexes(sentences):
        index_sentences = []
        max_length = 0
        for words in sentences:
            index_words = []
            for word in words:
                index_chars = []
                if len(word) > max_length:
                    max_length = len(word)

                for char in word[:MAX_CHAR_PER_WORD ]:
                    char_id = char_alphabet.get_index(char)
                    index_chars.append(char_id)

                index_words.append(index_chars)
            index_sentences.append(index_words)
        return index_sentences, max_length
    
    char_alphabet.get_index(word_end)

    index_sentences, max_char_per_word = get_character_indexes(sentences_list)
    max_char_per_word = min(MAX_CHAR_PER_WORD, max_char_per_word)
    logger.info("Maximum character length after %s set is %d" %(setType ,max_char_per_word))
    return index_sentences,max_char_per_word