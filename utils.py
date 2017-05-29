from gensim.models.word2vec import Word2Vec
import logging
import sys
import gzip
import numpy as np

def get_logger(name, level=logging.INFO, handler=sys.stdout,
               formatter='%(asctime)s - %(name)s - %(levelname)s - %(message)s'):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(formatter)
    stream_handler = logging.StreamHandler(handler)
    stream_handler.setLevel(level)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    return logger

def print_FLAGS(FLAGS,logger):
    Flags_Dict = {}
    logger.info("\nParameters:")
    for attr, value in sorted(FLAGS.__flags.items()):
      logger.info("{} = {}".format(attr, value))
      Flags_Dict[attr] = value
    logger.info("\n")
    return Flags_Dict


def load_word_embedding_dict(embedding,embedding_path,logger):
    """
    load word embeddings from file
    :param embedding:
    :param embedding_path:
    :param logger:
    :return: embedding dict, embedding dimention, caseless
    """
    if embedding == 'word2vec':
        # loading word2vec
        logger.info("Loading word2vec ...")
        word2vec = Word2Vec.load_word2vec_format(embedding_path, binary=True)
        embedd_dim = word2vec.vector_size
        return word2vec, embedd_dim, False
    elif embedding == 'glove':
        # loading GloVe
        logger.info("Loading GloVe ...")
        embedd_dim = -1
        embedd_dict = dict()
        with gzip.open(embedding_path, 'r') as file:
            for line in file:
                line = line.strip()
                if len(line) == 0:
                    continue

                tokens = line.split()
                if embedd_dim < 0:
                    embedd_dim = len(tokens) - 1 #BECAUSE THE ZEROTH INDEX IS OCCUPIED BY THE WORD
                else:
                    assert (embedd_dim + 1 == len(tokens))
                embedd = np.empty([1, embedd_dim], dtype=np.float64)
                embedd[:] = tokens[1:]
                embedd_dict[tokens[0]] = embedd
        return embedd_dict, embedd_dim, True
    else:
        raise ValueError("embedding should choose from [word2vec, glove]")

def get_max_length(word_sentences):
    max_len = 0
    for sentence in word_sentences:
        length = len(sentence)
        if length > max_len:
            max_len = length
    return max_len

def batch_iter(data, batch_size, num_epochs, shuffle=True):
    """
    Generates a batch iterator for a dataset.
    """
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int((len(data)-1)/batch_size) + 1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]

#this function will pad 0 at the beginning of the sentence. if you add beg = false it will add to the end
def padSequence(dataset,max_length,beginZero=True):
    dataset_p = []
    actual_sequence_length =[]
    #added np.atleast_2d here
    for x in dataset:
        row_length = len(x)
        actual_sequence_length.append(row_length)
        if(row_length <=max_length):
            if(beginZero):
                dataset_p.append(np.pad(x,pad_width=(max_length-len(x),0),mode='constant',constant_values=0))
            else:
                dataset_p.append(np.pad(x,pad_width=(0,max_length-len(x)),mode='constant',constant_values=0))
        else:
            dataset_p.append(x[0:max_length])
    return np.array(dataset_p),actual_sequence_length


