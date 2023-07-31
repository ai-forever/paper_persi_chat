import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

import numpy as np
import os, random, time, pickle
from Model import Model
from utils import load_data, build_vocab, preview_data, get_batches

if 'CUDA_VISIBLE_DEVICES' not in os.environ: 
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

FLAGS = tf.flags.FLAGS
flags = tf.flags


flags.DEFINE_boolean('train', False, 'train model')
flags.DEFINE_string('train_file', 'data/STAC/train.json', 'test file path')
flags.DEFINE_string('test_file', 'data/STAC/test.json', 'test file path')
flags.DEFINE_integer('display_interval', 500, 'step interval to display information')
flags.DEFINE_boolean('show_predictions', False, 'show predictions in the test stage')
flags.DEFINE_string('word_vector', 'glove/glove.6B.100d.txt', 'word vector')
flags.DEFINE_string('prefix', 'dev', 'prefix for storing model and log')
flags.DEFINE_integer('vocab_size', 1000, 'vocabulary size')
flags.DEFINE_integer('max_edu_dist', 20, 'maximum distance between two related edus') 
flags.DEFINE_integer('dim_embed_word', 100, 'dimension of word embedding')
flags.DEFINE_integer('dim_embed_relation', 100, 'dimension of relation embedding')
flags.DEFINE_integer('dim_feature_bi', 4, 'dimension of binary features')
flags.DEFINE_boolean('use_structured', True, 'use structured encoder')
flags.DEFINE_boolean('use_speaker_attn', True, 'use speaker highlighting mechanism')
flags.DEFINE_boolean('use_shared_encoders', False, 'use shared encoders')
flags.DEFINE_boolean('use_random_structured', False, 'use random structured repr.')
flags.DEFINE_integer('num_epochs', 50, 'number of epochs')
flags.DEFINE_integer('num_units', 256, 'number of hidden units')
flags.DEFINE_integer('num_layers', 1, 'number of RNN layers in encoders')
flags.DEFINE_integer('num_relations', 16, 'number of relation types')
flags.DEFINE_integer('batch_size', 4, 'batch size')
flags.DEFINE_float('keep_prob', 0.5, 'probability to keep units in dropout')
flags.DEFINE_float('learning_rate', 0.1, 'learning rate')
flags.DEFINE_float('learning_rate_decay', 0.98, 'learning rate decay factor')
flags.DEFINE_string('f', '', 'kernel')


class DiscoDialParser:
    def __init__(self, train_file, prefix, word_vector):
        FLAGS.word_vector = word_vector
        self.map_relations = {}
        data_train = load_data(train_file, self.map_relations, toprint=False)
        vocab, embed = build_vocab(data_train)
        model_dir, log_dir = prefix + '_model', prefix + '_log'
        self.len_output_feed = 6
        
        if tf.__version__ < "2.0.0":
            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True
            self.sess = tf.Session(config=config)
        
        else:
            gpu_devices = tf.config.experimental.list_physical_devices('GPU')
            for device in gpu_devices:
                tf.config.experimental.set_memory_growth(device, True)
            self.sess = tf.compat.v1.Session()
        
        
        with self.sess.as_default():
            self.model = Model(self.sess, FLAGS, embed, data_train)
            saver = tf.train.Saver(write_version=tf.train.SaverDef.V2, max_to_keep=None, pad_step_number=True)
            saver.restore(self.sess, tf.train.latest_checkpoint(model_dir))
    
    def parse(self, dialogues):
        '''
        dialogue format list of dicts: [{'speaker': '...', 'text': '...'}]
        '''
        if type(dialogues[0]) == dict:
            dialogues = [dialogues]
        test = [{'edus': dial, 'relations': []} for dial in dialogues]
        data_test = load_data(test, self.map_relations, toprint=False)
        test_batches = get_batches(data_test, 1, sort=False)
        
        map_relations_inv = {}
        for item in self.map_relations:
            map_relations_inv[self.map_relations[item]] = item
    
        s = np.zeros(self.len_output_feed)
        random.seed(0)
        idx = 0
        result = []
        for k, batch in enumerate(test_batches):
            if len(batch[0]['edus']) == 1: 
                continue    
            ops = self.model.step(batch)
            for i in range(self.len_output_feed):
                s[i] += ops[i]
            if FLAGS.show_predictions:
                idx = preview_data(batch, ops[-1], self.map_relations, vocab, idx)
            
            for i in range(len(batch)):
                cur_res = batch[i].copy()
                rels = []
                for relation in ops[-1][i]:
                    rels.append([relation[0], relation[1], map_relations_inv[relation[2]]])
                cur_res['relations_pred'] = rels
                result.append(cur_res)
        
        result = [el['relations_pred'] for el in result]
        if len(result) == 1:
            result = result[0]
        return result
    