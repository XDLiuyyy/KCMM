import requests
import pandas as pd
import torch
import pickle
from transformers import BertTokenizer, RobertaTokenizer
from collections import defaultdict
from bidict import bidict
from model.text_to_uri import standardized_uri
import numpy as np
import re
import string
import urllib.parse
import time
import os
from nltk.stem import WordNetLemmatizer


# %%
class GraphUtils:
    def __init__(self):
        self.mp_all = defaultdict(set)
        self.words_to_id = bidict()
        self.words_encode_idx = 0
        self.conceptnet_numberbatch_en = dict()
        self.tokenizer = BertTokenizer.from_pretrained('bert-large-uncased')
        self.args = {
            'n_gram': 3,
            'mp_pickle_path': 'conceptnet5/res_all.pickle',
            'conceptnet_numberbatch_en_path': 'conceptnet5/numberbatch-en.txt',
            'reduce_noise_args': {

                'relation_white_list': ['/r/ReceivesAction', '/r/RelatedTo', '/r/UsedFor'],

                'relation_black_list': ['/r/ExternalURL', '/r/Synonym', '/r/Antonym',
                                        '/r/DistinctFrom', '/r/dbpedia/genre', '/r/dbpedia/influencedBy'],

                'stop_words': ['a', 'able', 'about', 'across', 'after', 'all', 'almost', 'also', 'am', 'among', 'an',
                               'and', 'any', 'are', 'as', 'at', 'be', 'because', 'been', 'but', 'by', 'can', 'cannot',
                               'could', 'dear', 'did', 'do', 'does', 'either', 'else', 'ever', 'every', 'for', 'from',
                               'get', 'got', 'had', 'has', 'have', 'he', 'her', 'hers', 'him', 'his', 'how', 'however',
                               'i', 'if', 'in', 'into', 'is', 'it', 'its', 'just', 'least', 'let', 'like', 'likely',
                               'may', 'me', 'might', 'most', 'must', 'my', 'neither', 'no', 'nor', 'not', 'of', 'off',
                               'often', 'on', 'only', 'or', 'other', 'our', 'own', 'rather', 'said', 'say', 'says',
                               'she', 'should', 'since', 'so', 'some', 'than', 'that', 'the', 'their', 'them', 'then',
                               'there', 'these', 'they', 'this', 'tis', 'to', 'too', 'twas', 'us', 'wants', 'was', 'we',
                               'were', 'what', 'when', 'where', 'which', 'while', 'who', 'whom', 'why', 'will', 'with',
                               'would', 'yet', 'you', 'your'],
                'weight_limit': 1.0,
                'edge_count_limit': 100,
            }
        }

    def load_mp_all_by_pickle(self, fpath):
        graph_zip = None
        with open(fpath, 'rb') as f:
            graph_zip = pickle.load(f)
        self.mp_all, = graph_zip
        return graph_zip

    def reduce_graph_noise(self, is_overwrite=True):
        relation_white_list = self.args['reduce_noise_args']['relation_white_list']
        relation_black_list = self.args['reduce_noise_args']['relation_black_list']
        stop_words = self.args['reduce_noise_args']['stop_words']
        weight_limit = self.args['reduce_noise_args']['weight_limit']
        edge_count_limit = self.args['reduce_noise_args']['edge_count_limit']
        is_black_list = True
        if len(relation_white_list) != 0:

            is_black_list = False

        new_mp = defaultdict(set)
        for key, values in self.mp_all.items():
            st_words = key
            if st_words in stop_words:
                continue
            to_values = sorted(list(values), key=lambda x: x[2], reverse=True)
            edge_count = 0
            for value in to_values:
                to_words = value[0]
                to_relation = value[1]
                to_weight = value[2]
                if to_words in stop_words:
                    continue
                if to_weight < weight_limit:
                    continue
                if is_black_list:
                    if to_relation in relation_black_list:
                        continue
                else:
                    if to_relation not in relation_white_list:
                        continue
                new_mp[st_words].add((to_words, to_relation, to_weight))
                edge_count += 1
                if edge_count >= edge_count_limit:
                    break

        if is_overwrite:
            self.mp_all = new_mp
        return new_mp

    def merge_graph_by_downgrade(self, is_overwrite=True):
        new_mp = defaultdict(set)
        refine_sent = lambda s: re.match('/c/en/([^/]+)', s).group(1)
        for key, values in self.mp_all.items():
            st_words = refine_sent(key)
            for value in values:
                to_words = refine_sent(value[0])
                to_relation = value[1]
                to_weight = value[2]
                new_mp[st_words].add((to_words, to_relation, to_weight))
        if is_overwrite:
            self.mp_all = new_mp
        return new_mp

    def init(self, is_load_necessary_data=True):
        self.__init__()
        if is_load_necessary_data:
            self.load_mp_all_by_pickle(self.args['mp_pickle_path'])
            self.load_conceptnet_numberbatch(self.args['conceptnet_numberbatch_en_path'])

    def get_features_from_words(self, words):
        words = standardized_uri('en', words).replace('/c/en/', '')
        res = self.conceptnet_numberbatch_en.get(words)
        if res is None:
            res = self.get_default_oov_feature()
        return res

    def get_default_oov_feature(self):
        return [0.0 for _ in range(300)]

    def load_conceptnet_numberbatch(self, fpath):
        if len(self.conceptnet_numberbatch_en) != 0:
            return
        self.conceptnet_numberbatch_en.clear()
        with open(fpath, 'r', encoding='UTF-8') as f:
            n, vec_size = list(map(int, f.readline().split(' ')))
            print('load conceptnet numberbatch: ', n, vec_size)
            for i in range(n):
                tmp_data = f.readline().split(' ')
                words = str(tmp_data[0])
                vector = list(map(float, tmp_data[1:]))
                self.conceptnet_numberbatch_en[words] = vector
            print('load conceptnet numberbatch done!')

    def get_words_from_id(self, id):
        return self.words_to_id.inverse.get(id)

    def get_id_from_words(self, words, is_add=False):
        if self.words_to_id.get(words) is None and is_add:
            self.words_to_id[words] = self.words_encode_idx
            self.words_encode_idx += 1
        return self.words_to_id.get(words)

    def encode_index(self, mp):
        x_index_id = []
        edge_index = []
        edge_weight = []
        self.words_encode_idx = 0
        self.words_to_id.clear()
        # 建边
        for key, values in mp.items():
            st_id = self.get_id_from_words(key, is_add=True)
            x_index_id.append(st_id)
            for value in values:
                to_words = value[0]
                to_relation = value[1]
                to_weight = value[2]

                ed_id = self.get_id_from_words(to_words, is_add=True)

                edge_index.append([st_id, ed_id])
                edge_weight.append(to_weight)
                edge_index.append([ed_id, st_id])
                edge_weight.append(to_weight)
        x = [self.get_features_from_words(self.get_words_from_id(i)) for i in range(self.words_encode_idx)]
        x_index = torch.zeros(len(x), dtype=torch.bool)
        x_index[x_index_id] = 1
        return torch.tensor(x), x_index, torch.tensor(edge_index, dtype=torch.long).t(), torch.tensor(edge_weight)

    def get_submp_by_sentences(self, sentences: list, is_merge=False):

        def get_submp_by_one(mp_all, sent, n_gram=1, stop_words=[]):
            lemmatizer = WordNetLemmatizer()
            mp_sub = defaultdict(set)
            sent = sent.strip(',|.|?|;|:|!').lower()
            tokens = self.tokenizer.tokenize(sent)
            tokens += ['#'] + re.sub('[^a-zA-Z0-9,]', ' ', sent).split(' ')
            for gram in range(1, n_gram + 1):
                start, end = 0, gram
                while end <= len(tokens):
                    q_words = '_'.join(tokens[start:end])
                    start, end = start + 1, end + 1

                    if gram == 1 and q_words in stop_words:
                        continue
                    if q_words.find('#') != -1:
                        continue
                    if gram == 1:
                        q_words = lemmatizer.lemmatize(q_words, pos='n')

                    if mp_all.get(q_words) is not None and mp_sub.get(q_words) is None:
                        mp_sub[q_words] |= mp_all[q_words]
            return mp_sub

        if is_merge:
            sent = ' '.join(sentences)
            sentences = [sent]

        res = []
        for i in sentences:
            res.append(get_submp_by_one(self.mp_all, i, self.args['n_gram'],
                                        stop_words=self.args['reduce_noise_args']['stop_words']))
        return res

