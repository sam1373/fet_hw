import numpy as np
import random
import logging

import datautils

import torch
import torch.nn as nn

from os.path import join

TOKEN_UNK = '<UNK>'
TOKEN_ZERO_PAD = '<ZPAD>'
TOKEN_EMPTY_PAD = '<EPAD>'
TOKEN_MENTION = '<MEN>'

EL_DIR = 'el_files'

EL_CANDIDATES_DATA_FILE = join(EL_DIR, 'enwiki-20151002-candidate-gen.pkl')
WIKI_FETEL_WORDVEC_FILE = join(EL_DIR, 'enwiki-20151002-nef-wv-glv840B300d.pkl')
WIKI_ANCHOR_SENTS_FILE = join(EL_DIR, 'enwiki-20151002-anchor-sents.txt')
TYPE_VOCAB = join(EL_DIR, 'figer-type-vocab.txt')
WID_TYPES_FILE = join(EL_DIR, 'wid-types-figer.txt')


def get_parent_type(t):
    p = t.rfind('/')
    if p <= 1:
        return None
    return t[:p]


def get_parent_types(t):
    parents = list()
    while True:
        p = get_parent_type(t)
        if p is None:
            return parents
        parents.append(p)
        t = p


def get_parent_type_ids_dict(type_id_dict):
    d = dict()
    for t, type_id in type_id_dict.items():
        d[type_id] = [type_id_dict[p] for p in get_parent_types(t)]
    return d


class GlobalRes:
    def __init__(self, type_vocab_file, word_vecs_file):
        self.type_vocab, self.type_id_dict = datautils.load_type_vocab(type_vocab_file)
        self.parent_type_ids_dict = get_parent_type_ids_dict(self.type_id_dict)
        self.n_types = len(self.type_vocab)

        print('loading {} ...'.format(word_vecs_file), end=' ', flush=True)
        self.token_vocab, self.token_vecs = datautils.load_pickle_data(word_vecs_file)
        self.token_id_dict = {t: i for i, t in enumerate(self.token_vocab)}
        print('done', flush=True)
        self.zero_pad_token_id = self.token_id_dict[TOKEN_ZERO_PAD]
        self.mention_token_id = self.token_id_dict[TOKEN_MENTION]
        self.unknown_token_id = self.token_id_dict[TOKEN_UNK]
        self.embedding_layer = nn.Embedding.from_pretrained(torch.from_numpy(self.token_vecs))
        self.embedding_layer.padding_idx = self.zero_pad_token_id
        self.embedding_layer.weight.requires_grad = False
        # self.embedding_layer.share_memory()




class ELDirectEntityVec:
    def __init__(self, n_types, type_to_id_dict, el_system, wid_types_file):
        self.n_types = n_types
        self.el_system = el_system
        self.rand_assign_rate = 1.1
        print('loading {} ...'.format(wid_types_file))
        logging.info('rand_assign_rate={}'.format(self.rand_assign_rate))
        self.wid_types_dict = datautils.load_wid_types_file(wid_types_file, type_to_id_dict)

    def get_entity_vecs(self, mention_strs, prev_pred_results, min_popularity=10, true_wids=None,
                        filter_by_pop=False, person_type_id=None, person_l2_type_ids=None, type_vocab=None):
        all_entity_vecs = np.zeros((len(mention_strs), self.n_types), np.float32)
        el_sgns = np.zeros(len(mention_strs), np.float32)
        probs = np.zeros(len(mention_strs), np.float32)
        candidates_list = self.el_system.link_all(mention_strs, prev_pred_results)
        #print(candidates_list)
        for i, el_candidates in enumerate(candidates_list):
            # el_candidates = self.el_system.link(mstr)
            if not el_candidates:
                continue
            wid, mstr_target_cnt, popularity = el_candidates[0]
            if filter_by_pop and popularity < min_popularity:
                continue
            types = self.wid_types_dict.get(wid, None)

            if types is None:
                continue

            #print(wid, types)
            #print([type_vocab[i] for i in types])

            probs[i] = mstr_target_cnt / (sum([cand[1] for cand in el_candidates]) + 1e-7)

            el_sgns[i] = 1
            for type_id in types:
                all_entity_vecs[i][type_id] = 1

            if person_type_id is not None and person_type_id in types and (
                    self.rand_assign_rate >= 1.0 or np.random.uniform() < self.rand_assign_rate):
                for _ in range(3):
                    rand_person_type_id = person_l2_type_ids[random.randint(0, len(person_l2_type_ids) - 1)]
                    if all_entity_vecs[i][rand_person_type_id] < 1.0:
                        all_entity_vecs[i][rand_person_type_id] = 1.0
                        break
        return all_entity_vecs, el_sgns, probs


def __get_l2_person_type_ids(type_vocab):
    person_type_ids = list()
    for i, t in enumerate(type_vocab):
        if t.startswith('/person') and t != '/person':
            person_type_ids.append(i)
    return person_type_ids


def get_entity_vecs_for_samples(el_entityvec: ELDirectEntityVec, mention_strs,
                                  filter_by_pop=False, person_type_id=None, person_l2_type_ids=None, type_vocab=None):
    prev_pred_labels = None
    return el_entityvec.get_entity_vecs(
        mention_strs, prev_pred_labels, filter_by_pop=filter_by_pop, person_type_id=person_type_id,
        person_l2_type_ids=person_l2_type_ids, type_vocab=type_vocab)