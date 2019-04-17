#   Author: Anna Kozlova
#   Created: 26/01/2019

from typing import Dict
from typing import List
from typing import Tuple

import numpy as np
import tensorflow as tf
import tensorflow_hub as hub

from mention_detection.tokenization import make_start_to_token_dict


elmo = hub.Module("http://files.deeppavlov.ai/deeppavlov_data/elmo_ru-news_wmt11-16_1.5M_steps.tar.gz")


def get_sequence_embeddings(tokens: List[List[str]]):
    lens = [len(sentence) for sentence in tokens]

    embeddings = elmo(
        {
            'tokens': tokens,
            'sequence_len': lens,  # check if it is valid for all cases
        },
        signature='tokens',
        as_dict=True)["elmo"]

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        return sess.run(embeddings)


class WordContextEmbeddings:
    def __init__(self, doc, mentions, left_context_len=3, right_context_len=3):
        self.doc = doc
        self.mentions = mentions

        self.left_context_len = left_context_len
        self.right_context_len = right_context_len

        self.start2token: Dict[int, Tuple[int, str]] = make_start_to_token_dict(doc)
        self.token2word: Dict[int, str] = {value[0]: value[1] for value in self.start2token.values()}

        self.doc_embeddings = get_sequence_embeddings([list(self.token2word.values())])

    def get_left_context(self, mention: Tuple[int, int]):
        # if there are not enough words for left context
        if mention[0] < self.left_context_len:
            # take all the words from the beginning
            context = (0, mention[0] - 1)
        else:  # normal case
            context = (mention[0] - self.left_context_len, mention[0] - 1)
        # return get_sequence_embeddings([context])
        return context

    def get_right_context(self, mention: Tuple[int, int]):
        text_token_number = len(self.token2word.items())
        # if there are not enough words for right context
        if mention[1] >= text_token_number - self.right_context_len:
            # take all the words from the left
            context = (mention[1]+1, text_token_number - 1)
        else:  # normal case
            context = (mention[1]+1, mention[1] + self.right_context_len)
        # return get_sequence_embeddings([context])
        return context

    def get_embedding_for_tokens(self, tokens: Tuple[int,int]):
        start = tokens[0]
        end = tokens[1] + 1
        embed = np.mean(self.doc_embeddings[:, start: end], axis=1).flatten()

        return embed

    def get_mention_with_context_embedding(self, mention: Tuple[int, int]):
        left_context = self.get_embedding_for_tokens(self.get_left_context(mention))
        right_context = self.get_embedding_for_tokens(self.get_right_context(mention))
        mention_tokens = self.get_embedding_for_tokens(mention)

        return np.concatenate([left_context, mention_tokens, right_context])
