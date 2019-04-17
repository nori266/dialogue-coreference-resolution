#   Author: Anna Kozlova
#   Created: 26/01/2019

from typing import Dict
from typing import List
from typing import Tuple

from mention_classification.parse_dialog_mentions import ParseDialogAnnotation
from mention_detection.mention_detection import get_mentions
from mention_detection.tokenization import make_start_to_token_dict


class MentionComparison:
    """
    Helps to find out what kind of mentions from annotation do not correspond
    to my extracted mentions.

    To run:

    comp = MentionComparison(doc, annotation)
    annotated_words, my_words, annotated_extra_words, my_extra_words = comp.compare_mentions()
    """
    def __init__(self, doc: str, annotation: Dict):

        self.doc = doc
        self.annotation = annotation

        self.start2token: Dict[int, Tuple[int, str]] = make_start_to_token_dict(doc)
        self.token2word: Dict[int, str] = {value[0]: value[1] for value in self.start2token.values()}

    def compare_mentions(self) -> Tuple[List, List, List, List]:
        """
        :return: tuple of extra mentions from both sources:

        annotated_words - mentions that were in annotation
        my_words - mentions detected with my mention_detection
        annotated_extra_words - mentions that are in annotation, but not in my detection (most interesting part)
        my_extra_words - mention that are in my detection, but not in the annotation
        """
        annotated_mentions: List[List[Tuple[int, int]]] = ParseDialogAnnotation(
                                                            self.doc, self.annotation
                                                        ).get_mentions_from_annotation()

        flat_annotated_mentions = [mention for coref_chain in annotated_mentions
                                   for mention in coref_chain]
        set_annotated = set(flat_annotated_mentions)

        my_mentions = get_mentions(self.doc)
        set_my = set(my_mentions)

        annotated_extra = set_annotated - set_my
        my_extra = set_my - set_annotated

        annotated_extra_words = [self.get_tokens_by_mention(mention) for mention in annotated_extra]
        my_extra_words = [self.get_tokens_by_mention(mention) for mention in my_extra]
        annotated_words = [self.get_tokens_by_mention(mention) for mention in set_annotated]
        my_words = [self.get_tokens_by_mention(mention) for mention in set_my]

        return annotated_words, my_words, annotated_extra_words, my_extra_words

    def get_tokens_by_mention(self, mention: Tuple[int, int]) -> List[Tuple[int, str]]:
        """Returns a list of tokens with theis token numbers.

        :param mention: tuple of token numbers like (7, 8) - first and last tokens
        :return: list of tokens with theis token numbers
        """
        mention_tokens = []
        for token_number in range(mention[0], mention[1]+1):
            mention_tokens.append((token_number, self.token2word[token_number]))
        return mention_tokens
