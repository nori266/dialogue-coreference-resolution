#   Author: Anna Kozlova
#   Created: 26/01/2019

from itertools import combinations
from typing import List
from typing import Set
from typing import Tuple


class TrainSetGeneration:
    def __init__(
            self,
            doc_mentions: List[Tuple[int, int]],
            annotated_coref_chains: List[Set[Tuple[int]]]
    ) -> None:
        self.doc_mentions = doc_mentions
        self.annotated_coref_chains = annotated_coref_chains

    def __generate_mention_pairs(self) -> List[Tuple[Tuple[int, int]]]:
        """Returns all possible mention combination without repetition.

        :return: list of pairs
        """
        return self.__filter_pairs(list(combinations(self.doc_mentions, 2)))

    def __do_intersect(self, mention1: Tuple[int, int], mention2: Tuple[int, int]) -> bool:
        """Checks if the two mentions intersect.

        :param mention1:
        :param mention2:
        :return: True or False
        """
        if mention1[0] == mention2[0] or mention1[1] == mention2[1]:
            return True

        if mention1[0] < mention2[0]:
            earlier_mention = mention1
            latest_mention = mention2
        else:
            earlier_mention = mention2
            latest_mention = mention1

        if earlier_mention[1] >= latest_mention[0]:
            return True

        return False

    def __filter_pairs(self, pairs: List[Tuple[Tuple[int, int]]]) -> List[Tuple[Tuple[int, int]]]:
        """Delete pairs that have token intersection.
        For example: "исследовательница китайской политики" and "исследовательница"
        are obviously not coreferent, because it is the same mention.

        :param pairs: generated pairs
        :return: filtered pairs
        """
        return [pair for pair in pairs if not self.__do_intersect(pair[0], pair[1])]

    def label_pairs(self):
        labeled_pairs = []
        # Provides sets instead of tuples, because order in pair in unimportant.
        pairs = [set(pair) for pair in self.__generate_mention_pairs()]

        for pair in pairs:
            for chain in self.annotated_coref_chains:
                if pair.issubset(chain):
                    labeled_pairs.append((pair, True))
                    break
            else:  # if did not find any chain for this pair
                labeled_pairs.append((pair, False))

        return labeled_pairs
