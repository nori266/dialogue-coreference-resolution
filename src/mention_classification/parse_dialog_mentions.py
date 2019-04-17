#   Author: Anna Kozlova
#   Created: 26/01/2019

from typing import Any
from typing import Dict
from typing import List
from typing import Set
from typing import Tuple

from mention_detection.tokenization import make_start_to_token_dict


class ParseDialogAnnotation:
    def __init__(self, doc: str, annotation_json: Dict[str, Any]):
        self.doc = doc
        self.annotation_json = annotation_json
        self.start_to_token = make_start_to_token_dict(self.doc)
        self.token_to_start = {token[0]: start for (start, token) in self.start_to_token.items()}

    def get_mentions_from_annotation(self) -> List[Set[Tuple[int]]]:
        coref_chains = []

        for coref_chain in self.annotation_json.values():
            mentions = coref_chain.values()
            coref_chain = set()

            for mention in mentions:
                start = int(mention["sh"])
                length = int(mention["ln"])
                my_mention = self.__get_mention(start, length)

                if my_mention is not None:
                    coref_chain.add(my_mention)

            coref_chains.append(coref_chain)

        return coref_chains

    def __get_mention(self, start_annotated_mention: int, length_annotated_mention: int):
        if start_annotated_mention in self.start_to_token:  # TODO search in some interval

            first_token = self.start_to_token[start_annotated_mention]  # (1, 'исследовательница')
            first_token_number = first_token[0]
            last_token_number = first_token_number

            end_annotated_mention = start_annotated_mention + length_annotated_mention

            if len(self.token_to_start.keys()) > last_token_number + 1:
                start_next_token = self.token_to_start[last_token_number + 1]
            else:
                return first_token_number, last_token_number

            while end_annotated_mention > start_next_token and len(self.token_to_start.keys()) > last_token_number + 1:
                last_token_number += 1
                start_next_token = (self.token_to_start[last_token_number+1])

            return first_token_number, last_token_number
        print (f'mention with start {start_annotated_mention} is not found')
        return None