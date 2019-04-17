#   Author: Anna Kozlova
#   Created: 26/01/2019

from mention_classification.pair_labeling import TrainSetGeneration
from mention_classification.parse_dialog_mentions import ParseDialogAnnotation
from mention_classification.get_embeddings import WordContextEmbeddings
from mention_detection.mention_detection import get_mentions

if __name__ == '__main__':
    text = "Новозеландская исследовательница китайской политики, которая написала статью."
    annotation = {"0": {"0": {"sh": "0", "ln": "51"}, "1": {"sh": "53", "ln": "7"}},
                  "1": {
                      "0": {"sh": "53", "ln": "7"}, "1": {"sh": "61", "ln": "8"}, "2": {"sh": "70", "ln": "6"}
                  }
                  }
    ann_mentions = ParseDialogAnnotation(text, annotation).get_mentions_from_annotation()
    mentions = get_mentions(text)
    pairs = TrainSetGeneration(mentions, ann_mentions).label_pairs()

    wce = WordContextEmbeddings(text, mentions)

    labeled_pairs = []
    for pair in pairs:
        mention = tuple(pair[0])
        print(mention)
        labeled_pairs.append((wce.get_mention_with_context_embedding(mention), pair[1]))

    print(len(labeled_pairs), labeled_pairs[0])
