from typing import List, Iterator, Dict
from overrides import overrides
from ..token_indexers import TokenIndexer
from .. import Vocabulary
from . import Field


class TripleField(Field):
    """
    A ``TripleField`` is a data field that is commonly used in NLP Relation Extraction task, and we
    can use it to store triples in an instance.

    Parameters
    ----------
    name : ``str``
        Field name. This is necessary and must be unique (not the same as other
        field names).
    tokens : ``List``
        Field content contains a list of triple. The shape of a triple is like [entity1_description, entity2_description, relation].
    indexers : ``List[TokenIndexer]``, optional (default=``list()``)
        Indexer list that defines the vocabularies associated with the field.
    """

    def __init__(self,
                 name: str,
                 tokens: List,
                 indexers: List[TokenIndexer] = list()):
        self.name = name
        self.tokens = tokens
        self.indexers = indexers

    def __iter__(self) -> Iterator[str]:
        return iter(self.tokens)

    def __getitem__(self, idx: int) -> str:
        return self.tokens[idx]

    def __len__(self) -> int:
        return len(self.tokens)

    def __str__(self) -> str:
        return '{}: [{}]'.format(self.name, ', '.join(self.tokens))

    @overrides
    def count_vocab_items(self, counters: Dict[str, Dict[str, int]]) -> None:
        """
        We count the number of relations if the relation needs to be counted to some
         counters. You can pass directly if there is no string that needs
        to be counted.

        Parameters
        ----------
        counters : ``Dict[str, Dict[str, int]]``
            Element statistics for datasets. if field indexers indicate that
            this field is related to some counters, we use field content to
            update the counters.
        """
        for idxer in self.indexers:
            for token in self.tokens:
                idxer.count_vocab_items(token[2], counters)

    @overrides
    def index(self, vocab: Vocabulary) -> None:
        """
        Gets one or more index mappings for each relation in the Field.

        Parameters
        ----------
        vocab : ``Vocabulary``
            ``vocab`` is used to get the index of each relation
        """
        self.indexes = {}
        self.indexes['rel'] = []
        for idxer in self.indexers:
            rels = [token[2] for token in self.tokens]
            rel_indices = idxer.tokens_to_indices(rels, vocab)
            
            for i in range(len(self.tokens)):
                token = self.tokens[i]
                rel_index = rel_indices['rel'][i]
                triple_index = [token[0], token[1], rel_index]

                self.indexes['rel'].append(triple_index)