from typing import List, Dict, Iterator
from overrides import overrides
from ..token_indexers import TokenIndexer
from .. import Vocabulary
from . import Field


class MetaField(Field):
    """
    A ``MetaField`` is a raw token field, and we can use it to store raw data.

    Parameters
    ----------
    name : ``str``
        Field name. This is necessary and must be unique (not the same as other
        field names).
    tokens : ``List[str]``
        Field content that contains a list of string.
    """

    def __init__(self,
                 name: str,
                 tokens: List[str]):
        self.name = name
        self.tokens = tokens

    def __iter__(self) -> Iterator[str]:
        return iter(self.tokens)

    def __getitem__(self, idx: int) -> str:
        return self.tokens[idx]

    def __len__(self) -> int:
        return len(self.tokens)

    def __str__(self) -> str:
        return '{}: [{}]'.format(self.name, (self.tokens))

    @overrides
    def count_vocab_items(self, counters: Dict[str, Dict[str, int]]) -> None:
        """
        ``MetaField`` doesn't need index operation.
        """
        pass

    @overrides
    def index(self, vocab: Vocabulary) -> None:
        """
        ``MetaField`` doesn't need index operation.
        """
        self.indexes = self.tokens
