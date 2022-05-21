from typing import Dict, List, Callable, TypeVar
from overrides import overrides
from itertools import chain
import os
from .. import Vocabulary
from . import TokenIndexer
from transformers import AutoTokenizer
Indices = TypeVar("Indices", List[int], List[List[int]])


class TokenizerIndexer(TokenIndexer):
    """
    A ``CharTokenIndexer`` determines how string token get represented as
    arrays of list of character indices in a model.

    Parameters
    ----------
    related_vocabs : ``List[str]``
        Which vocabularies are related to the indexer.
    transform : ``Callable[[str,], str]``, optional (default=``lambda x:x``)
        What changes need to be made to the token when counting or indexing.
        Commonly used are lowercase transformation functions.
    """

    def __init__(
            self,
            transformers_path) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(os.environ['TRANSFORMERS_CACHE']+transformers_path)

    @overrides
    def count_vocab_items(
            self,
            token: str,
            counters: Dict[str, Dict[str, int]]) -> None:
        """
        Each character in the token is counted directly as an element.

        Parameters
        ----------
        counter : ``Dict[str, Dict[str, int]]``
            We count the number of strings if the string needs to be counted to
            some counters.
        """
        pass

    @overrides
    def tokens_to_indices(
            self,
            tokens: List[str],
            vocab: Vocabulary) -> Dict[str, List[List[int]]]:
        """
        Takes a list of tokens and converts them to one or more sets of indices.
        During the indexing process, each token item corresponds to a list of
        index in the vocabulary.

        Parameters
        ----------
        vocab : ``Vocabulary``
            ``vocab`` is used to get the index of each item.
        """
        res = {}
        res['input_ids'] = [self.tokenizer.encode(tok, add_special_tokens=False)
                            for tok in tokens]
        res['length'] = list(map(len, res['input_ids']))
        res['input_ids'] = self.tokenizer.build_inputs_with_special_tokens(list(chain.from_iterable(res['input_ids'])))
        return res

