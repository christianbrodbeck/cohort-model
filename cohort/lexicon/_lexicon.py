# Author: Christian Brodbeck <christianbrodbeck@nyu.edu>
"""Estimate lexical probabilities associates with phonetic input
"""
from __future__ import annotations

from collections import defaultdict
import contextlib
from functools import cached_property
from itertools import chain
from math import log
from pathlib import Path
import re
from typing import Dict, Collection, Iterator, List, Literal, Optional, Sequence, Tuple, Union

import attr
from eelbrain import fmtxt
from trftools.align import TextGrid


@attr.s(auto_attribs=True, slots=True, repr=False)
class Pronunciation:
    """Pronunciation represented as sequence of phonemes"""
    phones: Tuple[str, ...]  # Sequence of phonemes

    def string(self):
        return ' '.join(self.phones)

    def __repr__(self):
        return f"<Pronunciation: {self.string()}>"

    def __len__(self):
        return len(self.phones)

    def __hash__(self):
        return hash(self.phones)

    def __add__(self, other):
        if isinstance(other, str):
            return Pronunciation(self.phones + tuple(other.split(' ')))
        elif isinstance(other, Pronunciation):
            return Pronunciation(self.phones + other.phones)
        else:
            raise TypeError(repr(other))

    def __getitem__(self, item):
        if isinstance(item, slice):
            return Pronunciation(self.phones[item])
        elif isinstance(item, int):
            return self.phones[item]
        else:
            raise TypeError(f"Invalid index for Pronunciation: {item!r}")

    def startswith(self, prefix):
        return self.phones[:len(prefix)] == prefix.phones

    def endswith(self, postfix):
        return self.phones[-len(postfix):] == postfix.phones

    def is_neighbor(self, other: Pronunciation):
        n_self = len(self.phones)
        n_other = len(other.phones)
        if n_self == n_other:
            return sum([ps == po for ps, po in zip(self.phones, other.phones)]) == n_self - 1
        elif abs(n_self - n_other) == 1:
            n_pre = sum([ps == po for ps, po in zip(self.phones, other.phones)])
            n_post = sum([ps == po for ps, po in zip(self.phones[::-1], other.phones[::-1])])
            return n_pre + n_post == min(n_self, n_other)
        else:
            return False

    def is_silence(self):
        return not self.string().strip()

    @classmethod
    def coerce(cls, pronunciation: PronunciationArg):
        if isinstance(pronunciation, cls):
            return pronunciation
        elif isinstance(pronunciation, str):
            if pronunciation.strip():
                return cls(tuple(pronunciation.split(' ')))
            else:
                return cls((' ',))
        elif isinstance(pronunciation, Sequence):
            if not isinstance(pronunciation, tuple):
                pronunciation = tuple(pronunciation)
            return Pronunciation(pronunciation)
        else:
            raise TypeError(f"pronunciation={pronunciation!r}")

    @classmethod
    def from_nested_string(cls, string, phone_re=None):
        if '[' in string:
            out = ['']
            while True:
                i = string.find('[')
                if i == -1:
                    out = [s + string for s in out]
                    break
                substring = string[:i]
                string = string[i+1:]
                alternatives, i_start = cls._parse_alternatives(string)
                out = [s + substring + alt for s in out for alt in alternatives]
                string = string[i_start:]
        else:
            out = [string]
        if phone_re:
            out = [re.findall(phone_re, s) for s in out]
        return [cls(s) for s in out]

    @staticmethod
    def _parse_alternatives(string: str) -> (List[str], int):
        # pattern is [ x / y ]
        # x or y can be [ u / v ] recursively
        # assumes that leading [ is removed from string
        out = []
        alternatives = {''}
        i = i_start = 0
        while True:
            c = string[i]
            if c not in '[/]':
                i += 1
                continue

            if i > i_start:
                substring = string[i_start:i]
                alternatives = {a + substring for a in alternatives}

            if c == '[':
                inner_alternatives, i_start = Pronunciation._parse_alternatives(string[i+1:])
                alternatives = {a + ia for a in alternatives for ia in inner_alternatives}
                i = i_start = i + i_start + 1
            elif c == '/':
                out.append(alternatives)
                alternatives = {''}
                i_start = i = i + 1
            elif c == ']':
                for a in out:
                    alternatives.update(a)
                return alternatives, i + 1


@attr.s(auto_attribs=True, slots=True, repr=False)
class Word:
    """Represent a lexical item"""
    graphs: str  # Word in lower case
    pronunciations: Tuple[Pronunciation]
    segmentation: Tuple[str]  # Morphological segmentation; can be None
    activation: float  # updated dynamically

    def __repr__(self):
        ps = ', '.join([p.string() for p in self.pronunciations])
        if self.segmentation is None:
            seg = ''
        else:
            seg = ', ' + '+'.join(self.segmentation)
        return f"<Word {self.graphs!r}: ({ps}){seg}, activation={self.activation:g}>"

    def is_neighbor(self, other: Pronunciation):
        return any(other.is_neighbor(p) for p in self.pronunciations)

    def startswith(self, prefix):
        return any(p.startswith(prefix) for p in self.pronunciations)

    def endswith(self, postfix):
        return any(p.endswith(postfix) for p in self.pronunciations)


@attr.s(auto_attribs=True, repr=False)
class Cohort:
    prefix: Pronunciation
    words: List[Word]

    def __attrs_post_init__(self):
        self._cohorts = {}  # cache

    def __repr__(self):
        return f"<{self._desc()}>"

    def __str__(self):
        return str(self.table(15))

    def _desc(self):
        return f"Cohort {self.prefix.string()}, {len(self.words)} words, {self._form_count} pronunciations"

    @cached_property
    def complete_words(self):
        return [word for word in self.words if self.prefix in word.pronunciations]

    @cached_property
    def incomplete_words(self):
        return [word for word in self.words if self.prefix not in word.pronunciations]

    def add_phone(self, phone: str) -> Cohort:
        if phone not in self._cohorts:
            prefix = Pronunciation((*self.prefix.phones, phone))
            self._cohorts[phone] = Cohort(prefix, [w for w in self.words if w.startswith(prefix)])
        return self._cohorts[phone]

    @cached_property
    def segmented_words(self):
        return [w for w in self.words if w.segmentation is not None]

    def entropy(self, model: str, smooth: float = 0):
        if model == 'form':
            activations = (word.activation for word in self.words)
        elif model == 'morpheme':
            activations = self._morpheme_activation().values()
        elif model == 'lemma':
            activations = self._lemma_counts.values()
        else:
            raise ValueError(f"{model=}")

        if smooth:
            activations = [a + smooth for a in activations]
        else:
            activations = [a for a in activations if a]

        total_activation = sum(activations)
        ps = (a / total_activation for a in activations)
        return -sum(p * log(p, 2) for p in ps)

    def phoneme_entropy(self, smooth: float = 0):
        """Entropy for next phoneme given the cohort"""
        i = len(self.prefix)
        # collect activation of next phoneme continuations
        activations = defaultdict(lambda: 0)
        for word in self.words:
            next_phones = {p.phones[i] if len(p.phones) > i else '' for p in word.pronunciations if p.startswith(self.prefix)}
            activation = (word.activation + smooth) / len(next_phones)
            for p in next_phones:
                activations[p] += activation
        # remove 0
        if not smooth:
            activations = {ph: activation for ph, activation in activations.items() if activation}
        # renormalize
        total = sum(activations.values())
        ps = [activation / total for activation in activations.values()]
        return -sum(p * log(p, 2) for p in ps)

    @cached_property
    def _form_count(self):
        return sum(p.startswith(self.prefix) for word in self.words for p in word.pronunciations)

    def activation(self):
        """Combined (sum) activation of all cohort members"""
        return sum(word.activation for word in self.words)

    def size(self, model):
        """Cohort size

        Parameters
        ----------
        model : 'form' | 'morpheme' | 'lemma'
            Model used to compute cohort size.
        """
        if model == 'form':
            return self._form_count
        elif model == 'morpheme':
            return self._morpheme_count
        elif model == 'lemma':
            return self._lemma_count
        else:
            raise ValueError(f"model={model!r}")

    @cached_property
    def _lemma_counts(self):
        # TODO: Exclude inflectional morphemes
        raise NotImplementedError()

    @cached_property
    def _lemma_count(self):
        return len(self._lemma_counts)

    @cached_property
    def _morpheme_cohort(self):
        "(i, morphemes) index and cohort of the morpheme that is currently being processed"
        if len(self.segmented_words) == 0:
            return -1, set()
        morpheme_i = 0
        while True:
            # number of morphemes at this stage
            morpheme_cohort = {w.segmentation[morpheme_i] for w in self.segmented_words}
            if len(morpheme_cohort) > 1:
                return morpheme_i, morpheme_cohort

            morpheme_i += 1
            # if we're not sure whether we're expecting another morpheme, we're
            # not processing it:
            if any(len(w.segmentation) <= morpheme_i for w in self.segmented_words):
                return morpheme_i - 1, morpheme_cohort

    @cached_property
    def morpheme_i(self):
        "Index of the morpheme that is currently being processed"
        return self._morpheme_cohort[0]

    def _morpheme_activation(self):
        """{morpheme: activation} dict

        As long as there are more than one morpheme in segmentation[0] we have
        not reached UP0. Once there is only one morpheme, we reached UP0. We
        stay at morpheme 0 until the word with segmentation [segmentation[0]]
        drops out of the cohort.
        """
        # which morpheme is being processed?
        morpheme_i, mcohort = self._morpheme_cohort
        # summed frequencies for each continuation
        lf = {m: 0. for m in mcohort}
        for w in self.segmented_words:
            lf[w.segmentation[morpheme_i]] += w.activation
        return lf

    @cached_property
    def _morpheme_count(self):
        """Number of morphemes in the cohort"""
        return len(self._morpheme_cohort[1])

    def table(
            self,
            n: int = None,
            segmented: bool = False,
            sort: str = None,
    ):
        """List cohort members

        Parameters
        ----------
        n
            Number of members to list (default is all members).
        segmented
            Only include words with morphological segmentation.
        sort
            Set to ``'activation'`` to sort the table by the words' activation
            value.
        """
        all_words = self.segmented_words if segmented else self.words
        if sort == 'activation':
            all_words = sorted(all_words, key=lambda word: word.activation, reverse=True)
        elif sort:
            raise ValueError(f'{sort=}')
        # number of words
        drop_words = n is not None and n < len(all_words)
        show_words = all_words[:n] if drop_words else all_words
        # data columns
        graphs = []
        segmentations = []
        activation = []
        pronunciations = []
        for word in show_words:
            graphs.append(word.graphs)
            segmentation = '' if word.segmentation is None else '+'.join(word.segmentation)
            segmentations.append(segmentation)
            activation.append(word.activation)
            i = 0
            for pronunciation in word.pronunciations:
                if not pronunciation.startswith(self.prefix):
                    continue
                pronunciations.append(pronunciation.string())
                if i > 0:
                    graphs.append('')
                    segmentations.append('')
                    activation.append('')
                i += 1
        # table options
        segmentation_columns = any(segmentations)
        # determine table columns
        headers = ['Graphs', 'Pronunciation']
        columns = 'll'
        data = [graphs, pronunciations]
        if segmentation_columns:
            headers.append("Segmentation")
            columns += 'l'
            data.append(segmentations)
        headers.append('Activation')
        columns += 'r'
        data.append(activation)
        # table
        table = fmtxt.Table(columns)
        table.cells(*headers)
        table.midrule()
        for row in zip(*data):
            table.cells(*row)
        if drop_words:
            table.cell('...')
        # sum row
        total = sum(word.activation for word in all_words)
        table.midrule()
        table.cell(len(all_words))
        table.cells(*([''] * (1 + segmentation_columns)))
        table.cell(total)
        table.caption(self._desc())
        return table


@attr.s(auto_attribs=True, repr=False)
class Lexicon:
    "Compute lexical properties such as phoneme-by-phoneme entropy"
    words: List[Word]  # words in the lexicon

    def __attrs_post_init__(self):
        self._cohort = Cohort(Pronunciation(()), self.words)

    def __repr__(self):
        return f"<Lexicon: {len(self.words)} words>"

    def __getstate__(self):
        return {'words': self.words}

    def __setstate__(self, state):
        self.__init__(**state)

    @contextlib.contextmanager
    def preserve_activation(self):
        activations = [word.activation for word in self.words]
        try:
            yield
        finally:
            for word, activation in zip(self.words, activations):
                word.activation = activation

    @cached_property
    def _word_dict(self):
        word_dict = defaultdict(tuple)
        for word in self.words:
            word_dict[word.graphs] += (word,)
        return word_dict

    def cohort(self, prefix: PronunciationArg) -> Cohort:
        """Cohort of the given prefix

        Parameters
        ----------
        prefix
            Space-delimited arpabet phones.

        Returns
        -------
        cohort
            Cohort containing all words compatible with prefix.
        """
        if not prefix:
            return self._cohort
        cohort = None
        for cohort in self.cohorts(prefix):
            pass
        return cohort

    def cohorts(self, phones: PronunciationArg) -> Iterator[Cohort]:
        "Iterate over cohorts for each phoneme"
        phones = Pronunciation.coerce(phones)
        if phones.is_silence():
            return Cohort(phones, []),
        cohort: Cohort = self._cohort
        for phone in phones.phones:
            cohort = cohort.add_phone(phone)
            yield cohort

    def lookup(self, graphs: str) -> Tuple[Word, ...]:
        return self._word_dict.get(graphs, ())

    def lookup_pronunciation(self, phones: PronunciationArg) -> List[Word]:
        "Look up all words that have a pronunciation matching ``phones``"
        phones = Pronunciation.coerce(phones)
        return [w for w in self.words if phones in w.pronunciations]

    def neighbors(self, seed: PronunciationArg) -> List[Word]:
        """Find phonological neighbors

        Paramaters
        ----------
        seed
            Space-delimited arpabet phones.

        Returns
        -------
        neighbors
            All neighbors.
        """
        seed_pronunciation = Pronunciation.coerce(seed)
        return [word for word in self.words if word.is_neighbor(seed_pronunciation)]

    def uniqueness_point(self, phones, model, n=1):
        """Index of the phoneme that reduces the cohort to one.

        Returns the index of the last phoneme + 1 if the sequence is not unique.

        Parameters
        ----------
        phones : tuple of str
            Phonemes.
        model : str
            Decomposition model to use.
        n : int
            Find point where cohort size first reaches a number ``<= n``
            (default 1, i.e. the uniqueness point).
        """
        i = 0
        for i, cohort in enumerate(self.cohorts(phones)):
            if cohort.size(model) <= n:
                return i
        return i + 1

    # Predictors
    # ----------
    # All these functions return a list of values with one entry per phoneme
    def buffer(self, prefix, model='form'):
        "Number of items in phoneme buffer"
        if model is None:
            model = 'form'
        prefix = Pronunciation.coerce(prefix)
        if prefix.is_silence():
            return 0,
        if model == 'form':
            return tuple(range(1, len(prefix) + 1))
        elif model == 'morpheme':
            out = []
            morpheme_i = 0
            buffer_size = 0
            for cohort in self.cohorts(prefix):
                if cohort.morpheme_i > morpheme_i:
                    morpheme_i = cohort.morpheme_i
                    buffer_size = 1
                else:
                    buffer_size += 1
                out.append(buffer_size)
            return out
        else:
            raise ValueError("model=%r" % (model,))

    def cohort_size(self, phones, model='form'):
        """Yield cohort size for each phone

        Parameters
        ----------
        phones : str
            Space-delimited arpabet phones.
        model : str
            Decomposition model to use.
        """
        return (cohort.size(model) for cohort in self.cohorts(phones))

    def entropy(
            self,
            phones: PronunciationOrTextGrid = None,
            model: str = 'form',
            smooth: float = 0,
    ):
        """Sequence of cohort entropy at each phoneme

        Parameters
        ----------
        phones
            Space-delimited arpabet phones.
        model
            Decomposition model to use.
        smooth
            Add this activation value to each word.
        """
        if phones is None:
            return self._cohort.entropy(model, smooth)
        elif isinstance(phones, TextGrid):
            return chain.from_iterable(self.entropy(r.pronunciation, model, smooth) for r in phones.realizations)
        phones = Pronunciation.coerce(phones)
        if phones.is_silence():
            return [0.]
        return [cohort.entropy(model, smooth) for cohort in self.cohorts(phones)]

    def find_nonwords(self, grid: TextGrid) -> fmtxt.Table:
        "Find pronunciations that are missing from the lexicon"
        nonwords = fmtxt.Table('lll')
        nonwords.cells('Time', 'Word', 'Pronunciation')
        nonwords.midrule()
        for r in grid.realizations:
            pronunciation = Pronunciation.coerce(r.phones)
            if pronunciation.is_silence():
                continue
            cohort = self.cohort(pronunciation)
            if cohort.size('form') > 0:
                continue
            nonwords.cells(f'{r.times[0]:.3f}', r.graphs, pronunciation.string())
        return nonwords

    def morpheme_onset(self, phones, start=0):
        """1 for phonemes that constitute a morpheme onset"""
        i = start - 1
        for cohort in self.cohorts(phones):
            if cohort.morpheme_i > i:
                i += 1
                yield 1
            else:
                yield 0

    def surprisal(
            self,
            phones: PronunciationOrTextGrid,
            smooth: float = 0,
            allow_nonwords: bool = False,
    ):
        """Phoneme surprisal for each phone

        Parameters
        ----------
        phones
            Space-delimited arpabet phones.
        smooth
            Add this activation value to each word.
        allow_nonwords
            When the cohort becomes empty, set surprisal to 0 (default is to
            raise a ``RuntimeError``).
        """
        if isinstance(phones, TextGrid):
            return chain.from_iterable(self.surprisal(r.pronunciation, smooth, allow_nonwords) for r in phones.realizations)
        phones = Pronunciation.coerce(phones)
        if phones.is_silence():
            return [0.]
        activations = [cohort.activation() + len(cohort.words) * smooth for cohort in self.cohorts(phones)]
        if all(activations):
            n_phones = len(activations)
            pad = 0
        elif allow_nonwords:
            n_phones = activations.index(0)
            pad = len(activations) - n_phones
        else:
            raise RuntimeError(f"Surprisal is not defined for nonwords: {phones.string()!r}")
        total_activation = sum(word.activation for word in self.words) + len(self.words) * smooth
        activations.insert(0, total_activation)
        surprisal = [-log(activations[i+1] / activations[i], 2) for i in range(n_phones)]
        if pad:
            surprisal.extend([0] * pad)
        return surprisal

    def word_surprisal(self, graphs):
        word_activation = sum(word.activation for word in self.lookup(graphs))
        if word_activation == 0:
            raise ValueError(f'{graphs=}: word with activation 0, surprisal undefined')
        total_activation = sum(word.activation for word in self.words)
        return -log(word_activation / total_activation, 2)

    def phoneme_entropy(self, phones: PronunciationOrTextGrid, smooth: float = 0):
        """Entropy for next phoneme given the cohort"""
        if isinstance(phones, TextGrid):
            return chain.from_iterable(self.phoneme_entropy(r.pronunciation, smooth) for r in phones.realizations)
        phones = Pronunciation.coerce(phones)
        if phones.is_silence():
            return [0.]
        return [cohort.phoneme_entropy(smooth) for cohort in self.cohorts(phones)]


def generate_lexicon(
        pronunciations: Dict[str, Union[str, Collection[str]]],
        activation: Dict[str, float] = None,  # {word: activation}
        segmentations: Dict[str, Optional[str]] = None,  # {word: segmentation}; '+'-delimited
        default_activation: Union[float, Literal['drop']] = None,
):
    """Create a lexicon

    Parameters
    ----------
    pronunciations
        Lexicon entries in the format
        ``{word: pronunciation}`` or
        ``{word: (pronunciation_1, pronunciation_2, ...)}``;
        pronunciations are strings of space-delimited phonemes.
    activation
        Mapping each ``word`` to an activation value (e.g., word frequency).
        If omitted, all words are given activation 1 (or ``default_activation``).
    segmentations
        Optional mapping of words to morphological segmentations.
    default_activation
        What to do about pronunciations that are missing from ``activations``.
        By default, a :exc:`KeyError` is raised for missing words.
        Set to a number to supply a default activation;
        Set to ```drop```` to ignore the word (don't add it to the lexicon).

    Examples
    --------
    Standard lexicon based on the intersection of CMUPD and SUBTLEX::

        import trftools
        import cohort.lexicon

        cmupd = trftools.dictionaries.read_cmupd(strip_stress=True)
        subtlex = trftools.dictionaries.read_subtlex()
        frequencies = {word: entry['FREQcount'] for word, entry in subtlex.items()}
        pronunciations = {word: pronunciations for word, pronunciations in cmupd.items() if word in frequencies}
        lexicon = cohort.lexicon.generate_lexicon(pronunciations, frequencies)

    This `lexicon` can be pickled.
    """
    if activation is None:
        if default_activation is None:
            default_activation = 1
        elif isinstance(default_activation, str):
            raise ValueError(f"{default_activation=} with {activation=}")
        activation = defaultdict(lambda: default_activation)

    words = []
    segmentation = None
    for word, word_pronunciations in pronunciations.items():
        try:
            activation_ = activation[word]
        except KeyError:
            if isinstance(default_activation, str):
                if default_activation == 'drop':
                    continue
                else:
                    raise ValueError(f"{default_activation=}")
            elif default_activation:
                activation_ = default_activation
            else:
                raise KeyError(f"{word} missing from activation")
        if segmentations is not None:
            segmentation = segmentations[word]
            if segmentation is not None:
                segmentation = tuple(segmentation.split('+'))
        if isinstance(word_pronunciations, str):
            word_pronunciations = (word_pronunciations,)
        word_pronunciations = tuple([Pronunciation(tuple(p.split(' '))) for p in word_pronunciations])
        words.append(Word(word, word_pronunciations, segmentation, activation_))
    return Lexicon(words)


PathArg = Union[Path, str]
PronunciationArg = Union[str, Sequence[str], Pronunciation]
PronunciationOrTextGrid = Union[Pronunciation, TextGrid]
