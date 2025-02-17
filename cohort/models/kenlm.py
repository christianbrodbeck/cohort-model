# Author: Christian Brodbeck <christianbrodbeck@nyu.edu>
from itertools import chain
from math import log
from typing import Any, Optional, Sequence

from eelbrain import Dataset, Factor, Var
import kenlm
import numpy
from trftools.align import TextGrid
from trftools.dictionaries._arpabet import ARPABET

from ..lexicon._lexicon import Lexicon
from ._transcript import tokenize_transcript


class LexicalNGram:

    def __init__(
            self,
            lm: kenlm.Model,
            lower: bool = False,
            upper: bool = False,
    ):
        """KenLM language model wrapper

        Parameters
        ----------
        lm
            KenLM language model.
        lower
            Model is trained all lower-case (transform all input to lower-case).
        upper
            Model is trained all upper-case.
        """
        if lower and upper:
            raise ValueError(f"{lower=} and {upper=} are mutually exclusive")
        self.lm = lm
        self.lower = lower
        self.upper = upper
        self._targets = None
        self._target_lexicon = None

    def update_lexicon(
            self,
            lexicon: Lexicon,
            context: str,
            bos: bool = False
    ) -> None:
        if self.upper:
            context = context.upper()
        elif self.lower:
            context = context.lower()
        context = context.encode()

        if self._targets is None or self._target_lexicon is not lexicon:
            if self.upper:
                self._targets = [word.graphs.upper().encode() for word in lexicon.words]
            elif self.lower:
                self._targets = [word.graphs.lower().encode() for word in lexicon.words]
            else:
                self._targets = [word.graphs.encode() for word in lexicon.words]
            self._target_lexicon = lexicon

        logp_context = self.lm.score(context, bos=bos, eos=False)
        for target, word in zip(self._targets, lexicon.words):
            logp = self.lm.score(b'%s %s' % (context, target), bos=bos, eos=False) - logp_context
            word.activation = 10 ** logp


def lexical_ngram(
        grid: TextGrid,
        transcript: Optional[str],
        lexicon: Lexicon,
        lm: kenlm.Model,
        n: int = 5,
        lower: bool = False,
        upper: bool = False,
) -> Dataset:
    """Lexical N-Gram model

    .. warning::
        This function will temporarily modify the ``lexicon``. This allows
        effective caching of cohort trees. If using parallelization, make
        sure to use a separate deep copy of ``lexicon`` in each thread.

    Parameters
    ----------
    grid
        TextGrid for sequence of phones (``' '`` for silence) and (onset)
        times corresponding to the phones.
    transcript
        Transcript (used to find sentence boundaries at ``.?!;``). Use ``None``
        to skip comparison with a transcript. Make sure that case
        (uppercase/lowercase) matches the ``grid``.
    lexicon
        Lexicon of all known words.
    lm
        KenLM model.
        If either ``lower`` or ``upper`` is set to ``True``, case in the inputs
        (``gird`` and ``transcript``) will be normalized before querying the
        ``lm``. If neither is specified, case is not modified.
    n
        N (order) of the N-gram model (default assumes a 5-gram model).
    lower
        ``lm`` was is trained on all lower-case text.
    upper
        ``lm`` was is trained on all upper-case text.

    See also
    --------
    cohort.models.check_transcript
    """
    if transcript is None:
        tokens = None
    else:
        tokens = tokenize_transcript(transcript, stops=True)
    context_model = LexicalNGram(lm, lower, upper)

    n_context = n - 1
    words = []
    time = []
    surprisal = []
    cohort_entropy = []
    phoneme_entropy = []
    token_i = -1
    with lexicon.preserve_activation():
        context_model.update_lexicon(lexicon, '', bos=True)
        for realization in grid.realizations:
            if realization.is_silence():
                time.append(realization.times[0])
                surprisal.append(0)
                cohort_entropy.append(0)
                phoneme_entropy.append(0)
                continue
            if tokens:
                token_i += 1
                if tokens[token_i] == '.':
                    words = []
                    token_i += 1
                assert realization.graphs == tokens[token_i]
            # language model
            context_model.update_lexicon(lexicon, ' '.join(words[-n_context:]), bos=len(words) < n_context)
            words.append(realization.graphs)
            # predictors
            time.extend(realization.times)
            surprisal.extend(lexicon.surprisal(realization.pronunciation))
            cohort_entropy.extend(lexicon.entropy(realization.pronunciation))
            phoneme_entropy.extend(lexicon.phoneme_entropy(realization.pronunciation))

    ds = Dataset({
        'time': Var(time),
        'phone': Factor(grid.phones),
        'surprisal': Var(surprisal),
        'cohort_entropy': Var(cohort_entropy),
        'phoneme_entropy': Var(phoneme_entropy),
    }, info={'tstop': grid.tstop})
    return ds


def sublexical(
        grid: TextGrid,
        lm: kenlm.Model,
        alphabet: Sequence[str] = None,
        eos_duration: float = 0.400,
        skip_silence: bool = False,
) -> Dataset:
    """Sublexical (N-Phone) language model

    Parameters
    ----------
    grid
        TextGrid for sequence of phones (``' '`` for silence) and (onset)
        times corresponding to the phones.
    lm
        KenLM model; trained with ``' , '`` for pauses.
    alphabet
        List of all permissible phones, including silence (``' '``). Default is
        the Arpabet without stress, which is appropriate for TextGrids that have
        been processed with ``.strip_stress()``.
    eos_duration
        Any silence longer than this will be modeled as ``<eos>`` rather than
        as ``','``.
    skip_silence
        Set entropy and surprisal to 0 during silence.
    """
    log10_2 = log(2, 10)
    if alphabet is None:
        alphabet = ARPABET
    grid_phones = set(chain.from_iterable(r.phones for r in grid.realizations))
    # model silence as ','
    if ' ' not in alphabet:
        alphabet = list(alphabet)
        alphabet.append(' ')
    alphabet_punc = list(alphabet)
    alphabet_punc[alphabet.index(' ')] = ','
    phones = list(grid.phones)
    phones_punc = [',' if p == ' ' else p for p in phones]
    # check alphabet
    if missing := grid_phones.difference(alphabet):
        raise ValueError(f"alphabet is missing the following phones which appear in grid: {missing}")

    times = list(grid.times)
    assert len(phones) == len(times)
    max_index = len(phones) - 1

    rows = []
    if phones[0] == ' ':
        surprisal_i = 0
        last_eos = 0
    else:
        surprisal_i = -lm.score(phones[0], bos=True, eos=False) / log10_2
        last_eos = -1
    for i, phone in enumerate(phones):
        if phone == ' ':
            if (i == max_index) or (times[i+1] - times[i] >= eos_duration):
                last_eos = i

        if i - last_eos < 4:
            i_start, bos = last_eos + 1, True
        else:
            i_start, bos = i - 3, False
        prefix_phones = phones_punc[i_start: i + 1]  # up until and including current phone
        # entropy for next phone
        if prefix_phones:
            prefix = ' '.join(prefix_phones)
            log_p_prefix = lm.score(prefix, bos=bos, eos=False)
            log_ps = [lm.score(f'{prefix} {phone}', bos=bos, eos=False) - log_p_prefix for phone in alphabet_punc]
        else:
            log_ps = [lm.score(phone, bos=bos, eos=False) for phone in alphabet_punc]
        log2_ps = [p / log10_2 for p in log_ps]
        if skip_silence and phone == ' ':
            rows.append([False, 0, 0])
        else:
            if i == last_eos:
                entropy = 0
            else:
                entropy = -sum([2 ** p * p for p in log2_ps])
            rows.append([phone != ' ', entropy, surprisal_i])

        if i < max_index:
            surprisal_i = -log2_ps[alphabet.index(phones[i + 1])]
    ds = Dataset({'phone': Factor(phones), 'time': Var(times)}, info={'tstop': grid.tstop, 'version (sublexical)': 2})
    ds.update(Dataset.from_caselist(['any', 'entropy', 'surprisal'], rows))

    # Add silence and speech onsets
    ds['silence'] = Var(ds['phone'] == ' ')
    ds['onset'] = Var(numpy.diff(numpy.append([1], ds['silence'].x)) == -1)
    return ds
