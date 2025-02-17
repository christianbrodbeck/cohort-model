# Author: Christian Brodbeck <christianbrodbeck@nyu.edu>
from itertools import chain

from eelbrain import Dataset, Factor, Var
import numpy
from trftools.align import TextGrid

from ..lexicon._lexicon import Lexicon


def lexical(
        grid: TextGrid,
        lexicon: Lexicon,
        position: bool = False,
        allow_nonwords: bool = False,
):
    """Lexical (cohort) model

    grid
        TextGrid for words.
    lexicon
        Lexicon of all known words.
    position
        Include phoneme position as ``pos`` variable.
    allow_nonwords
        When the cohort becomes empty, set surprisal to 0 (default is to
        raise a ``RuntimeError``).
    """
    ds = Dataset({
        'time': Var(grid.times),
        'phone': Factor(grid.phones),
        'surprisal': Var(lexicon.surprisal(grid, allow_nonwords=allow_nonwords)),
        'cohort_entropy': Var(lexicon.entropy(grid)),
        'phoneme_entropy': Var(lexicon.phoneme_entropy(grid)),
    }, info={'tstop': grid.tstop})
    pos = numpy.array(list(chain.from_iterable(([-1] if r.is_silence() else range(len(r.phones)) for r in grid.realizations))))
    if position:
        ds['pos'] = Var(pos)
    ds['any'] = Var(pos >= 0)
    ds['p0'] = Var(pos == 0)
    ds['p1_'] = Var(pos > 0)
    return ds
