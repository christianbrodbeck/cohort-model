# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.4
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

from eelbrain import Dataset, Factor, Var
import trftools
from trftools.align import TextGrid
from trftools.align._textgrid import Realization
import cohort.lexicon


# # Setup
# Generate a small TextGrid

realizations = (
  Realization(phones=('IH', 'F'), times=(0.0, 0.05), graphs='if', tstop=0.08),
  Realization(phones=('Y', 'UW'), times=(0.08, 0.15), graphs='you', tstop=0.21),
  Realization(phones=('HH', 'AE', 'P', 'AH', 'N', 'D'), times=(0.21, 0.27, 0.38, 0.45, 0.49, 0.55), graphs='happened', tstop=0.58),
  Realization(phones=('T', 'AH'), times=(0.58, 0.61), graphs='to', tstop=0.66),
  Realization(phones=('F', 'AY', 'N', 'D'), times=(0.66, 0.78, 0.94, 0.99), graphs='find', tstop=1.02),
  Realization(phones=('Y', 'ER', 'S', 'EH', 'L', 'F'), times=(1.02, 1.09, 1.16, 1.26, 1.4, 1.46), graphs='yourself', tstop=1.51),
  Realization(phones=('AA', 'N'), times=(1.51, 1.54), graphs='on', tstop=1.6),
  Realization(phones=('DH', 'AH'), times=(1.6, 1.64), graphs='the', tstop=1.68),
  Realization(phones=('B', 'AE', 'NG', 'K', 'S'), times=(1.68, 1.78, 1.92, 1.96, 2.01), graphs='banks', tstop=2.05),
  Realization(phones=('AH', 'V'), times=(2.05, 2.1), graphs='of', tstop=2.13),
  Realization(phones=('DH', 'IY'), times=(2.13, 2.17), graphs='the', tstop=2.29),
  Realization(phones=('OW', 'HH', 'AY', 'OW'), times=(2.29, 2.34, 2.43, 2.59), graphs='ohio', tstop=2.67),
)
textgrid = TextGrid(realizations)
textgrid.print()

# Generate the lexicon

cmupd = trftools.dictionaries.read_cmupd(strip_stress=True)
subtlex = trftools.dictionaries.read_subtlex()
frequencies = {word: entry['FREQcount'] for word, entry in subtlex.items()}
pronunciations = {word: pronunciations for word, pronunciations in cmupd.items() if word in frequencies}
lexicon = cohort.lexicon.generate_lexicon(pronunciations, frequencies)

lexicon.cohort("S P EY S").table()

# # Test with TextGrid

data = Dataset({
    'time': Var(textgrid.times),
    'phone': Factor(textgrid.phones),
    'surprisal': Var(lexicon.surprisal(textgrid)),
})
data.head()

# # Test preserving original activation
# ## Direct

lexicon.words[2]

with lexicon.preserve_activation():
    lexicon.words[2].activation = 2
    print(lexicon.words[2])
print(lexicon.words[2])

# ## Effect on surprisal

word = lexicon.lookup('YOU')[0]
with lexicon.preserve_activation():
    word.activation = 2e8
    data = Dataset({
        'time': Var(textgrid.times),
        'phone': Factor(textgrid.phones),
        'surprisal': Var(lexicon.surprisal(textgrid)),
    })
data.head(5)

data = Dataset({
    'time': Var(textgrid.times),
    'phone': Factor(textgrid.phones),
    'surprisal': Var(lexicon.surprisal(textgrid)),
})
data.head()

# # TODO: test with N-Gram model
