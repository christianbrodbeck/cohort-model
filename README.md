# Installing

Create a `conda` environment with all the dependencies (from the directory in which this `README` file is located), then activate the environment and install the `cohort` module:

```bash
$ conda env create --file=environment.yml
$ conda activate cohort
$ pip install .
```

# Usage

Generate a lexicon based on SUBTLEX-US and the CMUPD:

```python
import trftools
import cohort.lexicon

cmupd = trftools.dictionaries.read_cmupd(strip_stress=True)
subtlex = trftools.dictionaries.read_subtlex()
frequencies = {word: entry['FREQcount'] for word, entry in subtlex.items()}
pronunciations = {word: pronunciations for word, pronunciations in cmupd.items() if word in frequencies}
lexicon = cohort.lexicon.generate_lexicon(pronunciations, frequencies)
```

Then use the lexicon to show a cohort or compute phoneme-by-phoneme surprisal:

```python
>>> print(lexicon.cohort('S P EY S'))
Graphs       Pronunciation         Activation
---------------------------------------------
SPACE        S P EY S                  3369.0
SPACECRAFT   S P EY S K R AE F T        115.0
SPACED       S P EY S T                  43.0
SPACEPORT    S P EY S P AO R T            8.0
SPACER       S P EY S ER                  5.0
SPACERS      S P EY S ER Z                3.0
SPACES       S P EY S AH Z              137.0
             S P EY S IH Z                   
SPACESHIP    S P EY S SH IH P           188.0
SPACESHIPS   S P EY S SH IH P S          23.0
SPACESUIT    S P EY S UW T                6.0
SPACESUITS   S P EY S UW T S              7.0
SPACEY       S P EY S IY                 26.0
SPACING      S P EY S IH NG              28.0
---------------------------------------------
13                                     3958.0
---------------------------------------------
Cohort S P EY S, 13 words, 14 pronunciations

>>> lexicon.surprisal('S P EY S P AO R T')
[4.350741654733829,
 4.654226146382313,
 4.206014137830506,
 0.38706600494499555,
 8.950555897047511,
 -0.0,
 -0.0,
 -0.0]
```

Calculate surprisal for all phonemes in a TextGrid:

```python
grid = trftools.align.TextGrid.from_file(path_to_textgrid)
grid = grid.strip_stress()
surprisal = lexicon.surprisal(grid)
```

Generate an [Eelbrain](https://eelbrain.readthedocs.io/) `Dataset` with all phonemes:

```python
import eelbrain

data = eelbrain.Dataset({
    'phone': grid.phones, 
    'surprisal': lexicon.surprisal(grid),
    'cohort_entropy': lexicon.entropy(grid),
})
```

Or a [pandas](https://pandas.pydata.org/) `DataFrame`:

```python
import pandas

data = pandas.DataFrame({
    'phone': grid.phones, 
    'surprisal': lexicon.surprisal(grid),
    'cohort_entropy': lexicon.entropy(grid),
})
```

For details see local documentation.
