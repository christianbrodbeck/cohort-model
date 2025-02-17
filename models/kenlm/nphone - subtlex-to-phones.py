# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.14.0
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# # Setup
# # ! Originally developed in `cssl/appleseed` !

# +
from collections import Counter, defaultdict
from pathlib import Path
import random
import re
import string

from num2words import num2words
import cohort.models
import trftools


subtlex_path = Path('~').expanduser() / 'Data' / 'Corpus' / 'Subtlex US'
path = subtlex_path / 'Subtlex.US.txt'
dst_path = subtlex_path / 'subtlex.US.phone.txt'

# +
ENGLISH_DICT_FILE = '/Users/christian/Documents/MFA/pretrained_models/dictionary/english.dict'

cmupd = trftools.dictionaries.read_cmupd(True)
english = trftools.dictionaries.read_dict(ENGLISH_DICT_FILE, True)
dic = trftools.dictionaries.combine_dicts([cmupd, english])
trftools.dictionaries.fix_apostrophe_pronounciations(dic)
# for random choice:
dic = {key: tuple(values) for key, values in dic.items()}
# -

dic[','] = (',',)
dic["FUCKIN"] = ('F AH K IH N',)
dic['MM'] = dic['MMM'] = dic['M']

text = path.read_text('latin-1')

print(text[:1000])

string.punctuation

# +
TO_SILENCE = ',.?!:;'
REPLACE = {
    '&amp': '',  # only one occurrence
    '\x92': "'",
    '´': "'",
    ':00': '',
    # common mis-spellings
    'lndia': 'India',
    'lsland': 'Island',
    'lnspector': 'Inspector', 
    'ltal': 'Ital',
    'lnternet': 'Internet',
    "I'Il": "I'll",
    'WelI': 'Well',
    'welI': 'well',
    'FBl': 'FBI',
    " 't": "'t"
}
text_pp = text
for src, dst in REPLACE.items():
    text_pp = text_pp.replace(src, dst)

# strip punctuation
punc = set(string.punctuation).difference(TO_SILENCE)
punc.remove("'")
punc.update('§¶')
trans_dic = {c: ' ' for c in punc}
trans_dic.update({c: '. ' for c in TO_SILENCE})
trans = str.maketrans(trans_dic)
# -

def num2word_sub(match):
    return num2words(match.group(0)).translate(trans)


# +
all_missing = Counter()
examples = defaultdict(list)
excluded = 0
lines = text_pp.splitlines()
p_lines = []
for i, line in enumerate(lines):
    line_pp = line
    line_pp = re.sub(r'\d+', num2word_sub, line_pp)
    line_pp = line_pp.translate(trans)#.replace(" ' ", ' ')
    line_pp = line_pp.rstrip('. ')
    line_pp = line_pp.upper()
    words = cohort.models.tokenize_transcript(line_pp, stops=True)
    words = [',' if word == '.' else word for word in words]
    missing = [word for word in words if word not in dic]
    if missing:
        excluded += 1
#         print(f"{i}: {line} ({missing})")
        for m in missing:
            all_missing[m] += 1
            examples[m].append(line)
    else:
        p_lines.append(' '.join([random.choice(dic[word]) for word in words]))

p_lines.append('')
# -

print(f"{excluded} of {len(lines)} ({excluded / len(lines):.2%})")
pairs = list(all_missing.items())
pairs.sort(key=lambda pair: pair[1], reverse=True)
print(',\t\t'.join("%s: %s" % p for p in pairs[:100]))

p_lines[:10]

out_text = '\n'.join(p_lines)

# check that it is just phonemes
print(', '.join(set(out_text.split())))

dst_path.write_text(out_text)
