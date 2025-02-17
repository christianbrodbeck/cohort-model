# Author: Christian Brodbeck <christianbrodbeck@nyu.edu>
"""Preprocess SUBTLEX-US for lexical N-gram model"""
from pathlib import Path
import string

from cohort.models._transcript import tokenize_transcript


subtlex_root = Path('/Users/christian/Data/Corpus/Subtlex US')
subtlex_path = Path('/Users/christian/Data/Corpus/Subtlex US/Subtlex.US.txt')
corpus_path = subtlex_path.with_suffix('.tokenized.txt')


# Weird characters
APOSTROPHE = ['\x91', '\x92', '\x9d']
SPACE = ['\xa0', '\x80', '\x93', '\x94', '\x96', '\x97', '\x99', '\x9c', '¤', '¿']
FILTERS = '!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n' + ''.join(SPACE)
VALID_CHARS = set(string.ascii_lowercase + string.digits + "' ")
ACCENTED = {
    'á': 'a',
    'é': 'e',
    'ñ': 'n',
    'ü': 'u',
}

if __name__ == '__main__':
    text = subtlex_path.read_text('latin1')
    # Basic substitutions
    text = text.lower()
    for c in APOSTROPHE:
        text = text.replace(c, "'")
    for s, t in ACCENTED.items():
        text = text.replace(s, t)
    # Tokenize lines
    skipped = good = 0
    tokenized_lines = []
    for line in text.splitlines():
        tokens = tokenize_transcript(line, filters=FILTERS)
        tokenized = ' '.join(tokens)
        if bad := set(tokenized).difference(VALID_CHARS):
            print(f"{line} {bad}")
            skipped += 1
            continue
        tokenized_lines.append(tokenized)
        good += 1
    print(f"Used {good/(good+skipped):.1%}: {good=}, {skipped=}")
    corpus_path.write_text('\n'.join(tokenized_lines))
