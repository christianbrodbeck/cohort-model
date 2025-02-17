# Author: Christian Brodbeck <christianbrodbeck@nyu.edu>
"""Preprocessing for transcripts"""
import re
from typing import List, Sequence

from trftools.align._textgrid import APOSTROPHE_AFFIXES, TextGrid


APOSTROPHE_WORDS = {"CAN'T", "DON'T", "'EM", "O'CLOCK", "SHAN'T", "WON'T"}
APOSTROPHE_TOKENS = {*APOSTROPHE_AFFIXES, *APOSTROPHE_WORDS}


def split_by_apostrophe(
        words: Sequence[str],
        split: bool = True,
):
    "Assumes that ``word`` is uppercase"
    if not split:
        return [word.strip("'") for word in words]
    out = []
    for word in words:
        if "'" in word and (upper := word.upper()) not in APOSTROPHE_TOKENS:
            if word == "'":
                continue
            if upper.endswith("N'T"):
                out.append(word[:-3])
                out.append(word[-3:])
                continue
            i = word.find("'")
            if i == 0 or i == len(word) - 1:
                out.append(word.replace("'", ''))
            else:
                if upper[i:] in APOSTROPHE_AFFIXES:
                    out.append(word[:i])
                    out.append(word[i:])
                else:
                    out.extend(word.split("'"))  # e.g., "o'clock"
        else:
            out.append(word)
    return out


def tokenize_transcript(
        transcript: str,
        stops: bool = False,
        filters: str = None,
        split_apostrophe: bool = True,
) -> List[str]:
    """Tokens are alphanumeric

    Parameters
    ----------
    transcript
        Transcript to tokenize.
    stops
        Include stops (``.?!;``) as ``.`` tokens.
    filters
        Characters to ignore (treat as white-space).
    split_apostrophe
        Split words at apostrophes (barring a few exceptions).
    """
    # possible apostrophe
    transcript = transcript.replace("´", "'")
    transcript = transcript.replace("’", "'")
    if filters is None:
        filters = '!"#$%&()*+,.-/:;<=>?@[\\]^_“”’‘`{|}~\t\n'
    if stops:
        transcript = re.sub('[.?!;]', ' .', transcript)
        if '.' in filters:
            filters = filters.replace('.', '')
    # based on keras_preprocessing.text.text_to_word_sequence
    translate_dict = {c: ' ' for c in filters}
    translate_map = str.maketrans(translate_dict)
    transcript = transcript.translate(translate_map)
    if stops:  # merge successive periods
        transcript = re.sub(r'(\.\s*)+', '. ', transcript)
    tokens = [i for i in transcript.split(' ') if i]
    tokens = split_by_apostrophe(tokens, split_apostrophe)
    return tokens


def check_transcript(
        grid: TextGrid,
        transcript: str,
) -> bool:
    """Verify alignment between TextGrid and transcript"""
    tokens = tokenize_transcript(transcript.upper(), stops=True)
    token_i = -1
    history = []
    for i, realization in enumerate(grid.realizations):
        if realization.is_silence():
            history.append((' ', ''))
            continue
        token_i += 1
        if tokens[token_i] == '.':
            history.append(('', '.'))
            token_i += 1
        history.append((realization.graphs, tokens[token_i]))
        if realization.graphs != tokens[token_i]:
            print(f'Mismatch at realization {i}:')
            print('Grid       Transcript')
            print('---------------------')
            for a, b in history[-10:]:
                print(a.ljust(10), b)
            return False
    return True
