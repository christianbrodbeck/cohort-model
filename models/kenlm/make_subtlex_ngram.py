"""Training kenlm from SUBTLEX-US corpus"""
from pathlib import Path
import subprocess


subtlex_root = Path('/Users/christian/Data/Corpus/Subtlex US')
subtlex_path = Path('/Users/christian/Data/Corpus/Subtlex US/Subtlex.US.txt')
corpus_path = subtlex_path.with_suffix('.tokenized.txt')
arpa_path = subtlex_root / 'Subtlex.US.arpa'
bin_path = subtlex_root / 'Subtlex.US.mmap'

# build
proc1 = subprocess.run(['lmplz', '-o', '5', '-S', '80%', '-T', '/tmp'], stdin=corpus_path.open(), stdout=arpa_path.open('w'))
# then build mmap:
proc2 = subprocess.run(['build_binary', '-s', arpa_path, bin_path])
