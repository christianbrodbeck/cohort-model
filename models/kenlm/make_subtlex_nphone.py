"""Training kenlm N-Phone model from SUBTLEX-US corpus"""
from pathlib import Path
import subprocess


subtlex_root = Path('/Users/christian/Data/Corpus/Subtlex US')
corpus_path = subtlex_root / 'Subtlex.US.phone.txt'
arpa_path = subtlex_root / 'Subtlex.US.phone.arpa'
bin_path = subtlex_root / 'Subtlex.US.phone.mmap'

# build
proc1 = subprocess.run(['lmplz', '-o', '5', '-S', '80%', '-T', '/tmp', '--discount_fallback'], stdin=corpus_path.open(), stdout=arpa_path.open('w'))
# then build mmap:
proc2 = subprocess.run(['build_binary', '-s', arpa_path, bin_path])
