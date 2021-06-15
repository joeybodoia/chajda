import csv
import pytest

# the sys import is needed so that we can import from the current project
import sys
sys.path.append('.')
from chajda.tsvector import load_all_langs, lemmatize
from chajda.tsquery.augments import create_annoy_index_if_needed, load_fasttext_model, augments_fasttext
import fasttext
import fasttext.util

# load the input lang/text pairs
inputs = []
with open('tests/input.csv', 'rt', encoding='utf-8', newline='\n') as f:
    inputs = list(csv.DictReader(f, dialect='excel', strict=True))

# filter the input languages to
test_langs = None  # ['en','ko','ja','zh']
if test_langs is not None:
    inputs = [input for input in inputs if input['lang'] in test_langs]

# pre-loading all languages ensures that the benchmark times accurately reflect
# the performance of the model's execution time, and not load time
langs = [input['lang'] for input in inputs]
load_all_langs(langs)

# pre-loading for augments_fasttext
create_annoy_index_if_needed('en', 'king', 5)
load_fasttext_model('en', 'king', 5)
################################################################################
# test cases
################################################################################


@pytest.mark.parametrize('test', inputs, ids=[input['lang'] for input in inputs])
def test__lemmatize(test, benchmark):
    benchmark(lemmatize, test['lang'], test['text'])

def test__augments_fasttext_annoy(benchmark):
    result = benchmark(augments_fasttext, 'en', 'king')
    assert result == ['kingthe', 'kingly']

def test__augments_fasttext_fasttext(benchmark):
    result = benchmark(augments_fasttext, 'en', 'king', annoy=False)
    assert result == ['queen', 'kingthe']
