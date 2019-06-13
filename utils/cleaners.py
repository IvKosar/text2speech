import re
from unidecode import unidecode
from utils.numbers_utils import normalize_numbers


# List of (regular expression, replacement) pairs for abbreviations:
_abbreviations = [(re.compile('\\b%s\\.' % x[0], re.IGNORECASE), x[1]) for x in [
    ('mrs', 'misses'),
    ('mr', 'mister'),
    ('dr', 'doctor'),
    ('st', 'saint'),
    ('co', 'company'),
    ('jr', 'junior'),
    ('maj', 'major'),
    ('gen', 'general'),
    ('drs', 'doctors'),
    ('rev', 'reverend'),
    ('lt', 'lieutenant'),
    ('hon', 'honorable'),
    ('sgt', 'sergeant'),
    ('capt', 'captain'),
    ('esq', 'esquire'),
    ('ltd', 'limited'),
    ('col', 'colonel'),
    ('ft', 'fort'),
]]


def english_cleaners(text):
    """Pipeline for English text, including number and abbreviation expansion."""
    text = unidecode(text)
    text = text.lower()
    text = normalize_numbers(text)
    for regex, replacement in _abbreviations:
        text = re.sub(regex, replacement, text)
    text = re.sub(re.compile(r'\s+'), ' ', text)
    return text
