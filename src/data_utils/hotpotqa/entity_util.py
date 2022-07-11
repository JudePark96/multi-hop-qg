import logging
from typing import List

import spacy

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

# https://towardsdatascience.com/explorations-in-named-entity-recognition-and-was-eleanor-roosevelt-right-671271117218
# List of labels of Spacy NER. ('CARDINAL', 'DATE', 'EVENT', 'FAC', 'GPE', 'LANGUAGE', 'LAW', 'LOC', 'MONEY',
# 'NORP', 'ORDINAL', 'ORG', 'PERCENT', 'PERSON', 'PRODUCT', 'QUANTITY', 'TIME', 'WORK_OF_ART')
# We use as below:
# {CARDINAL, DATE, FAC, GPE, LOC, MONEY, QUANTITY, ORDINAL, ORG, PERCENT, PERSON, PRODUCT, QUANTITY, TIME}
TARGET_ENTITY_LABELS = ['CARDINAL', 'DATE', 'FAC', 'GPE', 'LOC', 'MONEY', 'QUANTITY', 'ORDINAL', 'ORG', 'PERCENT',
                        'PERSON', 'PRODUCT', 'QUANTITY', 'TIME']


def extract_entities(nlp: spacy, context: List[str]):





  pass
