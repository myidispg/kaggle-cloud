DATA_DIR = 'data'

LABEL2IDX = {'Fish': 0,
             'Flower': 1,
             'Gravel': 2,
             'Sugar': 3,}

IDX2LABELS = dict([(value, key) for (key, value) in LABEL2IDX.items()])

# Whether to shuffle the dataset or not
SHUFFLE_DATASET = True
# Validation dataset size
VALIDATION_SPLIT = 0.01

IMAGE_SIZE_ORIG = (1400, 2100) # Height x Width
IMAGE_SIZE_SMALL = (350, 525) # Height x Width