# Experiments

## Baseline - No data enrichment

### CFG1: Base transforms = CTA clipping, Normalization and Noise
    "PRECISION": 0.35599078341013823,
    "count_valid_PRECISION": 31,
    "RECALL": 0.40194805194805194,
    "count_valid_RECALL": 33,
    "MCC": 0.4440142053028482,
    "count_valid_MCC": 25,
    "DICE": 0.18915802258876024,
    "count_valid_DICE": 39,
    "HD95": 0.7585607835722907,
    "count_valid_HD95": 39,
    "VOLSIM": 0.20516828085302266,
    "count_valid_VOLSIM": 39

### CFG2: Base transforms + Laterality flipping
    "PRECISION": 0.3999178981937602,
    "count_valid_PRECISION": 29,
    "RECALL": 0.405988455988456,
    "count_valid_RECALL": 33,
    "MCC": 0.45530958487671397,
    "count_valid_MCC": 25,
    "DICE": 0.196474570296797,
    "count_valid_DICE": 37,
    "HD95": 0.746414433408201,
    "count_valid_HD95": 37,
    "VOLSIM": 0.2127805939130405,
    "count_valid_VOLSIM": 37

### CFG3: Base transforms + Laterality flipping + Noisy Coordinates
    "PRECISION": 0.3987684729064039,
    "count_valid_PRECISION": 29,
    "RECALL": 0.39841269841269844,
    "count_valid_RECALL": 33,
    "MCC": 0.44756275971995857,
    "count_valid_MCC": 25,
    "DICE": 0.19012748522134976,
    "count_valid_DICE": 37,
    "HD95": 0.7535203720174866,
    "count_valid_HD95": 37,
    "VOLSIM": 0.20604663727895067,
    "count_valid_VOLSIM": 37

### CFG4: Base transforms + Laterality flipping + Noisy Coordinates + Coordiante Dropout
    "PRECISION": 0.3987301587301587,
    "count_valid_PRECISION": 30,
    "RECALL": 0.3968975468975469,
    "count_valid_RECALL": 33,
    "MCC": 0.4543909151688761,
    "count_valid_MCC": 25,
    "DICE": 0.1949283397245144,
    "count_valid_DICE": 38,
    "HD95": 0.7569361261427358,
    "count_valid_HD95": 38,
    "VOLSIM": 0.21055203315691076,
    "count_valid_VOLSIM": 38

## CFG2 + Data Enrichment

### Background samples
    "PRECISION": 0.3993855606758832,
    "count_valid_PRECISION": 31,
    "RECALL": 0.42871572871572877,
    "count_valid_RECALL": 33,
    "MCC": 0.47218251663217525,
    "count_valid_MCC": 26,
    "DICE": 0.21189468156701122,
    "count_valid_DICE": 38,
    "HD95": 0.7277087360675939,
    "count_valid_HD95": 38,
    "VOLSIM": 0.23156495051814488,
    "count_valid_VOLSIM": 38

### S1 in training
    "PRECISION": 0.4517241379310345,
    "count_valid_PRECISION": 29,
    "RECALL": 0.4633477633477634,
    "count_valid_RECALL": 33,
    "MCC": 0.5048641591865631,
    "count_valid_MCC": 26,
    "DICE": 0.23564049168141768,
    "count_valid_DICE": 36,
    "HD95": 0.7015273972131667,
    "count_valid_HD95": 36,
    "VOLSIM": 0.25343359105969193,
    "count_valid_VOLSIM": 36

### S1 in training + synthetic samples
    "PRECISION": 0.42238095238095236,
    "count_valid_PRECISION": 30,
    "RECALL": 0.45901875901875905,
    "count_valid_RECALL": 33,
    "MCC": 0.49199272829187635,
    "count_valid_MCC": 26,
    "DICE": 0.21612233360535982,
    "count_valid_DICE": 37,
    "HD95": 0.7242708761127741,
    "count_valid_HD95": 37,
    "VOLSIM": 0.2334473802926509,
    "count_valid_VOLSIM": 37