# NER evaluation

Entity-level scores on held-out data. See evaluate_ner.py for definitions.


## test_synth (200 examples)

| System | Strict P | Strict R | Strict F1 | Lenient F1 |
|---|---|---|---|---|
| base | 0.572 | 0.426 | 0.489 | 0.609 |
| legacy_v1 | 0.000 | 0.000 | 0.000 | 0.046 |
| finetuned | 0.927 | 0.917 | 0.922 | 0.972 |
| regex | 0.925 | 0.561 | 0.699 | 0.719 |
| hybrid | 0.945 | 0.882 | 0.912 | 0.945 |

Per-class strict F1:

| System | DRUG | DOSAGE | FREQUENCY | DURATION | CONDITION | ROUTE |
|---|---|---|---|---|---|---|
| base | 0.751 | 0.636 | 0.050 | 0.474 | 0.088 | 0.044 |
| legacy_v1 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| finetuned | 0.821 | 1.000 | 0.986 | 1.000 | 0.752 | 0.977 |
| regex | 0.934 | 0.949 | 0.000 | 0.000 | 0.897 | 0.000 |
| hybrid | 0.903 | 0.849 | 0.990 | 1.000 | 0.780 | 0.977 |

## gold_test (41 examples)

| System | Strict P | Strict R | Strict F1 | Lenient F1 |
|---|---|---|---|---|
| base | 0.569 | 0.569 | 0.569 | 0.672 |
| legacy_v1 | 0.000 | 0.000 | 0.000 | 0.000 |
| finetuned | 0.967 | 0.994 | 0.980 | 0.980 |
| regex | 0.964 | 0.615 | 0.751 | 0.758 |
| hybrid | 0.949 | 0.966 | 0.957 | 0.974 |

Per-class strict F1:

| System | DRUG | DOSAGE | FREQUENCY | DURATION | CONDITION | ROUTE |
|---|---|---|---|---|---|---|
| base | 0.632 | 0.648 | 0.085 | 0.947 | 0.560 | 0.600 |
| legacy_v1 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| finetuned | 1.000 | 1.000 | 0.956 | 1.000 | 0.966 | 0.923 |
| regex | 0.988 | 0.977 | 0.000 | 0.000 | 0.836 | 0.000 |
| hybrid | 1.000 | 0.907 | 0.956 | 1.000 | 0.964 | 0.923 |
