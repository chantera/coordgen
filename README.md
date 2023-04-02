# Coordgen

This is an implementation of "[Coordination Generation via Synchronized Text-Infilling](https://aclanthology.org/2022.coling-1.517/)".

## Install

To install this package:

```sh
pip install git+https://github.com/chantera/coordgen.git@main#egg=coordgen
```

## Example

The code below is an example using `CoordinationGenerator`.

```py
from coordgen.models import AutoModelForCoordinationGeneration

model = AutoModelForCoordinationGeneration.from_pretrained("t5-small")

raw = "Gold will retain its gain, he said."
span = (10, 25)  # => `retain its gain`

for text, coord in model.generate([(raw, span)]):
    print(text, coord)
```
