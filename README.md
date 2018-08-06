# PyMC4

Pre-release development of high-level probabilistic programming interface for TensorFlow.  Please contribute or participate [on github](https://github.com/pymc-devs/pymc4).

## Installation Instructions
 - Install using `pip`
 ``` bash
 pip install --user git+https://github.com/pymc-devs/pymc4.git@functional#egg=pymc4
 ```
 
## Simple Example
### Import pymc4 and Edward2
``` python
import pymc4 as pm
from tensorflow_probability import edward2 as ed
```
### Model Initialization
``` python
model = pm.Model()
```

### Model Definition
The model has to be defined in a single function with `@[model-name].define` decorator.
``` python
@model.define
def simple(cfg):
    normal = ed.Normal(loc=0., scale=1., name='normal')
```

### Sampling
``` python
trace = pm.sample(model)
```

### Visualize the trace using arviz
``` python
# See https://github.com/arviz-devs/arviz
# pip install git+git://github.com/arviz-devs/arviz.git
import arviz as az

posterior_data = az.convert_to_xarray(trace, chains=1)
az.posteriorplot(posterior_data, figsize=(8, 4), textsize=15, round_to=2);
```


Here is a [blog post](https://sharanry.github.io/post/eight-schools-model/) showcasing the differences between PyMC3 and PyMC4 using the Eight Schools model.
 
## Contributors

For a full list of code contributors based on code checkin activity, see the [GitHub contributor page](https://github.com/pymc-devs/pymc4/graphs/contributors).

As PyMC4 builds upon TensorFlow, particularly the TensorFlow Probability and Edward2 modules, its design is heavily influenced by innovations introduced in these packages. In particular, early development was partially derived from a [prototype](https://github.com/tensorflow/probability/blob/9c2a4c8bbeddebded2b998027ec7111dcdfd9070/discussion/higher_level_modeling_api_demo.ipynb) written by Josh Safyan.

## License

[Apache License, Version 2.0](https://github.com/pymc-devs/pymc4/blob/master/LICENSE)
