# megadescribe

Megadescribe, a reference to the describe function of `pandas`, takes a pandas dataframe and outputs pivots of every categorical variable, summary statistics of every continuous variable, and the top n must unusual rows.

## Origin story

I am frequently writing SQL pulling from a complicated database schema. Often it takes several if not tens of iterations on my query before I am satisfied that what I am pulling is no more and no less than the data I am looking for. Megadescribe quickly surfaces things in my output that are unusual, and often points me directly to the next change I need to make to my query.

## Libraries used

```
import pandas as pd
import numpy as np
from IPython.display import display, HTML
from dateutil.parser import parse
from datetime import datetime as dt
```

## Authors

* **Nate Matthews**