import megadescribe as md
import numpy as np
import random
import string
import pandas as pd
from datetime import datetime as dt, timedelta as td
import pytest

def test_typeerror_at_initiation():
    with pytest.raises(TypeError):
        md.ColumnClassifier(0)

def test_column_classifications():
    df = pd.DataFrame()
    def one_to_randint():
        for x in range(random.randint(1,3)):
            yield x+1
    date_columns = ['date_'+str(x) for x in one_to_randint()]
    for col in date_columns:
        df[col] = pd.Series([dt.today() - td(x) for x in np.random.uniform(0,1000,10)])
    
    numeric_columns = ['numeric_'+str(x) for x in one_to_randint()]
    for col in numeric_columns:
        df[col] = pd.Series([x for x in np.random.uniform(0,1000,10)])
    
    categorical_columns = ['categorical_'+str(x) for x in one_to_randint()]
    for col in categorical_columns:
        df[col] = pd.Series([random.choice(string.ascii_lowercase) for _ in range(10)])

    instance = md.ColumnClassifier(df)
    assert set(instance.dates()).issuperset(set(date_columns))
    assert set(instance.numerics()).issuperset(set(numeric_columns))
    assert set(instance.categoricals()).issuperset(set(categorical_columns))
    assert set(instance.dates()).issubset(set(date_columns))
    assert set(instance.numerics()).issubset(set(numeric_columns))
    assert set(instance.categoricals()).issubset(set(categorical_columns))