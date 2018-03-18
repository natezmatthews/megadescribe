import megadescribe as md
import numpy as np
import random
import string
import pandas as pd
from datetime import datetime as dt, timedelta as td
import pytest
import sys

def test_typeerror_at_initiation():
    with pytest.raises(TypeError):
        md.column_classifier(0)

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

    instance = md.column_classifier(df)
    assert set(instance.dates()).issuperset(set(date_columns))
    assert set(instance.numerics()).issuperset(set(numeric_columns))
    assert set(instance.categoricals()).issuperset(set(categorical_columns))
    assert set(instance.dates()).issubset(set(date_columns))
    assert set(instance.numerics()).issubset(set(numeric_columns))
    assert set(instance.categoricals()).issubset(set(categorical_columns))

############################
# Unusual row finder tests #
############################

def test_categorical_score_same_value():
    def score_for_uniform_values(n):
        randfloat = random.uniform(sys.float_info.min,sys.float_info.max)
        df = pd.DataFrame({'A':([randfloat] * n)})
        instance = md.surface_unusual_rows(df)
        out = instance.categorical_score('A')
        assert out.name == 'A'
        assert isinstance(out.index,pd.RangeIndex)
        assert np.array_equal(out.values,np.array(([0] * n)))

    # We'll try lenth 1, and five randomly sampled lengths between 2 and 100
    score_for_uniform_values(1)
    for n in random.sample(range(2,101), 5):
        score_for_uniform_values(n)