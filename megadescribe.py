import pandas as pd
import numpy as np
from IPython.display import display, HTML
from dateutil.parser import parse
from datetime import datetime as dt

class ColumnClassifier():
    """Classify the columns into dates, categorical variables, 
       and continuous variables, using reasonable guesses"""
    def __init__(self,df):
        if not isinstance(df,pd.DataFrame):
            raise TypeError("Argument was not a pandas DataFrame")

        self._num_columns = len(df.columns)
        
        dates = ['<M8', 'datetime64']
        numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
        def last_two_letters_lower(text):
            # We expect to run into some column names that are not strings, and
            # in that circumstance, rather than error out, the desired behavior
            # is to return None:
            if not isinstance(text,str):
                return None
            chars_to_return = min(2,len(text))
            return c[-chars_to_return:].lower()

        self.__objects = [c for c in df.select_dtypes(include=['object']).columns]
        self.__datevals = [c for c in df.select_dtypes(include=dates).columns]
        self.__numvals = [c for c in df.select_dtypes(include=numerics).columns]
        self.__allnulls = []
        self.__idsuffix = []
        self.__ynsuffix = []
        for c in df.columns:
            if df[c].isnull().all():
                self.__allnulls += [c]
            if df[c].map(self.__isdate__).all():
                self.__datevals += [c]
            if last_two_letters_lower(c) == 'id':
                self.__idsuffix += [c]
            if last_two_letters_lower(c) == 'yn':
                self.__ynsuffix += [c]

    def __len__():
        return self._num_columns
    
    def __isdate__(self,string):
        try: 
            parse(string)
            return True
        except:
            return False

    def __combine(self,include=[],exclude=[]):
        """Combine and remove duplicates from the 'include' items, 
           and remove the 'exclude' items."""
        exclude = set(exclude)
        toret = []
        for c in include:
            if c not in exclude:
                toret += [c]
                exclude.add(c)
        return toret

    def dates(self):
        return self.__combine(include = self.__datevals)
        
    def categoricals(self):
        return self.__combine(include = self.__ynsuffix + self.__objects,
                            exclude = self.__datevals)
    def numerics(self):
        return self.__combine(include = self.__numvals,
                            exclude = self.__idsuffix + self.categoricals() +\
                                      self.__allnulls)

class UnusualRowScore():
    """Give each row a score that sums up how 'unusual' its values are, where
       a value is considered unusual for a column of continuous variables when
       it has a high percentile, and is considered unusual for a column of 
       categorical variables when it is rare."""
    def __init__(self,df,colclass):
        self.df = df
        self.scores = pd.DataFrame(index=df.index)

        for col in colclass.dates():
            self.date_score(col)
        for col in colclass.numerics():
            self.numeric_score(col)
        for col in colclass.categoricals():
            self.categorical_score(col)

    def categorical_score(self,col):
        # Find the % of the data in each category
        self.scores[col] = self.df[col].values
        dist = pd.value_counts(self.df[col].values,normalize=True)
        if dist.empty:
            return
        # We will give categories with the highest frequency a score of 0, and
        # the categories with the lowest frequency a score of 1
        seen = set() # For deduping in order, using this comprehension here:
        dist2 = [x for x in dist.values if not (x in seen or seen.add(x))]
        dist3 = [dist2[0] / x if x != 0 else 0 for x in dist2]
        dist4 = [x - 1 for x in dist3]
        dist5 = [x / dist4[-1] if dist4[-1] != 0 else 0 for x in dist4]
        dist6 = pd.DataFrame(dist2,dist5).reset_index()
        # Mapping series
        mapser = pd.merge(pd.DataFrame(dist).reset_index(),dist6,on=0)
        mapdict = mapser[['index_x','index_y']].set_index('index_x').to_dict()['index_y']
        self.scores[col] = self.scores[col].map(mapdict)
        
    def cont_score(self,theseries,col):
        percen = theseries.rank(pct=True)
        self.scores[col] = percen.map(lambda x: 2 * abs(0.5 - x))
    
    def numeric_score(self,col):
        self.cont_score(self.df[col],col)
    
    def __dtseconds__(self,x):
        if not x:
            return None
        try:
            if isinstance(x,str):
                x = parse(x)
            return (x - dt.fromtimestamp(0)).total_seconds()
        except:
            return None

    def date_score(self,col):
        seconds = self.df[col].map(self.__dtseconds__)
        self.cont_score(seconds,col)

    def show(self,n=5):
        # Sort descending by the sum of the scores 
        sort_index = self.scores.sum(axis=1).sort_values(ascending=False).index
        to_display = self.df.loc[sort_index]
        if this_is_a_notebook():
            display(to_display.head(n))
        else:
            print(to_display.head(n))

def this_is_a_notebook():
    try:
        get_ipython
        return True
    except:
        False

def readable_numbers(x):
    toinsert = ''
    if isinstance(x, float):
        decimalplaces = 3
        if x.is_integer():
            decimalplaces = 0
        toinsert = '.' + str(decimalplaces) + 'f'
    return ('{:,' + toinsert + '}').format(x)

def header(text):
    if not isinstance(text, str):
        raise TypeError("Header must be a string")
    textlen = len(text)
    bounds = "+{}+".format((textlen + 2) * "-") 
    textline = "\n+ {} +\n".format(text)
    print(bounds + textline + bounds)

def megadescribe(df,n=5):
    """Quickly see many statistics and pivots of your data"""
    colclass = ColumnClassifier(df)
    strf = lambda x: "{0:.4f} %".format(x * 100)
    
    # Time to look at categorical variables
    
    if not colclass.categoricals():
        print("No categorical variables.")
    else:
        header("Categorical Variables")
        nullcols = []
        for col in colclass.categoricals():
            collen = len(df[col])
            nmnull = df[col].isnull().sum()
            todisp = pd.value_counts(df[col].values).iloc[:5] / collen
            if not todisp.empty:
                todispdf = pd.DataFrame(todisp.rename(str(col)).map(strf))
                if this_is_a_notebook():
                    display(todispdf)
                else:
                    print(todispdf)
                print('Top {:d} represent {:.1%} of rows.'.format(5,todisp.sum()))
                if nmnull > 0:
                    print('{:.1%} of rows are null\n\n'.format(nmnull/collen))
                else:
                    print('No rows are null\n\n')
            else:
                nullcols = nullcols + [str(col)]
        if nullcols:
            print('The following categorical columns are entirely null:\n')
            for col in nullcols:
                print(col)

    # Time to look at continuous variables
    if not colclass.numerics():
        print("No continous variables.")
    else:
        header("Continuous Variables")
        desc = pd.DataFrame()
        for col in colclass.numerics():
            s = df[col]
            d = {}
            d['count'] = s.count()
            d['sum'] = s.sum()
            d['mean'] = s.mean()
            numnull = s.isnull().sum() 
            d['%null'] = float(numnull) / len(s)
            d['min'] = s.min()
            d['10%'] = s.dropna().quantile(0.1)
            d['50%'] = s.dropna().quantile(0.5)
            d['90%'] = s.dropna().quantile(0.9)
            d['max'] = s.max()
            desc[col] = pd.Series(d)

        colorder = ['count','sum','mean','%null','min','10%','50%','90%','max']
        todisp = desc.applymap(readable_numbers).T[colorder]
        todisp = todisp.style.applymap(lambda x: 'text-align:right')
        if this_is_a_notebook():
            display(todisp)
        else:
            print(todisp)
        droppnulls = df[colclass.numerics()].dropna()

    # Time to look at unusual rows
    if not df.empty:
        header("Rows with high percentile values and/or rare categories")
        unusualrows = UnusualRowScore(df,colclass)
        unusualrows.show(n)