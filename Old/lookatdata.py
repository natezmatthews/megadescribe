import pandas as pd
import numpy as np
from IPython.display import display, HTML
from dateutil.parser import parse
from datetime import datetime as dt
import seaborn as sns

def megadescribe(df):
    """Quickly see many statistics and pivots of your data"""
    class ColumnClassifier():
        """The rest of megadescribe will need good guesses about what's in each column of the dataframe"""
        def __init__(self,df):
            assert type(df) == pd.core.frame.DataFrame, "Argument was not a pandas DataFrame"
            assert len(df.columns) > 0, "There are no columns"
            
            dates = ['<M8', 'datetime64']
            numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
            
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
                if c[-2:].lower() == 'id':
                    self.__idsuffix += [c]
                if c[-2:].lower() == 'yn':
                    self.__ynsuffix += [c]
        
        def __isdate__(self,string):
            try: 
                parse(string)
                return True
            except: # We'll be conservative here, this is a quick description tool
                return False

        def __cmb__(self,include=[],exclude=[]):
            exclude = set(exclude)
            toret = []
            for c in include:
                if c not in exclude:
                    toret += [c]
                    exclude.add(c)
            return toret

        def dates(self):
            return self.__cmb__(include = self.__datevals) # Using __cmb__ for its deduping, and for consistency
            
        def categoricals(self):
            return self.__cmb__(include = self.__ynsuffix + self.__objects,
                                exclude = self.__datevals)
        def numerics(self):
            return self.__cmb__(include = self.__numvals,
                                exclude = self.__idsuffix + self.categoricals() + self.__allnulls)

    class UnusualRowScore():
        """Give each row a score that sums up how 'unusual' its values are, where a value is
           considered unusual for a column of continuous variables when it has a high percentile,
           and is considered unusual for a column of categorical variables when it is rare."""
        def __init__(self,df,colclass):
            self.df = df
            self.scores = pd.DataFrame(index=df.index)

            for col in colclass.dates():
                    self.dateDeviScore(col)
            for col in colclass.numerics():
                    self.numDeviScore(col)
            for col in colclass.categoricals():
                    self.catDeviScore(col)

        def catDeviScore(self,col):
            # Find the % of the data in each category
            self.scores[col] = self.df[col].values
            dist = pd.value_counts(self.df[col].values,normalize=True)
            if dist.empty:
                return
            # We will give categories with the highest frequency a score of 0, and
            # the categories with the lowest frequency a score of 1
            seen = set() # For deduping in order with this comprehension here:
            dist2 = [x for x in dist.values if not (x in seen or seen.add(x))]
            dist3 = [dist2[0] / x if x != 0 else 0 for x in dist2]
            dist4 = [x - 1 for x in dist3]
            dist5 = [x / dist4[-1] if dist4[-1] != 0 else 0 for x in dist4]
            dist6 = pd.DataFrame(dist2,dist5).reset_index()
            # Mapping series
            mapser = pd.merge(pd.DataFrame(dist).reset_index(),dist6,on=0)
            mapdict = mapser[['index_x','index_y']].set_index('index_x').to_dict()['index_y']
            self.scores[col] = self.scores[col].map(mapdict)
            
        def contDeviScore(self,theseries,col):
            percen = theseries.rank(pct=True)
            self.scores[col] = percen.map(lambda x: 2 * abs(0.5 - x))
        
        def numDeviScore(self,col):
            self.contDeviScore(self.df[col],col)
        
        def __dtseconds__(self,x):
            if not x:
                return None
            try:
                xdt = parse(x)
                return (x - dt.fromtimestamp(0)).total_seconds()
            except:
                return None

        def dateDeviScore(self,col):
            seconds = self.df[col].map(self.__dtseconds__)
            self.contDeviScore(seconds,col)

        def show(self):
            scoresums = self.scores.sum(axis=1).rename("scoresums_onlyawombatwouldnameacolumnthis") # Random in the kid sense https://xkcd.com/1210/
            todisp = self.df.join(scoresums)
            todisp = todisp.sort_values(by='scoresums_onlyawombatwouldnameacolumnthis',ascending=False).drop('scoresums_onlyawombatwouldnameacolumnthis',axis=1)
            if this_is_a_notebook():
                display(todisp.head())
            else:
                print(todisp.head())

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
        assert isinstance(text, str), "Header must be a string"
        textlen = len(text)
        bounds = "+{}+".format((textlen + 2) * "-") 
        textline = "\n+ {} +\n".format(text)
        print(bounds + textline + bounds)

    colclass = ColumnClassifier(df)
    
    # Time to look at categorical variables
    
    header("Categorical Variables")
    nullcols = []
    for col in colclass.categoricals():
        collen = len(df[col])
        nmnull = df[col].isnull().sum()
        todisp = pd.value_counts(df[col].values).iloc[:5] / collen
        if not todisp.empty:
            todispdf = pd.DataFrame(todisp.rename(str(col)).map(lambda x: "{0:.4f} %".format(x * 100)))
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

    if desc.empty:
        print('No continuous variables')
    else:
        colorder = ['count','sum','mean','%null','min','10%','50%','90%','max']
        todispformatted = desc.applymap(readable_numbers).T[colorder].style.applymap(lambda x: 'text-align:right')
        header("Continuous Variables")
        if this_is_a_notebook():
            display(todispformatted)
        else:
            print(todispformatted)
        droppnulls = df[colclass.numerics()].dropna()
        if this_is_a_notebook():
            if len(droppnulls.columns) < 11:
                sns.pairplot(droppnulls)
            else:
                print('Pairplot will not be displayed as there were {} continuous variables'.format(len(droppnulls.columns)))

    # Time to look at unusual rows
    
    header("Rows with high percentile values and/or rare categories")
    unusualrows = UnusualRowScore(df,colclass)
    unusualrows.show()