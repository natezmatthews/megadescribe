import pandas as pd
from IPython.display import display
from dateutil.parser import parse
import seaborn as sns

class ColumnClassifier():
    def __init__(self,df):
        assert type(df) == pd.core.frame.DataFrame, "Argument was not a pandas DataFrame"
        assert len(df.columns) > 0, "There are no columns"
        
        dates = ['<M8', 'datetime64']
        numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
        
        self.__objects = [c for c in df.select_dtypes(include=['object']).columns]
        self.__datevals = [c for c in df.select_dtypes(include=dates).columns]
        self.__numvals = [c for c in df.select_dtypes(include=numerics).columns]
        self.__index = df.columns[0] # The index method makes a smarter guess later
        self.__allnulls = []
        self.__anynulls = []
        self.__unique = []
        self.__idsuffix = []
        self.__ynsuffix = []
        for c in df.columns:
            if df[c].isnull().all():
                self.__allnulls += [c]
            if df[c].isnull().any():
                self.__anynulls += [c]
            if df[c].count() == df[c].nunique():
                self.__unique += [c]
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
        except (ValueError, TypeError):
            return False

    def __cmb__(self,include=[],exclude=[]):
        exclude = set(exclude)
        toret = []
        for c in include:
            if c not in exclude:
                toret += [c]
                exclude.add(c)
        return toret
    
    def index(self):
        uniquenevernull = self.__cmb__(include = self.__unique,
                                       exclude = self.__anynulls)
        if uniquenevernull:
            self.__index = uniquenevernull
        return self.__index
    
    def allnulls(self):
        return self.__allnulls
    
    def ids(self):
        return self.__idsuffix
        
    def dates(self):
        return self.__datevals
        
    def categoricals(self):
        return self.__cmb__(include = self.__ynsuffix + self.__objects,
                            exclude = [self.__index] + self.__datevals)
    def numerics(self):
        return self.__cmb__(include = self.__numvals,
                            exclude = self.__idsuffix + self.categoricals() + self.__allnulls)
    
    def yns(self):
        return self.__ynsuffix
    
    def signal(self):
        return self.__cmb__(include = self.numerics() + self.categoricals() + self.dates(),
                            exclude = [self.__index] + self.__idsuffix + self.__allnulls)
    
    def getdict(self):
        toret = {}
        for f in [self.index,self.allnulls,self.ids,self.dates,self.categoricals,self.numerics,self.yns,self.signal]:
            toret[f.__name__] = f()
        return toret

# Show me the top N values in each category:
def CategoricalVars(df,colclass,topN=5):
    def displayprint(x):
        try:
            get_ipython # Display if in a Notebook, print otherwise
            display(x)
        except:
            print(x)

    nullcols = []
    for x in colclass['categoricals']:
        collen = len(df[x])
        nmnull = df[x].isnull().sum()
        todisp = pd.value_counts(df[x].values).iloc[:topN] / collen
        if not todisp.empty:
            displayprint(pd.DataFrame(todisp.rename(str(x)).map(lambda x: "{0:.4f} %".format(x * 100))))
            print('Top {} represent {:.1%} of rows.'.format(topN,todisp.sum()))
            if nmnull > 0:
                print('{:.1%} of rows are null\n\n'.format(nmnull/collen))
            else:
                print('No rows are null\n\n')
        else:
            nullcols = nullcols + [str(x)]
    if nullcols:
        print('The following categorical columns are entirely null:\n')
        for x in nullcols:
            print(x)

def ContinuousVars(df,colclass):
    def displayprint(x):
        try:
            get_ipython # Display if in a Notebook, print otherwise
            display(x)
        except:
            print(x)

    def readableNumbers(x):
        toinsert = ''
        if isinstance(x, float):
            decimalplaces = 3
            if x.is_integer():
                decimalplaces = 0
            toinsert = '.' + str(decimalplaces) + 'f'
        return ('{:,' + toinsert + '}').format(x)

    desc = pd.DataFrame()
    for col in colclass['numerics']:
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
        displayprint(desc.applymap(readableNumbers).T[colorder].style.applymap(lambda x: 'text-align:right'))
        droppnulls = df[colclass['numerics']].dropna()
        try:
            get_ipython # The pairplot is catered towards the situation where a Jupyter Notebook is printing matplotlib inline
            if len(droppnulls.columns) < 11:
                sns.pairplot(droppnulls)
            else:
                print('{} variables is too many for a readable pairplot'.format(len(droppnulls.columns)))
        except:
            pass