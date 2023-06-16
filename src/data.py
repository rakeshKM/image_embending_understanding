# Import libraries and classes required for this example:
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler 
import pandas as pd 

class Dataset:
    def __init__(self,datapath):
        self.train = []
        self.test = []
        self.scaler = []
        self.load(datapath)
        self.preprocess()
        
    def load(self,datapath):
        # Convert dataset to a pandas dataframe:
        dataset = pd.read_csv(datapath, header=0, index_col=0) 

        # Assign values to the X and y variables:
        X = dataset.iloc[:, :-1].values
        y = dataset.iloc[:, 4].values 

        # Split dataset into random train and test subsets:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20) 
        
        self.train = {'x': X_train,
                      'y': y_train}
        self.test = {'x': X_test,
                     'y': y_test}

    def preprocess(self):
        # Standardize features by removing mean and scaling to unit variance:
        self.scaler = StandardScaler()
        self.train['x'] = self.scaler.fit_transform(self.train['x'])
        self.test['x'] = self.scaler.transform(self.test['x'])
        
    def describe(self):
        print(f"Train shape X: {self.train['x'].shape}")
        print(f"Train shape y: {self.train['y'].shape}")
        print(f"Test  shape X: {self.test['x'].shape}")
        print(f"Test  shape y: {self.test['y'].shape}")