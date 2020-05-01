"""
Extends Scikit-learns PCA functionality to transform Spark Dataframes
as well.

Author: Caio Martins

Since: 2020-04
"""
from pyspark.sql import SparkSession
from pyspark.sql.functions import expr
from pyspark.sql.dataframe import DataFrame
import pandas as pd
import matrix_multiplier as mp
from sklearn.decomposition import PCA

class BasePreprocessor:
    """
    Simple Base class that checks whether input is a Pandas or Spark DataFrame.
    """
    def fit(self, X, y=None):
        if isinstance(X, pd.core.frame.DataFrame):
            self._fit_pandas_(X, y)
        else:
            raise TypeError('Input dataframe is not a Pandas dataframe.')
        
    def transform(self, X):
        if isinstance(X, pd.core.frame.DataFrame):
            return self._transform_pandas_(X)
        elif isinstance(X, DataFrame):
            return self._transform_spark_(X)
        else:
            raise TypeError('Input dataframe is neither a Pandas or Spark dataframe.')

    def _fit_pandas_(self):
        raise RuntimeError('Method _fit_pandas_ not implemented yet.')
    
    def _transform_pandas_(self):
        raise RuntimeError('Method _transform_pandas_ not implemented yet.')
    
    def _transform_spark_(self):
        raise RuntimeError('Method _transform_spark_ not implemented yet.')

class PCA_enhanced(BasePreprocessor, PCA):
    """
    Enhanced PCA. Also transforms Spark DataFrames.
    Doing stardization first is a nice idea!
    """

    def _fit_pandas_(self, X, y=None):
        self.df_cols = list(X.columns)
         #can't use super because I also inherit form BasePreprocessor which
         #also implements fit, so I have to be explicit here
        PCA.fit(self, X, y)
        
    def _transform_pandas_(self, X):
        return PCA.transform(self, X[self.df_cols])
        
    def _transform_spark_(self, X):
        spark = SparkSession.builder.getOrCreate() #SparkSession object is a Singleton
        #It is necessary to subtract the mean from the columns to get
        #the same output from scikit-learns PCA
        for col, col_mean in zip(self.df_cols, self.mean_):
            X = X.withColumn(col, expr('{} - {}'.format(col, col_mean)))
        return spark.sql(str(mp.MatrixMultiplier(X,
                                                 self.components_.T,
                                                 df_cols=self.df_cols,
                                                 prefix='principal_component',
                                                 table_name='pca_table')))