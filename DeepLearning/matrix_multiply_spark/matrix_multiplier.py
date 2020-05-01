"""
Implementation of matrix multiplication between Spark DataFrame and
Numpy Matrix. Output is SQL code. Goal is to be able to use this to implement
the feedforward pass with Spark to Score models (NNs). Or implement 
preprocessing schemes which are linear transformations, such as PCA.

Author: Caio Martins

Since: 2020-04
"""
import numpy as np

class MatrixMultiplier:

    def __init__(self,
                 df,
                 matrix,
                 bias=None,
                 activation=None,
                 df_cols=None,
                 table_name=None,
                 prefix='col'):
        self.df = df
        self.matrix = matrix
        self.D, self.K = matrix.shape
        self.df_cols = df.columns
        if df_cols != None:
            self.df_cols = df_cols

        if self.D != len(self.df_cols):
            raise RuntimeError('Inner dimensions do not match (spark_df = {}, np_matrix = {})'\
                                .format(len(self.df_cols),
                                        self.D))
 

        self.bias = np.zeros(self.K)
        if isinstance(bias,list) or isinstance(bias,np.ndarray):
            if len(bias) == self.K:
                self.bias = bias
            else:
              raise RuntimeError('Bias and output dimensions do not match (bias = {}, output = {})'\
                                  .format(len(self.bias),
                                          self.K))
              
        self.activation = activation
        self.table_name = 'table'
        if table_name != None:
            self.table_name = table_name

        self.prefix = prefix

    def _multiply(self):
        """
        Matrix multiplication.
        """
        
        new_cols = list(set(self.df.columns) - set(self.df_cols))
        for i in range(self.K):
            new_col = []
            for j in range(self.D):
                new_col.append('{}*{}'.format(self.matrix[j,i],self.df_cols[j]))
            if self.activation == None:
                new_cols.append('({} + {})'\
                                .format(' + '.join(new_col),self.bias[i]) + ' as %s_%s' %(self.prefix,
                                                                                           i + 1) )
            elif self.activation == 'sigmoid':
                new_cols.append('1/(1 + exp(-({} + {})))'\
                                .format(' + '.join(new_col),self.bias[i]) + ' as %s_%s' %(self.prefix,
                                                                                           i + 1) )
            elif self.activation == 'tanh':
                new_cols.append('tanh({} + {})'\
                                .format(' + '.join(new_col),self.bias[i]) + ' as %s_%s' %(self.prefix,
                                                                                           i + 1) )
            elif self.activation == 'relu':
                new_cols.append('case when {cols} + {bias} > 0 then {cols} + {bias} else 0 end'\
                                .format(cols=' + '.join(new_col),bias=self.bias[i]) + 
                                ' as %s_%s' %(self.prefix, i + 1) )
                
        columns = ',\n\t'.join(new_cols)
        self.df.registerTempTable(self.table_name)
        
        query = 'select\n\t{}\nfrom\n\t{}'.format(columns, self.table_name)

        return query

    def __str__(self):
        return self._multiply()

