import pandas as pd

def m_add(safra_ini, qt):
    return (pd.to_datetime(safra_ini, format="%Y%m") + pd.DateOffset(months=qt)).strftime("%Y%m")

class Query(object):
    def __init__(self):
        self.main_table = None
        self._all_tables = []
        self._join_list = []
        self._variables = []

    def _in_var_list(self, var_name):
        for var in self._variables:
            if var_name == var.name:
                return True
        return False

    def _in_table_list(self, table_name):
        for tb in self._all_tables:
            if table_name == tb.table_name:
                return True
        return False

    def _get_table_ref(self, table_name):
        for tb in self._all_tables:
            if table_name == tb.table_name:
                return tb
        raise ValueError("Table not in table list.")

    def _get_table_joins(self):
        return '\n'.join(['''left join\n\t{db}.{tb} {alias}\n\ton {main_alias}.{main_key}={alias}.{key}'''.format(db=tb.database,
                                                                                  tb=tb.table_name,
                                                                                  alias=tb.table_alias,
                                                                                  key=tb.table_key,
                                                                                  main_alias=self.main_table.table_alias,
                                                                                  main_key=self.main_table.table_key) for tb in self._join_list])

    def _get_main_table(self):
        return '\t{} {}'.format(self.main_table.table_name, self.main_table.table_alias)

    def _get_all_variables(self):
        # all_vars = []
        all_vars = ',\n'.join(['\t{}.{} as {}'.format(var.table.table_alias,var.name, var.alias)
                                     if var.flag_alias else
                                     '\t{}.{}'.format(var.table.table_alias,var.name)
                                     for var in self._variables])
        # all_vars = ',\n'.join(all_vars)
        return all_vars

    def __str__(self):

        query = "SELECT\n{}\nFROM\n{}\n{}".format(self._get_all_variables(),
                                                  self._get_main_table(),
                                                  self._get_table_joins())

        return query

    @staticmethod
    def create():
        return QueryBuilder(Query())

class QueryBuilder(object):
    """docstring for QueryBuilder."""

    def __init__(self, query=Query()):
        self.query = query

    @property
    def variable(self):
        return QueryVariableBuilder(self.query)

    @property
    def left_join(self):
        return QueryLeftJoinBuilder(self.query)

    @property
    def main_table(self):
        return QueryMainTableBuilder(self.query)

    def build(self):
        return self.query

class QueryVariableBuilder(QueryBuilder):
    def __init__(self, query):
        super().__init__(query)

    def add(self, table_name, var_name, var_alias='', flag_var_alias=False):
        if not self.query._in_var_list(var_name):
            tb = self.query._get_table_ref(table_name=table_name)
            tb.add_var(var_name)
            self.query._variables.append(tb.variables[-1])
        return self

class QueryLeftJoinBuilder(QueryBuilder):
    def __init__(self, query):
        super().__init__(query)

    def add(self, database, table_name, table_alias, table_safra, table_key='nr_cpf_cnpj'):
        if not self.query._in_table_list(table_name=table_name):
            self.query._join_list.append(Table(database,
                                        table_name,
                                        table_alias,
                                        table_safra,
                                        table_key)
                                        )
            self.query._all_tables.append(self.query._join_list[-1])
        return self


class QueryMainTableBuilder(QueryBuilder):
    """docstring for QueryMainTableBuilder."""
    def __init__(self, query):
        super().__init__(query)

    def add(self, database, table_name, table_alias, table_safra, table_key='nr_cpf_cnpj'):
        self.query.main_table = Table(database,
                                table_name,
                                table_alias,
                                table_safra,
                                table_key)
        self.query._all_tables.append(self.query.main_table)
        return self

class Variable:
    """docstring for Variables."""

    def __init__(self, table, name, alias='', flag_alias=False):
        self.name = name
        self.table = table
        self.alias = alias
        self.flag_alias = flag_alias

class Relationship:
    """docstring for Relationship."""

    def __init__(self, left_table, right_table, left_table_keys, right_table_keys):
        self.left_table = left_table
        self.left_table_keys = left_table_keys
        self.table_right = table_right
        self.right_table_keys = right_table_keys

class Table:

    def __init__(self, database, table_name, table_alias, table_safra, table_key='nr_cpf_cnpj'):
        self.database = database
        self.table_name = table_name
        self.table_alias = table_alias
        self.table_safra = table_safra
        self.table_key = table_key
        self.variables = []

    @staticmethod
    def _var_name_checker(var_name):
        if len(var_name.strip())>0:
            return True
        return False

    def add_var(self, var_name, var_alias='', flag_alias=False):
        if (not self._in_var_list(var_name)) and (self._var_name_checker(var_name)):
            self.variables.append(Variable(table=self,
                                           name=var_name.strip(),
                                           alias=var_alias.strip(),
                                           flag_alias=flag_alias
                                           ))
        return self

    def _in_var_list(self, var_name):
        for var in self.variables:
            if var_name.strip().lower() == var.name.lower():
                return True
        return False


class ConfigQueryReader:
    """docstring for ConfigQueryReader."""

    def __init__(self, safra, query_dict):
        self.query = None
        self.safra = safra
        self.query_dict = query_dict

    def get_main_table(self):
        # if len(self.query_dict.keys()) == 0:
        #     raise ValueError("No tables in table dict.")
        main_tb_name = ''
        main_tb = None
        main_count_validator = 0
        for tb_name, tb in self.query_dict.items():
            if "MAIN" in tb.keys():
                main_count_validator += 1
                main_tb_name = tb_name.format(m_add(self.safra, -tb["MAIN"]))
                main_tb = tb
        if main_count_validator == 1:
            return main_tb_name, main_tb
        elif main_count_validator > 1:
            raise ValueError("More than one main table in table dict.")
        else:
            raise ValueError("No main table in table dict.")

    def build(self):

        tb_name, tb = self.get_main_table()
        #maybe the problem is I need to check if there is already a table when I first create it
        #need to create an initializer that returns table reference if table already exists =/
        #how do I do this?
        self.query = Query.create().main_table.add(tb["DATABASE"],
                                               tb_name.format(m_add(self.safra,tb['MAIN'])),
                                               tb["ALIAS"].format(m_add(self.safra,tb['MAIN'])),
                                               m_add(self.safra,tb['MAIN']))

        for tb_name, tb in self.query_dict.items():
            for var_name, var in tb["VARS"].items():
                for mX in var["SAFRAS"]:
                    self.query.left_join.add(tb["DATABASE"],
                                             tb_name.format(m_add(self.safra, -mX)),
                                             tb["ALIAS"].format(m_add(self.safra, -mX)),
                                             m_add(self.safra,-mX)).variable.\
                                             add(tb_name.format(m_add(self.safra, -mX)),
                                             var_name.format(m_add(self.safra,-(mX+var["DEFASAGEM"])))
                                             )
        self.query = self.query.build()

        return self

    def __str__(self):
        return str(self.query)



if __name__ == '__main__':

    # Testing

    safra = pd.Timestamp.now().strftime(format='%Y%m')
    print(m_add(safra, -1))



# from enum import Enum, auto
    # @staticmethod
    # def create(database, table_name, table_alias, table_safra, table_key='nr_cpf_cnpj'):
    #     return QueryBuilder(database,
    #                         table_name,
    #                         table_alias,
    #                         table_safra,
    #                         table_key
    #             )
# class Singleton(type):
#     """docstring for Singleton."""
#     _instances = {}
#
#     def __call__(cls, *args, **kwargs):
#         if cls not in cls._instances:
#             cls._instances[cls] = super(Singleton, cls)\
#             .__call__(*args,**kwargs)
#         return cls._instances[cls]
#
# class TableSwitch(Enum):
#     """docstring for TableSwitch."""
#     VALID = auto()
#     INVALID = auto()
#
# class TableTriggers(Enum):
#     TABLE_NAME_CHECK = auto()
#     TABLE_ALIAS_CHECK = auto()



    # @staticmethod
    # def create_table(database, table_name, table_alias, table_safra, table_key='nr_cpf_cnpj'):
    #     # print(1)
    #     # print(self.__state)
    #     return Table(database,
    #                  table_name,
    #                  table_alias,
    #                  table_safra,
    #                  table_key)

    # def _unique(self, alias, tb_complete_name):
    #     for tb in self.__created_objects:
    #         if alias == tb.table_alias.format(tb.table_safra): return False
    #         if tb_complete_name == tb.table_name.format(tb.table_safra): return False
    #     return True
    #
    # def _get_table(self, tb_complete_name):
    #     for tb in self.__created_objects:
    #         if tb_complete_name == tb.table_name.format(tb.table_safra): return tb
    #     raise ValueError('Table doesn\'t exist.')
