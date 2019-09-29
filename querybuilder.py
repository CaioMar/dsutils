import pandas as pd

def m_add(safra_ini, qt):
    return (pd.to_datetime(safra_ini, format="%Y%m") + pd.DateOffset(months=qt)).strftime("%Y%m")

class Query:
    def __init__(self):
        self.main_table = None
        self.join_list = []
        self._all_tables = []
        self._flag_var_alias = True

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
                                                                                  main_key=self.main_table.table_key) for tb in self.join_list])

    def _get_main_table(self):
        return '\t{} {}'.format(self.main_table.table_name, self.main_table.table_alias)

    def _get_all_variables(self):
        all_vars = []
        for tb in self._all_tables:
            all_vars += ['\n'.join(['\t{}.{} as {}'.format(tb.table_alias,var, var[:-6])
                                    for var in tb.variables])]
        all_vars = '\n'.join(all_vars)
        return all_vars

    def __str__(self):

        query = "SELECT\n{}\nFROM\n{}\n{}".format(self._get_all_variables(),
                                                  self._get_main_table(),
                                                  self._get_table_joins())

        return query

class QueryBuilder:
    """docstring for QueryBuilder."""

    def __init__(self, query=Query()):
        self.query = query

    @property
    def left_join(self):
        return QueryLeftJoinBuilder(self.query)

    @property
    def main_table(self):
        return QueryMainTableBuilder(self.query)

    def build(self):
        return self.query


class QueryLeftJoinBuilder(QueryBuilder):
    def __init__(self, query):
        super().__init__(query)

    def add_table(self, database, table_name, table_alias, table_safra, table_key='nr_cpf_cnpj'):
        self.query.join_list.append(Table(database,
                                    table_name,
                                    table_alias,
                                    table_safra,
                                    table_key)
                                    )
        self.query._all_tables.append(self.query.join_list[-1])
        return self


class QueryMainTableBuilder(QueryBuilder):
    """docstring for QueryMainTableBuilder."""
    def __init__(self, query):
        super().__init__(query)

    def add_table(self, database, table_name, table_alias, table_safra, table_key='nr_cpf_cnpj'):
        self.query.main_table = Table(database,
                                table_name,
                                table_alias,
                                table_safra,
                                table_key)
        self.query._all_tables.append(self.query.main_table)
        return self

class Table:

    def __init__(self, database, table_name, table_alias, table_safra, table_key='nr_cpf_cnpj'):
        self.database = database
        self.table_name = table_name
        self.table_alias = table_alias
        self.table_safra = table_safra
        self.table_key = table_key
        self.variables = []

    def add_var(self, var_name):
        self.variables.append(var_name)
        return self

class ConfigQueryReader:
    """docstring for ConfigQueryReader."""

    def __init__(self, safra, query_dict):
        self.query = None
        self.safra = safra
        self.query_dict = query_dict

    def get_main_table(self):
        if len(self.query_dict.keys()) == 0:
            raise ValueError("No tables in table dict.")
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
        self.query = QueryBuilder().main_table.add_table(tb["DATABASE"],
                                                         tb_name.format(m_add(self.safra,tb['MAIN'])),
                                                         tb["ALIAS"].format(m_add(self.safra,tb['MAIN'])),
                                                         m_add(self.safra,tb['MAIN']))

        for tb_name, tb in self.query_dict.items():
            for var_name, var in tb["VARS"].items():
                for mX in var["SAFRAS"]:
                    if not self.query.query._in_table_list(tb_name.format(m_add(self.safra, -mX))):
                        self.query.left_join.add_table(tb["DATABASE"],
                                                       tb_name.format(m_add(self.safra, -mX)),
                                                       tb["ALIAS"].format(m_add(self.safra, -mX)),
                                                       m_add(self.safra,-mX))
                    tb_ref = self.query.query._get_table_ref(tb_name.format(m_add(self.safra, -mX)))
                    tb_ref.add_var(var_name.format(m_add(self.safra,-(mX+var["DEFASAGEM"]))))

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
