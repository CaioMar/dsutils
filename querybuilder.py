class Query:
    def __init__(self):
        self.main_table = None
        self.join_list = []
        self._all_tables = []
        self._flag_var_alias = True

    def __str__(self):
        return ''
        # if len(self.join_list)>0:
        #     return f'select * from\n\t{self.main_table.database}.{self.main_table.table_name}_{self.main_table.table_safra} {self.main_table.table_alias}' + \
        #             ''.join(['\nleft join\n\t{db}.{tb}_{safra} {alias}\n\ton {main_table_alias}.{main_table_key}={alias}.{key}'.format(
        #                 db=tb.database,
        #                 tb=tb.table_name,
        #                 safra = tb.table_safra,
        #                 alias=tb.table_alias,
        #                 main_table_alias=self.main_table.table_alias,
        #                 main_table_key=self.main_table.table_key,
        #                 key=tb.table_key) for tb in self.join_list
        #                 ])
        # else:
        #     return f'select * from {self.main_table.database}.{self.main_table.table_name}{self.main_table.table_safra}{self.main_table.table_alias}'


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

    def builder(self):
        return self.query


class QueryLeftJoinBuilder(QueryBuilder):
    def __init__(self, query):
        super().__init__(query)

    def left_join(self, database, table_name, table_alias, table_safra, table_key='nr_cpf_cnpj'):
        self.join_list.append(Table(database,
                                    table_name,
                                    table_alias,
                                    table_safra,
                                    table_key)
                                    )
        return self


class QueryMainTableBuilder(QueryBuilder):
    """docstring for QueryMainTableBuilder."""

    def __init__(self, query):
        super().__init__(query)

    def main_table(self, database, table_name, table_alias, table_safra, table_key='nr_cpf_cnpj'):
        self.main_table = Table(database,
                                table_name,
                                table_alias,
                                table_safra,
                                table_key)
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

if __name__ == '__main__':
    d1 = Table.create_table('database','table1_{}','a_{}','201907').add_var('variavel1').add_var('variavel2')
    d2 = Table.create_table('database','table2_{}','b_{}','201907').add_var('variavel3')
# query = SelectQuery.create('database','table1','a','201907')\
#         .left_join('database','table2','b','201907')

    print(d1, d2, d1==d2)

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
