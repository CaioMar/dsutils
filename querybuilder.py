class QueryBuilder:
    def __init__(self, database, table_name, table_alias, table_safra, table_key='nr_cpf_cnpj'):
        self.__root = SelectQuery(database,
                                  table_name,
                                  table_alias,
                                  table_safra,
                                  table_key)
        
    def left_join(self, database, table_name, table_alias, table_safra, table_key='nr_cpf_cnpj'):
        self.__root.tables_to_left_join.append(Table(database,
                                                     table_name,
                                                     table_alias,
                                                     table_safra,
                                                     table_key)
                                               )
        return self

    def __str__(self):
        return str(self.__root)


class SelectQuery:
    def __init__(self, database, table_name, table_alias, table_safra, table_key='nr_cpf_cnpj'):
        self.main_table = Table(database,
                                table_name,
                                table_alias,
                                table_safra,
                                table_key)
        self.tables_to_left_join = []

    def __str__(self):
        if len(self.tables_to_left_join)>0:
            return f'select * from\n\t{self.main_table.database}.{self.main_table.table_name}_{self.main_table.table_safra} {self.main_table.table_alias}' + \
                    ''.join(['\nleft join\n\t{db}.{tb}_{safra} {alias}\n\ton {main_table_alias}.{main_table_key}={alias}.{key}'.format(
                        db=tb.database,
                        tb=tb.table_name,
                        safra = tb.table_safra,
                        alias=tb.table_alias,
                        main_table_alias=self.main_table.table_alias,
                        main_table_key=self.main_table.table_key,
                        key=tb.table_key) for tb in self.tables_to_left_join                        
                        ])
        else:
            return f'select * from {self.main_table.database}.{self.main_table.table_name}{self.main_table.table_safra}{self.main_table.table_alias}'
                   

    @staticmethod
    def create(database, table_name, table_alias, table_safra, table_key='nr_cpf_cnpj'):
        return QueryBuilder(database,
                            table_name,
                            table_alias,
                            table_safra,
                            table_key
                )

class Table:
    def __init__(self, database, table_name, table_alias, table_safra, table_key='nr_cpf_cnpj'):
        self.database = database
        self.table_name = table_name
        self.table_alias = table_alias
        self.table_safra = table_safra
        self.table_key = table_key

query = SelectQuery.create('database','table1','a','201907')\
        .left_join('database','table2','b','201907')

print(query)
