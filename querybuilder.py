import pandas as pd

def m_add(safra_ini, qt):
    return (pd.to_datetime(safra_ini, format="%Y%m") + pd.DateOffset(months=qt)).strftime("%Y%m")

class Query(object):
    def __init__(self):
        self.main_table = None
        self._all_tables = []
        self._join_list = []
        self._variables = []
        self._filters = []

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

    def _get_filters(self):
        if len(self._filters)>0:
            filter_start = 'WHERE \n'
            filters = '\nAND\n'.join(['{} {} {}'.format(filter.var_name,filter.operator,filter.value)  for filter in self._filters])
            return filter_start + filters
        else:
            return ''

    def _get_table_joins(self):
        return '\n'.join(['''left join\n\t{db}.{tb} {alias}\n\ton {main_alias}.{main_key}={alias}.{key}'''.format(db=tb.database,
                                                                                  tb=tb.table_name,
                                                                                  alias=tb.table_alias,
                                                                                  key=tb.table_key,
                                                                                  main_alias=self.main_table.table_alias,
                                                                                  main_key=self.main_table.table_key)
                                                                                  if tb.database != ''
                                                                                  else
                          '''left join\n\t{tb} {alias}\n\ton {main_alias}.{main_key}={alias}.{key}'''.format(tb=tb.table_name,
                                                                                                      alias=tb.table_alias,
                                                                                                      key=tb.table_key,
                                                                                                      main_alias=self.main_table.table_alias,
                                                                                                      main_key=self.main_table.table_key)
                                                                                  for tb in self._join_list])

    def _get_main_table(self):
        main_tb_part = '\t{}.{} {}'.format(self.main_table.database,self.main_table.table_name, self.main_table.table_alias)
        if self.main_table.database == '':
            main_tb_part = '\t{} {}'.format(self.main_table.table_name, self.main_table.table_alias)
        return main_tb_part

    def _get_all_variables(self):
        all_vars = ',\n'.join(['\t{} as {}'.format(var.name, var.alias)
                                     if var.fl_case_when else
                                     '\t{}.{}'.format(var.table.table_alias,var.name)
                                     for var in self._variables])
        return all_vars

    def __str__(self):

        query = "SELECT\n{}\nFROM\n{}\n{}\n{}".format(self._get_all_variables(),
                                                  self._get_main_table(),
                                                  self._get_table_joins(),
                                                  self._get_filters())

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

    @property
    def case_when(self):
        return QueryCaseWhenBuilder(self.query)

    # Next properties I will implement for the QueryBuilder
    # @property
    # def group_by(self):
    #     return QueryGroupByBuilder(self.query)
    #
    @property
    def where(self):
        return QueryWhereBuilder(self.query)
    #
    # @property
    # def primitive(self):
    #     return QueryPrimitiveBuilder(self.query)

    def build(self):
        return self.query


class Condition:
    def __init__(self, var_name, operator, value, l_operator=' AND '):
        self.var_name = var_name
        self.operator = operator
        self.value = value
        self.l_operator = l_operator

class QueryWhereBuilder(QueryBuilder):
    def __init__(self, query):
        super().__init__(query)

    def add(self, var_name, operator, value):
        self.query._filters.append(
                                    Condition(
                                    var_name,
                                    operator,
                                    value
                                    )
        )
        return self

class QueryCaseWhenBuilder(QueryBuilder):
    def __init__(self, query):
        super().__init__(query)

    def add(self, table_name, var_name, var_alias, transformation):
        if not self.query._in_var_list(var_name):
            tb = self.query._get_table_ref(table_name=table_name)
            var_name = self._create_case_when(tb, var_name, transformation)
            tb.add_var(var_name, var_alias=var_alias, fl_case_when=True)
            self.query._variables.append(tb.variables[-1])
        return self

    def _create_case_when(self, tb, var_name, transformation):
        case = "case\n"
        whens = []
        else_ = ''
        if transformation[-1][0]=="else":
            for line in transformation[:-1]:
                when = "\twhen"
                for condition in line[:-1]:
                    when += " {tb_alias}.{var_name}" + condition
                when += " then " + str(line[-1])
                whens += [when]
            whens = "\n\t".join(whens)
            else_ = "\n\telse {} end".format(transformation[-1][1])
        else:
            for line in transformation:
                when = "\twhen"
                for condition in line[:-1]:
                    when += " {tb_alias}.{var_name}" + condition
                when += " then " + str(line[-1])
                whens += [when]
            whens = "\n\t".join(whens)
        case_when = case + whens + else_
        return case_when.format(tb_alias=tb.table_alias, var_name=var_name)

class QueryVariableBuilder(QueryBuilder):
    def __init__(self, query):
        super().__init__(query)

    def add(self, table_name, var_name, var_alias=''):
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

    def __init__(self, table, name, alias='', fl_case_when=False, flag_alias=False):
        self.name = name
        self.table = table
        self.alias = alias
        self.flag_alias = flag_alias
        # self.type = var_type
        self.fl_case_when = fl_case_when

# Need to think of some smart way to apply this class so I can with more complex
# relationships between tables
class Relationship:

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

    def add_var(self, var_name, var_alias='', flag_alias=False, fl_case_when=False):
        if (not self._in_var_list(var_name)) and (self._var_name_checker(var_name)):
            self.variables.append(Variable(table=self,
                                           name=var_name.strip(),
                                           alias=var_alias.strip(),
                                           flag_alias=flag_alias,
                                           fl_case_when=fl_case_when
                                           ))
        return self

    def _in_var_list(self, var_name):
        for var in self.variables:
            if var_name.strip().lower() == var.name.lower():
                return True
        return False


class BaseReaderStrategy:

    def __init__(self, safras, query_dict):
        self.query = None
        self.safras = safras
        self.query_dict = query_dict

    def get_main_table(self, safra):

        main_tb_name = ''
        main_tb = None
        main_count_validator = 0
        for tb_name, tb in self.query_dict.items():
            if "MAIN" in tb.keys():
                main_count_validator += 1
                main_tb_name = tb_name.format(m_add(safra, -tb["MAIN"]))
                main_tb = tb
        if main_count_validator == 1:
            return main_tb_name, main_tb
        elif main_count_validator > 1:
            raise ValueError("More than one main table in table dict.")
        else:
            raise ValueError("No main table in table dict.")

    def build(self, safras, query_dict): pass

class UnionAllSafrasStrategy(BaseReaderStrategy):
    def build(self):

        self.query = "select * from \n ("

        subqueries = []
        for safra in self.safras:

            subqueries.append(str(SimpleQueryStrategy([safra], self.query_dict).build()))

        self.query += (')\n UNION ALL \n(').join(subqueries) + ')'

        return self.query

class StackedTableQueryStrategy(BaseReaderStrategy):
    def build(self):

        safra = self.safras[0]

        tb_name, tb = self.get_main_table(safra)

        main_tb_name = tb["ALIAS"].format(m_add(safra,tb['MAIN']))

        subquery_dict = {}

        subquery_dict[tb["ALIAS"].format(m_add(safra,tb['MAIN']))] = \
                                               Query.create().main_table.add(tb["DATABASE"],
                                               tb_name.format(m_add(safra,tb['MAIN'])),
                                               tb["ALIAS"].format(m_add(safra,tb['MAIN'])),
                                               m_add(safra,tb['MAIN'])).where.add(
                                               tb["ALIAS"].format(m_add(safra,tb['MAIN']))+'.data_ref',
                                               '=',
                                               m_add(safra,tb['MAIN'])
                                               )

        for tb_name, tb in self.query_dict.items():
            for var_name, var in tb["VARS"].items():
                for mX in var["SAFRAS"]:
                    if tb["ALIAS"].format(m_add(safra, -mX)) in subquery_dict.keys():
                        if "TRANSFORMATIONS" not in var.keys():
                            subquery_dict[tb["ALIAS"].format(m_add(safra, -mX))].variable.\
                                                 add(tb_name.format(m_add(safra, -mX)),
                                                 var_name.format(m_add(safra,-(mX+var["DEFASAGEM"])))
                                                 )
                        elif str(mX) in var["TRANSFORMATIONS"].keys():
                            subquery_dict[tb["ALIAS"].format(m_add(safra, -mX))].case_when.\
                                                 add(tb_name.format(m_add(safra, -mX)),
                                                 var_name.format(m_add(safra,-(mX+var["DEFASAGEM"]))),
                                                 var["ALIAS"],
                                                 var["TRANSFORMATIONS"][str(mX)]
                                                 )
                        else:
                            subquery_dict[tb["ALIAS"].format(m_add(safra, -mX))].variable.\
                                                 add(tb_name.format(m_add(safra, -mX)),
                                                 var_name.format(m_add(safra,-(mX+var["DEFASAGEM"])))
                                                 )
                    else:
                        subquery_dict[tb["ALIAS"].format(m_add(safra, -mX))] = \
                                               Query.create().main_table.add(tb["DATABASE"],
                                               tb_name.format(m_add(safra, -mX)),
                                               tb["ALIAS"].format(m_add(safra, -mX)),
                                               m_add(safra,-mX)).where.add(
                                               tb["ALIAS"].format(m_add(safra, -mX))+'.data_ref',
                                               '=',
                                               m_add(safra, -mX)
                                               ).\
                                               variable.\
                                               add(tb_name.format(m_add(safra, -mX)), 'nr_cpf_cnpj')
                        if "TRANSFORMATIONS" not in var.keys():
                            subquery_dict[tb["ALIAS"].format(m_add(safra, -mX))].variable.\
                                                 add(tb_name.format(m_add(safra, -mX)),
                                                 var_name.format(m_add(safra,-(mX+var["DEFASAGEM"])))
                                                 )
                        elif str(mX) in var["TRANSFORMATIONS"].keys():
                            subquery_dict[tb["ALIAS"].format(m_add(safra, -mX))].case_when.\
                                                 add(tb_name.format(m_add(safra, -mX)),
                                                 var_name.format(m_add(safra,-(mX+var["DEFASAGEM"]))),
                                                 var["ALIAS"],
                                                 var["TRANSFORMATIONS"][str(mX)]
                                                 )
                        else:
                            subquery_dict[tb["ALIAS"].format(m_add(safra, -mX))].variable.\
                                                 add(tb_name.format(m_add(safra, -mX)),
                                                 var_name.format(m_add(safra,-(mX+var["DEFASAGEM"])))
                                                 )

        stacked_query = Query.create().main_table.add('',
                           '({})'.format(str(subquery_dict[main_tb_name].build())),
                           main_tb_name,
                           0)
        for vr in subquery_dict[main_tb_name].query._variables:
           if vr.alias != '':
               stacked_query.variable.\
                    add('({})'.format(str(subquery_dict[main_tb_name].build())),
                        vr.alias)
           else:
               stacked_query.variable.\
                    add('({})'.format(str(subquery_dict[main_tb_name].build())),
                        vr.name)

        for tb_alias, subquery in subquery_dict.items():
            if tb_alias != main_tb_name:
                name = '({})'.format(str(subquery.build()))
                stacked_query.left_join.add('',
                                        name,
                                        tb_alias,
                                        0)
                for vr in subquery.query._variables:
                   if vr.alias != '':
                       stacked_query.variable.\
                            add(name,
                                vr.alias)
                   else:
                       stacked_query.variable.\
                            add(name,
                                vr.name)


        self.query = str(stacked_query.build())

        return self.query

class SimpleQueryStrategy(BaseReaderStrategy):
    def build(self):

        safra = self.safras[0]

        tb_name, tb = self.get_main_table(safra)

        subquery = Query.create().main_table.add(tb["DATABASE"],
                                               tb_name.format(m_add(safra,tb['MAIN'])),
                                               tb["ALIAS"].format(m_add(safra,tb['MAIN'])),
                                               m_add(safra,tb['MAIN']))

        for tb_name, tb in self.query_dict.items():
            for var_name, var in tb["VARS"].items():
                for mX in var["SAFRAS"]:
                    subquery.left_join.add(tb["DATABASE"],
                                             tb_name.format(m_add(safra, -mX)),
                                             tb["ALIAS"].format(m_add(safra, -mX)),
                                             m_add(safra,-mX))
                    if "TRANSFORMATIONS" not in var.keys():
                        subquery.variable.add(tb_name.format(m_add(safra, -mX)),
                                             var_name.format(m_add(safra,-(mX+var["DEFASAGEM"])))
                                             )
                    elif str(mX) in var["TRANSFORMATIONS"].keys():
                        subquery.case_when.add(tb_name.format(m_add(safra, -mX)),
                                             var_name.format(m_add(safra,-(mX+var["DEFASAGEM"]))),
                                             var["ALIAS"],
                                             var["TRANSFORMATIONS"][str(mX)]
                                             )
                    else:
                        subquery.variable.add(tb_name.format(m_add(safra, -mX)),
                                             var_name.format(m_add(safra,-(mX+var["DEFASAGEM"])))
                                             )

        self.query = str(subquery.build())

        return self.query

class ConfigQueryReader:

    def __init__(self, safras, query_dict):
        self.query = None
        self.safras = safras
        self.query_dict = query_dict
        self.query_strategy = None

    def build(self, stacked=False):
        self._set_strategy(stacked)
        self.query = self.query_strategy.build()
        return self

    def _set_strategy(self,stacked):
        if stacked:
            self.query_strategy = StackedTableQueryStrategy(self.safras, self.query_dict)
        elif len(self.safras)>1:
            self.query_strategy = UnionAllSafrasStrategy(self.safras, self.query_dict)
        elif len(self.safras)==1:
            self.query_strategy = SimpleQueryStrategy(self.safras, self.query_dict)

    def __str__(self):
        return str(self.query)
