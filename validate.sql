select * from
 (SELECT
	tb1_201901.var1_201901,
	case
	when tb1_201812.var1_201812=0 then 11
		when tb1_201812.var1_201812=1 then 12
	else 15 end as var1_,
	case
	when tb1_201901.var2_201901>=0 tb1_201901.var2_201901<=0 then 11
		when tb1_201901.var2_201901>=1 tb1_201901.var2_201901<=2 then 12
	else 15 end as var2_,
	tb2_201901.var3_201812,
	tb2_201901.var4_201901,
	tb2_201810.var4_201810
FROM
	db.tabela1_201901 tb1_201901
left join
	db.tabela1_201812 tb1_201812
	on tb1_201901.nr_cpf_cnpj=tb1_201812.nr_cpf_cnpj
left join
	db.tabela2_201901 tb2_201901
	on tb1_201901.nr_cpf_cnpj=tb2_201901.nr_cpf_cnpj
left join
	db.tabela2_201810 tb2_201810
	on tb1_201901.nr_cpf_cnpj=tb2_201810.nr_cpf_cnpj)
 UNION ALL
(SELECT
	tb1_201902.var1_201902,
	case
	when tb1_201901.var1_201901=0 then 11
		when tb1_201901.var1_201901=1 then 12
	else 15 end as var1_,
	case
	when tb1_201902.var2_201902>=0 tb1_201902.var2_201902<=0 then 11
		when tb1_201902.var2_201902>=1 tb1_201902.var2_201902<=2 then 12
	else 15 end as var2_,
	tb2_201902.var3_201901,
	tb2_201902.var4_201902,
	tb2_201811.var4_201811
FROM
	db.tabela1_201902 tb1_201902
left join
	db.tabela1_201901 tb1_201901
	on tb1_201902.nr_cpf_cnpj=tb1_201901.nr_cpf_cnpj
left join
	db.tabela2_201902 tb2_201902
	on tb1_201902.nr_cpf_cnpj=tb2_201902.nr_cpf_cnpj
left join
	db.tabela2_201811 tb2_201811
	on tb1_201902.nr_cpf_cnpj=tb2_201811.nr_cpf_cnpj)
 UNION ALL
(SELECT
	tb1_201903.var1_201903,
	case
	when tb1_201902.var1_201902=0 then 11
		when tb1_201902.var1_201902=1 then 12
	else 15 end as var1_,
	case
	when tb1_201903.var2_201903>=0 tb1_201903.var2_201903<=0 then 11
		when tb1_201903.var2_201903>=1 tb1_201903.var2_201903<=2 then 12
	else 15 end as var2_,
	tb2_201903.var3_201902,
	tb2_201903.var4_201903,
	tb2_201812.var4_201812
FROM
	db.tabela1_201903 tb1_201903
left join
	db.tabela1_201902 tb1_201902
	on tb1_201903.nr_cpf_cnpj=tb1_201902.nr_cpf_cnpj
left join
	db.tabela2_201903 tb2_201903
	on tb1_201903.nr_cpf_cnpj=tb2_201903.nr_cpf_cnpj
left join
	db.tabela2_201812 tb2_201812
	on tb1_201903.nr_cpf_cnpj=tb2_201812.nr_cpf_cnpj)
