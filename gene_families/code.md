## blast
```
# 建库，输入物种蛋白序列的 fasta 文件
makeblastdb -in your_species_protein.fa -dbtype prot -out your_species_name

# 比对，将模板基因（来自基因家族）的蛋白序列与建好的库进行比对，得到候选基因
blastp -db your_species_name -query reference_gene_family_protein.fa -evalue 1e-5 -outfmt 6 -out your_gene_family_name_your_species_name.blast

```
