## blast
```
makeblastdb -in your_species_protein.fa -dbtype prot -out your_species_name

blastp -db your_species_name -query reference_gene_family_protein.fa -evalue 1e-5 -outfmt 6 -out your_gene_family_name_your_species_name.blast

```
