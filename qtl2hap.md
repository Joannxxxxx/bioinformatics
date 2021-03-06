### qtl2hap
主要内容：从 GWAS 获得结果后提取 qtl 获得候选基因  
1、提取每条染色体上 pvalue 最小的前 N 个（比如前 30 个） snp - 生成前N个snp.sh，得到 chr$i.top30.txt 和 chr$i.top30.snp.txt 两个文件，前者是前 N 个 snp 的详细信息，包括 snp,chr,pos,pvalue，后者是前 N 个 snp 的简略信息，仅有 snp 名一列  
2、分别在曼哈顿图上标出这些 snp，选择峰尖位置的 snp，若某些峰尖被上一步遗漏，再回到上一步提取特定区域的 top snp，形成 chr$i.add.txt 和 chr$i.add.snp.txt  - 2对前N个snp逐一画曼哈顿图.sh，生成 qtl_snp.txt  
3、提取这些 snp 附近 50kb （按照 LD 衰减距离的一半来确定）范围的所有基因 - 对peak提取上下游50kb范围内的vcf.sh + 注释peak范围vcf.sh  
4、分别提取这些基因 - 提取qtl基因.sh + 提取基因range.sh + 提取基因vcf.sh + 生成基因不同序列的注释文件.sh  
  * 上游 2kb 启动子范围内的序列
  * 基因内部序列
  * 基因外显子序列
  * 基因非同义突变外显子序列  
  
5、对这些序列分别进行单倍型分析，记录有统计差异的基因 - 基因单倍型分析.sh  
6、对结果进行比较，选择 pvalue 较小且同时在基因内部与启动子序列上都显著（或同时在基因外显子序列与启动子序列，究竟如何选择要再研究）的基因作为关注对象  
