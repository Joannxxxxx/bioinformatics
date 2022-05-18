## 写在前面
1、文件夹结构 
├── 1_gz
├── 2_fq
├── 3_fq_repo
├── 4_trim
├── 5_trim_repo
├── 6_sam
├── 7_bam
├── reference
└── scripts
2、需要安装的软件 
* fastqc
* multicqc
* trim_galore
* hisat2
* samtools
* featureCounts

## 流程：从测序数据到表达矩阵
### 1、解压测序文件
一般测序获得数据为压缩格式，以 .gz 结尾，可以先作解压，也可以不做（第 2 步会说）。 
```
# gz2fq.sh 脚本内容
cd .. # 回到上级目录
mkdir fq # 创建 fq 目录以保存解压 gz 文件后的 fasta 文件
cd gz # 切换工作目录至 gz 文件夹

ls *.gz > gz_list # 将所有 gz 文件名做成一份列表

# 逐一解压 gz 文件至 fq 文件夹下
cat gz_list | while read line
do
        file=${line%.*}
        gunzip -c $line > ../fq/$file
done
```

```
cd scripts # 将工作路径切换到 scripts 文件夹
vim gz2fq.sh # 创建脚本
# 粘贴上述脚本内容，在英文输入模式下依次按 :wq 保存并退出
nohup sh gz2fq.sh 1>gz2fq.out 2>&1 & # 运行脚本
```

### 2、检查数据质量
fastqc 可以直接处理 gz 压缩文件，所以上一步的解压可以不做。  
```
# fq2qc.sh 脚本内容
cd ../fq # 切换工作目录至 fq 文件夹
mkdir fq_repo # 创建 fq_repo 以存放 qc 结果
ls *.fq > fq_list # 将所有 fq 文件名做成一份列表

# 对 fq 文件进行 qc 并把结果存放至 fq_repo 文件夹
cat fq_list | while read line
do
    fastqc -o fq_repo -t 12 $line
done
```

```
cd scripts # 将工作路径切换到 scripts 文件夹
vim fq2qc.sh # 创建脚本
# 粘贴上述脚本内容，在英文输入模式下依次按 :wq 保存并退出
nohup sh fq2qc.sh 1>fq2qc.out 2>&1 & # 运行脚本
```

```
cd ../fq/fq_repo # 将工作路径切换到 fq_repo 文件夹
multiqc ./ # 使用 multiqc 将所有 QC 报告合并为一个 html 以方便查看
```

### 3、过滤并查看过滤后的数据质量  

```
# fq2trim.sh 脚本内容
cd .. # 切换到上级目录
mkdir trim # 创建 trim 文件夹以存放过滤后的 fasta 文件

cd fq # 切换工作目录到 fq 文件
ls *R1_* > r1 # 将双端测序中的 r1 文件名制成一个列表
ls *R2_* > r2 # 将双端测序中的 r2 文件名制成一个列表
paste r1 r2 > pair_list # 将两个列表合并

# 过滤操作
cat pair_list | while read line
do
  arr=($line)
  fq1=${arr[0]}
  fq2=${arr[1]}
  trim_galore -q 25 --phred33 --length 50 -e 0.1 --stringency 3 -o ../trim -j 12 --paired $fq1 $fq2
done 
```

```
cd scripts # 将工作路径切换到 scripts 文件夹
vim fq2trim.sh # 创建脚本
# 粘贴上述脚本内容，在英文输入模式下依次按 :wq 保存并退出
nohup sh fq2trim.sh 1>fq2trim.out 2>&1 & # 运行脚本
```



```
# trim2qc.sh 脚本内容
cd ../trim # 切换工作目录至 trim 文件夹
mkdir trim_repo # 创建 trim_repo 以存放 qc 结果
ls *.fq > trim_list # 将所有 fq 文件名做成一份列表

# 对 fq 文件进行 qc 并把结果存放至 trim_repo 文件夹
cat trim_list | while read line
do
    fastqc -o trim_repo -t 12 $line
done
```

```
cd scripts # 将工作路径切换到 scripts 文件夹
vim trim2qc.sh # 创建脚本
# 粘贴上述脚本内容，在英文输入模式下依次按 :wq 保存并退出
nohup sh trim2qc.sh 1>trim2qc.out 2>&1 & # 运行脚本
```

```
cd ../trim/trim_repo # 将工作路径切换到 trim_repo 文件夹
multiqc ./ # 使用 multiqc 将所有 QC 报告合并为一个 html 以方便查看
```

### 4、构建索引

```
# build_index.sh 脚本内容
cd ../reference # 将工作路径切换到 reference 文件夹
extract_exons.py sativa.gtf > oryza_exon # 提取外显子
extract_splice_sites.py sativa.gtf > oryza_ss # 提取可变剪切
hisat2-build -p 12 sativa.fa sativa.gtf --exon oryza_exon --ss oryza_ss # 构建索引
mkdir index # 创建文 件夹以存放构建好的索引
mv *.ht2 index # 将索引文件移动到 index 文件夹
```

```
cd scripts # 将工作路径切换到 scripts 文件夹
vim build_index.sh # 创建脚本
# 粘贴上述脚本内容，在英文输入模式下依次按 :wq 保存并退出
nohup sh build_index.sh 1>build_index.out 2>&1 & # 运行脚本
```

### 5、比对
```
# hisat2_align.sh 脚本内容
cd ../trim # 将工作路径切换到 trim 文件夹

ls *R1_* > r1 # 将双端测序中的 r1 文件名制成一个列表
ls *R2_* > r2 # 将双端测序中的 r2 文件名制成一个列表
paste r1 r2 > pair_list # 将两个列表合并

# align
cat pair_list | while read line
do 
  arr=($line)
  fq1=${arr[0]}
  fq2=${arr[1]}
  prefix=${fq1%%.*}
  
  hisat2 -p 12 -x ../reference/index/sativa.gtf -1 $fq1 -2 $fq2 -S $prefix.sam
done
```

```
cd scripts # 将工作路径切换到 scripts 文件夹
vim hisat2_align.sh # 创建脚本
# 粘贴上述脚本内容，在英文输入模式下依次按 :wq 保存并退出
nohup sh hisat2_align.sh 1>hisat2_align.out 2>&1 & # 运行脚本
```

### 6、将 sam 文件转为 bam 文件
```
# sam2bam.sh 脚本内容
cd .. # 切换到上级目录
mkdir sam bam # 创建 sam bam 两个文件夹分别存放 sam bam 格式的文件
mv trim/*.sam sam # 将 trim 文件夹中的 sam 文件移动到 sam 文件夹
cd sam # 将工作路径切换到 sam 文件夹
ls *.sam > sam_list # 将所有 sam 文件名做成一份列表
# 将 sam 文件转为 bam 文件
cat sam_list | while read line
do
	samtools sort -o ../bam/${line%.*}.bam -O bam -@ 12 $line
done
```

```
cd scripts # 将工作路径切换到 scripts 文件夹
vim sam2bam.sh # 创建脚本
# 粘贴上述脚本内容，在英文输入模式下依次按 :wq 保存并退出
nohup sh sam2bam.sh 1>sam2bam.out 2>&1 & # 运行脚本
```

### 7、计算表达矩阵
```
cd bam # 将工作路径切换到 bam 文件夹
# 计算表达矩阵
nohup featureCounts -T 12 -a ../reference/sativa.gtf -o sativa.expr.txt *.bam 1>featureCounts.out 2>&1 &
```
