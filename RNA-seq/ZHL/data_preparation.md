## 背景说明
* 水稻物种，三个品种四种处理，共 3 乘 4 组，每一组 3 个重复，一共 36 个样本，另外，有两个品种在某处理下各做了一个样本，因此再增加两个，一共 38 个样本。
* 从测序公司处获得 clean data 。公司对 clean data 的标准是：当任一测序read中N含量超过该read碱基数的10%时，去除此paired reads；当任一测序read中含有的低质量（Q<=5）碱基数超过该条read碱基数的50%时，去除此paired reads；当任一测序read中含有接头序列，去除此paired reads。基于此标准进行的研究可以被高水平杂志认可(Yan L.Y. et al . 2013)。
* 之前做过水稻的转录组分析，因此有已经构建好的索引，不必再次构建

## 数据准备
### 1 把每个样本文件夹里的 r1 r2 提取出来，统一放到 trim 文件夹里

```
# 把每个样本文件夹里的 fq.gz 文件移出来
for file in ./*
do
  mv $file/*.gz ../trim
done
```

### 2 解压

```
# 创建脚本
vim gz2fq.sh

# 按 i 键，粘贴入以下内容
# 解压 gz 文件
for file in ./*
do
  gunzip $file
done
# 在英文输入模式下依次按 esc :wq 保存退出

# 运行脚本
nohup sh gz2fq.sh 1>gz2fq.out 2>&1 &
```

### 3 比对
```
# 将工作路径切换到 trim 文件夹
cd ../trim 

ls *_1.clean* > r1 # 将双端测序中的 r1 文件名制成一个列表
ls *_2.clean* > r2 # 将双端测序中的 r2 文件名制成一个列表
paste r1 r2 > pair_list # 将两个列表合并

# 将工作路径切换到 scripts 文件夹
cd ../scripts 
# 创建脚本
vim hisat2_align.sh 

# 按 i 键，粘贴入以下内容
# align
cd ../trim 
cat pair_list | while read line
do 
  arr=($line)
  fq1=${arr[0]}
  fq2=${arr[1]}
  prefix=${fq1%%-*} # 截取第一个 - 符号前的字符作为文件名
  
  hisat2 -p 12 -x ../../reference/index/Oryza_sativa.IRGSP-1.0.51.gtf -1 $fq1 -2 $fq2 -S $prefix.sam
done
# 在英文输入模式下依次按 esc :wq 保存退出

# 运行脚本
nohup sh hisat2_align.sh 1>hisat2_align.out 2>&1 & # 运行脚本
```

### 4 把 sam 文件转为 bam 文件
```
# 把 trim 文件夹里的 sam 文件移到 sam 文件夹
cd ../trim
mkdir ../sam ../bam
mv ./*.sam ../sam
```

```
cd ../scripts
vim sam2bam.sh

# 按 i 键，粘贴入以下内容
cd ../sam
for file in ./* 
do
	samtools sort -o ../bam/${file%.*}.bam -O bam -@ 12 $file
done
# 在英文输入模式下依次按 :wq 保存并退出

# 运行脚本
nohup sh sam2bam.sh 1>sam2bam.out 2>&1 & 
```


