## 背景说明
* 水稻物种，三个品种四种处理，共 3 乘 4 组，每一组 3 个重复，一共 36 个样本，另外，有两个品种在某处理下各做了一个样本，因此再增加两个，一共 38 个样本。
* 从测序公司处获得 clean data 。公司对 clean data 的标准是：当任一测序read中N含量超过该read碱基数的10%时，去除此paired reads；当任一测序read中含有的低质量（Q<=5）碱基数超过该条read碱基数的50%时，去除此paired reads；当任一测序read中含有接头序列，去除此paired reads。基于此标准进行的研究可以被高水平杂志认可(Yan L.Y. et al . 2013)。

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
  gunzip $line
done
# 按 esc ：wq 保存

# 运行脚本
nohup sh gz2fq.sh 1>gz2fq.out 2>&1 &
```



