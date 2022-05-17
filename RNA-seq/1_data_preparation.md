## 1、解压测序文件
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

## 2、检查数据质量
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
