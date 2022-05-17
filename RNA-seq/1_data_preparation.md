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
