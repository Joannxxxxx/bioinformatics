# 使用方式：用来操纵 3_DEG_analysis_git.r 脚本的循环
# 输入：前一步准备好的 Rdata、对照组组名

# 设置工作目录
wd=/your/working/directory
# 指定工作脚本
DEG_analysis=$wd/DEG_analysis.r

prefix=$1 # 输入 DEG_preparation 的文件名前缀
control=$2 # 输入对照组组名
case_groups=$wd/case_groups.txt # 所有实验组组名所在文件


cat $case_groups | while read group
do
    # echo $group
    # echo $prefix
    # echo $control
    Rscript $DEG_analysis $wd $prefix $control $group
done

