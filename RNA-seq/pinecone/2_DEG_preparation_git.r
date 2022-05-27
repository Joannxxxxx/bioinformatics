# 清除变量
rm(list=ls()) 

# 设置工作路径
wd <- "your/working/directory/"
setwd(wd) 
getwd() # 查看工作路径，确认无误后继续

# 载入需要的程序包
library(reshape2)
library(stringr)
library(DESeq2)

# 读入原始表达矩阵
initial_expr <- read.table('initial_expr_git.txt',header=T)
head(initial_expr) # 查看表达矩阵前几行
colnames(initial_expr) # 查看表达矩阵列名

mother_expr <- initial_expr[,7:ncol(initial_expr)] # 提取表达矩阵表达量相关列
ncol(mother_expr)

geneLists <- initial_expr[,1] # 提取表达矩阵第一列 Geneid
head(geneLists) 
rownames(mother_expr) <- geneLists # 设置 index
head(mother_expr)

# 列名规整
cols_old <- colnames(mother_expr)
cols_old
cols_new <- str_split(cols_old,'\\_',simplify = T)[,1]
cols_new
colnames(mother_expr) <- cols_new
colnames(mother_expr)
head(mother_expr) # 最终的表达矩阵

# 保存最终的表达矩阵
filename <- "mother_expr.txt"
write.table(mother_expr,filename,sep="\t",quote = FALSE)

# 从最终表达矩阵中提取你要比较的组
my_expr <- mother_expr[,c(1:9,34:36)]
head(my_expr)

# 保存你要分析的表达矩阵
filename <- "abcx_wt_expr.txt"
write.table(my_expr,filename,sep="\t",quote = FALSE)

# 构建组别说明
coldata <- data.frame(
  condition = factor(c(
    rep("A", 3),
    rep("B", 3),
    rep("C", 3),
    rep("X", 3))))

# 指定对照组
coldata$condition <- relevel(coldata$condition, ref = "A")

# 差异表达分析
dds <- DESeqDataSetFromMatrix(countData = my_expr,
                              colData = coldata,
                              design = ~ condition)
dds <- DESeq(dds)


# 将结果保存到本地
prefix <- "abcx_wt_"
filename <- paste0(prefix,'DEG_preparation.Rdata')
save(my_expr,coldata,dds,
     file=filename)

