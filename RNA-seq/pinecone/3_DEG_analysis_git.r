# 代码主体来自生信技能树：https://github.com/jmzeng1314/GEO 以及其他教程
# 我略做了修改，使步骤更加清晰

# 清除环境
rm(list=ls()) 

# 获取 sh 脚本文件中的参数
args<-commandArgs(TRUE)

wd <- args[1] # 工作路径
prefix <- args[2] # DEG_preparation 结果文件的前缀
control <- args[3] # 对照组组名
case <- args[4] # 实验组组名

setwd(wd) # 设置工作路径
getwd() # 查看工作路径，确认无误

wd_slash <- paste0(wd,"/") 

# 读取数据
# prefix <- "abcx_wt_"
filename <- paste0(prefix,'DEG_preparation.Rdata')
load(file=filename)

# 载入需要的程序包
library(corrplot)
library(pheatmap)
library(DESeq2)

# 指定对照组和实验组
# control <- "A" 
# case <- "B" 

# 获得差异表达分析结果
res <- results(dds,contrast=c("condition",case,control))
res <- res[complete.cases(res),] # 删去空值，等同于 na.omit
head(res)

# 将结果按照 padj 排序
resOrdered <- res[order(res$padj),]
resOrdered <- as.data.frame(resOrdered)
head(resOrdered) 

# 保存差异表达分析结果
filename <- paste0(case,'vs',control,'_DEG_result.txt')
write.table(resOrdered,filename,sep="\t")

# 将 padj < 0.05 且 |fold-change|>2 的基因界定为差异表达基因
sig <- resOrdered[!is.na(resOrdered$padj) &
                    resOrdered$padj<0.05 &
                    abs(resOrdered$log2FoldChange)>=1,]
sig_gene <- rownames(sig)

# 保存差异基因
filename <- paste0(case,'vs',control,'_sig_gene.txt')
write.table(sig_gene,filename,
            row.names = FALSE,col.names = FALSE,quote = FALSE)

# 画差异表达基因 top50 的热图
library(pheatmap)
choose_gene <- head(rownames(resOrdered),50) # 选择差异显著前 50 的基因
choose_matrix <- my_expr[choose_gene,]
choose_matrix <- t(scale(t(choose_matrix))) # 标准化
filename <- paste0(case,'vs',control,'_DEG_top50_heatmap.png')
pheatmap(choose_matrix,filename=filename)

# 画火山图
# 设置阈值：如何才算差异表达基因
# 一般主观认为 logFC > 1 也就是 |fold-change|>2 为差异表达
# 也可以使用其他的统计方法进行界定，如下两行
# logFC_cutoff <- with(resOrdered,mean(abs(log2FoldChange)) + 2*sd(abs(log2FoldChange)))
# resOrdered$change = as.factor(ifelse(resOrdered$pvalue < 0.05 & abs(resOrdered$log2FoldChange) > logFC_cutoff,
#                                    ifelse(resOrdered$log2FoldChange > logFC_cutoff ,'UP','DOWN'),'NOT')
# )
# 这里使用 logFC_cutoff=1 为阈值
logFC_cutoff=1
resOrdered$change = as.factor(ifelse(resOrdered$padj < 0.05 & abs(resOrdered$log2FoldChange) > logFC_cutoff,
                                   ifelse(resOrdered$log2FoldChange > logFC_cutoff ,'UP','DOWN'),'NOT')
)
tail(resOrdered)
this_tile <- paste0('Cutoff for logFC is ',round(logFC_cutoff,3),
                    '\nThe number of up gene is ',nrow(resOrdered[resOrdered$change =='UP',]) ,
                    '\nThe number of down gene is ',nrow(resOrdered[resOrdered$change =='DOWN',])
)
library(ggplot2)
g = ggplot(data=resOrdered, 
           aes(x=log2FoldChange, y=-log10(pvalue), 
               color=change)) +
  geom_point(alpha=0.4, size=1.75) +
  theme_set(theme_set(theme_bw(base_size=20)))+
  xlab("log2 fold change") + ylab("-log10 p-value") +
  ggtitle( this_tile ) + theme(plot.title = element_text(size=15,hjust = 0.5))+
  scale_colour_manual(values = c('blue','black','red')) ## corresponding to the levels(res$change)
print(g)
filename <- paste0(case,'vs',control,'_volcano.png')
ggsave(g,filename = filename)
