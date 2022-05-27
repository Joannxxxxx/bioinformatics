# 清除环境
rm(list=ls()) 

# 设置工作路径
wd <- "your/working/directory/"
setwd(wd) # 设置工作路径
getwd() # 查看工作路径，确认无误后继续

# 读取差异基因
case <- "B"
filename <- paste0(case,"vsA_sig_gene.txt")
diff_gene <- read.table(filename)

# 用 riceidconverter 和 AnnotationHub 转换ID
library(riceidconverter)
library(AnnotationHub)
# 获取水稻注释数据
hub <- AnnotationHub()
query(hub, "oryza sativa")
rice <- hub[['AH94061']]
OrgDb <- rice
# 将 RAP 转为 SYMBOL
symbol_map <- RiceIDConvert(myID = diff_gene,
                            fromType = 'RAP',
                            toType = 'SYMBOL')
# 去掉空值
symbol_map[symbol_map=="None"] <- NA
symbol_map[symbol_map=="#N/A"] <- NA
symbol_gene <- as.character(unique(na.omit(symbol_map$SYMBOL)))
# 将 SYMBOL 转为 Entrez
entrez_hub_gene <- mapIds(x=OrgDb,
                          keys=symbol_gene,
                          keytype = "SYMBOL",
                          column = "ENTREZID")
gene <- as.character(entrez_hub_gene)

# 创建 symbol 和 entrez 的 map 表
symbol_entrez <- na.omit(data.frame(symbol_gene,gene))
head(symbol_entrez)
colnames(symbol_entrez) <- c("SYMBOL","ENTREZ")
head(symbol_entrez)

# 创建差异基因的三种基因编号匹配表
diff_gene_map <- merge(symbol_entrez,symbol_map,by="SYMBOL",all.x=T)
head(diff_gene_map)
diff_gene_map_clean <- diff_gene_map[!duplicated(diff_gene_map$ENTREZ),]

# 载入需要的程序包
library(clusterProfiler) # 进行GO富集和KEGG富集
library(pathview)
library(ggplot2) #绘图  

# GO 富集分析
onts <- c('ALL','CC','BP','MF')

for (ont in onts){
  go <- enrichGO(gene,
                 OrgDb = OrgDb, 
                 ont=ont,
                 pAdjustMethod = 'BH',
                 pvalueCutoff = 0.01, 
                 qvalueCutoff = 0.05,
                 keyType = 'ENTREZID',
                 readable = TRUE)#进行GO富集，确定P值与Q值得卡值并使用BH方法对值进行调整。
  print(ont)
  print(dim(go))
  
  # 画气泡图
  filename <- paste0(case,"vsA_go_",ont,"_dot.pdf")
  g <- dotplot(go)
  ggsave(g,filename = filename)
}

# 放松阈值条件
for (ont in onts){
  go <- enrichGO(gene,
                 OrgDb = OrgDb, 
                 ont=ont,
                 pAdjustMethod = 'BH',
                 #pvalueCutoff = 0.01, 
                 qvalueCutoff = 0.1,
                 keyType = 'ENTREZID',
                 readable = TRUE)#进行GO富集，确定P值与Q值得卡值并使用BH方法对值进行调整。
  print(ont)
  print(dim(go))
  
  # 画气泡图
  filename <- paste0(case,"vsA_go_",ont,"_dot_loose.pdf")
  g <- dotplot(go)
  ggsave(g,filename = filename)
  
}

# KEGG 富集分析
kegg <- enrichKEGG(gene,
                   organism = 'osa',
                   keyType = 'kegg',
                   pvalueCutoff = 0.01,
                   qvalueCutoff = 0.05,
                   pAdjustMethod = 'BH')#进行KEGG富集
dim(kegg)

# 画气泡图
filename <- paste0(case,"vsA_kegg_dot.pdf")
g <- dotplot(kegg)
ggsave(g,filename = filename)

#pathway映射
# KEGG 通路查询网站：https://www.genome.jp/kegg/pathway.html
# 读取 DEG 分析结果，提取 log2FC 和 padj，这两列是 pathway 画图里的值
filename <- paste0(case,'vsA_DEG_result.txt')
deg_result <- read.table(file=filename)
head(deg_result)

deg_res_select <- data.frame(rownames(deg_result),
                     deg_result$log2FoldChange,
                     deg_result$padj)
head(deg_res_select)
colnames(deg_res_select ) <- c("RAP","log2FC","padj")
head(deg_res_select )

# 将 DEG 结果和差异基因匹配，连接键是 RAP
diff_deg_res <- merge(deg_res_select,diff_gene_map_clean,by="RAP")
head(diff_deg_res)

# 选择差异基因 DEG 分析结果中的特定列来画图
data <- data.frame(diff_deg_res$log2FC,
                   row.names = diff_deg_res$ENTREZ)

# 画特定通路图
name <- paste0(case,"vsA")
osa00904 <- pathview(gene.data = data,
                     pathway.id = "osa00904", #更改
                     species = "osa",
                     out.suffix = name,
                     #limit = list(gene=max(abs(geneList)), cpd=1)
)

