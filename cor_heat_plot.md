```
# 绘制相关系数热力图
# 清除环境
rm(list=ls()) 

# 设置工作路径
wd <- "your/working/directory"
setwd(wd) # 设置工作路径
getwd() # 查看工作路径，确认无误后继续

# 加载包
library(readxl)
library(showtext)
library(pheatmap)
library(ggplot2)

# 设置可以显示中文字体
showtext.auto(enable = TRUE)
font.add('Songti', '/Library/Fonts/Songti.ttc') # 第二个参数是你自己的字体文件所在路径

# 读入数据
data <- read_excel("data.xlsx",sheet=1,na="NA")
# 计算相关系数
data_cor <- cor(log2(data+1))

# 作图
g <- pheatmap(data_cor,
              display_numbers = TRUE
              # clustering_distance_rows = "correlation",
              #scale = "row",
              # display_numbers = T,
              #cluster_rows = F,
              #cluster_cols = F
)
filename <- "your_filename.pdf"
ggsave(g,filename = filename)
```
