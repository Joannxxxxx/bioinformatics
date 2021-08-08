### 20210808
#### 13:30-17:00
* 更新安装工具 Homebrew，更换镜像源教程[在此](https://blog.csdn.net/H_WeiC/article/details/107857302)
* 请教软件安装问题。加入 LDBlockShow 作者的 QQ 群 Reseqtools（群号：125293663）；在 stackoverflow 描述[问题](https://stackoverflow.com/questions/68698315/ld-symbols-not-found-for-architecture-x86-64-after-the-make-command)
* 用 LD block 确定 QTL 位点
  S1 确定距离范围，有 D’ 和 r^2 两种方法，在范围内的 SNP 划分为同一个 LD block 
  S2 与性状显著相关的 SNP 所在的 LD block 看作一个 QTL 位点
* 结果分析包括：
  1、表型描述性统计
  2、表型性状相关性分析，大维管束与小维管束相关性
  3、基因型分布，列一张表
  4、群体结构分析，
    用 admixture 分析结果 K=13，似乎不合理（太多了），而且原始数据也没有标籼稻粳稻的类别。
    主成分分析，选择2个主成分时，分成两簇，两簇之外其他点非常散乱。（可用 rMVP 的结果图）
    亲缘关系矩阵 kinship 暂时未知

### 20210807
#### 13:30-17:00
* 安装 LD 分析工具 [PopLDdecay](https://github.com/BGI-shenzhen/PopLDdecay) 遇到编译问题，未安装成功。尝试了改变编译方式、直接解压 jianguo 已编译完成的压缩包均未成功。
* 下载 LD 分析工具 [LDBlockShow](https://github.com/BGI-shenzhen/LDBlockShow/) 的预安装环境 [gcc](https://gcc.gnu.org/git.html)

