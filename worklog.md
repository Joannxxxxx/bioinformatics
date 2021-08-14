### 20210814
#### 13:30-17:30
* 关于 GWAS 结果的阈值选择<br>

背景：所有点都未通过 Bonferroni 阈值，因此进行阈值校正，以便进行下一步分析。GWAS 结果中最小 Pvalue 为 1.5E-7，进行 -log10() 转换后的 value（以下简称 log value）为 6.8，而 Bonferroni 的阈值是 9E-8（即 0.05/549008，其中 549008 为此次分析的 SNP 数量）log value 为7。<br>

两种 relaxed threshold：<br>
1、常见的阈值，Pvalue = 5E-5 即 log value = 4.3。在这个阈值下共有 105 个点，分布情况如下：<br>
chr   number
8     77
7      7
6      5
3      5
9      3
4      3
12     2
2      2
5      1

2、FDR 校正。选择了两个不同的 FDR 值：
  * 选择 FDR = 0.25（如果找到 100 个点，其中假阳性的点为 100✖️0.25=25 个。采用的 FDR 计算方式为 p✖️m/k，p 为 Pvalue，m 为实验次数即 SNP 个数，k 为 Pvalue 从小到达排列后的排名），找到 15 个点，分布在 8、6、4 号染色体上，分别有 11、3、1 个点位；
  * 选择 FDR = 0.3，找到 160 个点，分布情况如下：
    chr   number
    8     108
    7      11
    6       8
    4       7
    5       6
    12      5
    3       5
    9       3
    2       3
    1       3
    10      1


### 20210813
#### 13:30-17:30
* 看论文 Niu et al. BMC Genomics 了解单倍型分析的思路
* 尝试用 plink 进行单倍型分析，plink 提醒说自己的分析精度不高，建议使用其他软件包
* 关于单倍型分析的问题：如何提取基因的基因区 SNP ；选择 LDBlockShow 结果中的显著连锁区间作为输入，还是选择可能的基因区间作为输入；哪个软件包可用于单倍型分析（未搜索到主流信息）

### 20210812
#### 13:30-17:00
* 定位显著 SNP 点附近的基因
* 查看论文，几无收获

### 20210811
#### 13:30-18:00
* 使用新数据完成 GWAS 流程至 LDBlockShow
* 了解了连锁不平衡原理与算法(D' 和 r^2)，并且在 PopLDdecay 结果上具体查看（知行合一）。
* 研究了 PCA 的计算方式。PCA 的计算方式（之一）是对协方差矩阵进行特征分解，协方差矩阵是 (𝑋−𝜇)′(𝑋−𝜇) /(𝑛−1) ，公式里的减 𝜇（样本均值）过程就是 center。而 R 包 prcomp 里的 center = F 是对 𝐗′𝐗/(𝑛−1) 这个矩阵进行特征分解，得到的结果会有所不同，因此不该用 F 。此外，还可以通过 SVD 求解，暂不了解其推导过程。

### 20210810
#### 13:30-17:00
* 完成 PopLDdecay 和 LDBlockShow 做图流程。
* 更新下载工具 cpan，下载 SVG，似乎比 brew 好用。
* 下载新的基因型数据（29mio)，准备再做一次 GWAS 分析。

### 20210809
#### 19:00-20:30
* 从最近一次 GWAS 分析结果的 QQ 图看，自 p value = 1E-3 开始与 y=x 直线向上分离，可以证明曼哈顿图中的峰值点确有可能是我们感兴趣的点/来自自然选择（关于如何看曼哈顿图和 QQ 图见此文）
* 沈阳农业大学论文 2020 年，在 3.3 全基因组关联分析里列出了 22 个 QTL，此次分析的结果可与之比对相互验证（论文做 GWAS 分析时选择的阈值最低为 1E-5，这个阈值并不算严格）
* 先定位 QTL，然后根据水稻基因组功能注释网站得知该区间存在几个基因，对这些基因做单倍型分析，获得候选基因，并以前人的研究作为辅助验证

#### 13:30-17:00
* 安装 LDBlockShow 完成。
* 掌握了一些安装与环境变量的知识：
  Homebrew 是一种安装工具，通过 brew install 安装应用放在 /usr/local/Cellar/ 路径下。<br>
  在 ~/.bash_profile 文件中指定的环境变量包括 /usr/bin:/usr/loacl/bin:/usr/local/Cellar 这些，如果通过其他途径安装应用（不在上述路径），则需要在文件中设置环境变量。<br>
  查看环境变量的方式：echo $PATH （结果以冒号分隔）<br>
  设置环境变量的方式：export PATH = file/path:$PATH

### 20210808
#### 13:30-17:00
* 更新安装工具 Homebrew，更换镜像源教程[在此](https://blog.csdn.net/H_WeiC/article/details/107857302)
* 请教软件安装问题。加入 LDBlockShow 作者的 QQ 群 Reseqtools（群号：125293663）；在 stackoverflow 描述[问题](https://stackoverflow.com/questions/68698315/ld-symbols-not-found-for-architecture-x86-64-after-the-make-command)
* 用 LD block 确定 QTL 位点
  S1 确定距离范围，有 D’ 和 r^2 两种方法，在范围内的 SNP 划分为同一个 LD block <br>
  S2 与性状显著相关的 SNP 所在的 LD block 看作一个 QTL 位点<br>
* 结果分析包括：<br>
  1、表型描述性统计<br>
  2、表型性状相关性分析，大维管束与小维管束相关性<br>
  3、基因型分布，列一张表<br>
  4、群体结构分析，<br>
    用 admixture 分析结果 K=13，似乎不合理（太多了），而且原始数据也没有标籼稻粳稻的类别。<br>
    主成分分析，选择2个主成分时，分成两簇，两簇之外其他点非常散乱。（可用 rMVP 的结果图）<br>
    亲缘关系矩阵 kinship 暂时未知<br>

### 20210807
#### 13:30-17:00
* 安装 LD 分析工具 [PopLDdecay](https://github.com/BGI-shenzhen/PopLDdecay) 遇到编译问题，未安装成功。尝试了改变编译方式、直接解压 jianguo 已编译完成的压缩包均未成功。
* 下载 LD 分析工具 [LDBlockShow](https://github.com/BGI-shenzhen/LDBlockShow/) 的预安装环境 [gcc](https://gcc.gnu.org/git.html)

