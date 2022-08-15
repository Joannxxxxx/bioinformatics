## 高粱数据分析
```python
# 20220812更新
def get_outliers(data_ser, box_scale):
        """
        获得异常值的上下边界
        :param data_ser: 接收 pandas.Series 数据格式
        :param box_scale: 箱线图尺度，一般为 1.5
        """      
        iqr = data_ser.quantile(0.75) - data_ser.quantile(0.25)
        minimum = data_ser.quantile(0.25) - box_scale * iqr
        maximum = data_ser.quantile(0.75) + box_scale * iqr

        return (minimum,maximum)
    
def text_plot(rvalue,xing):
    rvalue = str(round(rvalue,2)) # 取两位小数
    words = "$r=" + "{" + rvalue + "}^" + xing + "$"
    #         words = "$\mathrm{r=" + "{" + rvalue + "}^" + xing + "}$"
    #         words = "$\mathrm{r=" + "{" + rvalue + "}^" + "{\star}" + "}$"
    print(words)
             #         plt.text(1, 1, equation,ha='center', va='center',transform=plt.gca().transAxes)  
    plt.text(0.01, 0.88, words, fontsize=30, transform=plt.gca().transAxes)
    
def plot_rvalue_and_pvalue(rvalue,pval):
    if pval < 0.01:
        xing = "{**}"
    elif pval < 0.05:
        xing = "{*}"
    else:
        xing = " "
        
    text_plot(rvalue,xing)

# 20220814更新
def plt_config(fontsize):
    """
    设置画图的参数，包括字体、字号等
    """  
    config = {
        "font.family": "serif",  # 使用衬线体
        "font.serif": ["SimHei"],  # 中易黑体
#         "font.serif": ["SimSun"],  # 中易宋体，在我电脑上不行
        "font.size": fontsize,  # 字号
        "axes.unicode_minus": False, #  # 解决保存图像负号'-'显示为方块的问题
        "mathtext.fontset": "stix",  # 设置 LaTeX 字体，stix 近似于 Times 字体
        }
    plt.rcParams.update(config)
        
def dropna_dropout_then_pair_cor(data,filepath,cols):
    """
    画两季性状两两之间的散点图，带相关系数
    :param data: 接收 pandas.DataFrame 数据格式
    :param filepath: 保存路径
    :param cols: 选择的性状
    """  
    from scipy.stats import pearsonr
    plt_config(25)
    
    tmp = data.copy()
    drop_dict = {}
    for col in cols:
        # 1、把同性状拿出来
        xname = col + "_x"
        yname = col + "_y"
        
        ind = data["品种编号"]
        x = data.loc[:,xname]
        y = data.loc[:,yname]
        xy = pd.concat([ind,x,y],axis=1) # 取出两年同性状的数据
        
        # 2、去掉缺失
        xna_list = data[data[xname].isnull()]["品种编号"].tolist() # xcol 的缺失样本
        yna_list = data[data[yname].isnull()]["品种编号"].tolist() # ycol 的缺失样本
        na_list = list(set(xna_list + yna_list))
        
        xy_dropna = xy[~xy["品种编号"].isin(na_list)] # 删掉缺失样本后
        
        # 3、计算同性状比值
        col_bi = col + "比"
        xy_dropna[col_bi] = xy_dropna[yname] / xy_dropna[xname]
        
        # 4、对比值进行离群点筛查
        minimun,maximun = get_outliers(xy_dropna[col_bi],1.5)
        pzbh_list = xy_dropna[(xy_dropna[col_bi] < minimun) | (xy_dropna[col_bi] > maximun)]["品种编号"].tolist() # 离群点的编号
        
        drop_dict[col] = {"xna_list": xna_list,
                          "xna_number":len(xna_list),
                         "yna_list":yna_list,
                          "yna_number":len(yna_list),
                         "na_list":na_list,
                          "na_number":len(na_list), 
                          "outliners": pzbh_list,
                         "outliners_number":len(pzbh_list),
                         "remain_number":data.shape[0] - len(na_list) - len(pzbh_list)       
                        } # 把缺失情况和离群点情况记录入字典
        
        # 5、在原数据上和同性状 xy 矩阵上删掉离群点
        tmp.loc[tmp["品种编号"].isin(pzbh_list),xname] = np.nan # 在原数据里把异常值删掉
        tmp.loc[tmp["品种编号"].isin(pzbh_list),yname] = np.nan
        xy_dropna_dropout = xy_dropna[~xy_dropna["品种编号"].isin(pzbh_list)] # 在单独拿出的同性状数据里删掉离群点
        xy_dropna_dropout = xy_dropna_dropout.set_index("品种编号")
        
        # 6、计算相关系数并画图
        sns.regplot(xname,yname,data=xy_dropna_dropout,color="b",scatter_kws={'alpha':0.3})
        
        xy_corr = xy_dropna_dropout.corr(method="pearson") # 计算相关系数矩阵
        rvalue = xy_corr.iloc[0,1]
        rvalue = round(rvalue,2) # 提取相关系数
        pval_matrix = xy_dropna_dropout.corr(method=lambda x, y: pearsonr(x, y)[1]) - np.eye(len(xy.columns)) # 计算 p 值矩阵
        pval = pval_matrix.iloc[0,1] # 提取 p 值
        plot_rvalue_and_pvalue(rvalue,pval) # 写上相关系数和显著性水平

        
        col = col.replace('/', 'sub')
        savepath = filepath + col + ".pdf" 
        plt.savefig(savepath, bbox_inches = 'tight') # 保存图片

        plt.show() # 展示图片
        
    drop_df = pd.DataFrame(drop_dict).T
    return tmp,drop_df
```

```python
# 20220812更新
def get_outliers(data_ser, box_scale):
        """
        获得异常值的上下边界
        :param data_ser: 接收 pandas.Series 数据格式
        :param box_scale: 箱线图尺度，一般为 1.5
        """      
        iqr = data_ser.quantile(0.75) - data_ser.quantile(0.25)
        minimum = data_ser.quantile(0.25) - box_scale * iqr
        maximum = data_ser.quantile(0.75) + box_scale * iqr

        return (minimum,maximum)
    
def text_plot(rvalue,xing):
    rvalue = str(round(rvalue,2)) # 取两位小数
    words = "$r=" + "{" + rvalue + "}^" + xing + "$"
    #         words = "$\mathrm{r=" + "{" + rvalue + "}^" + xing + "}$"
    #         words = "$\mathrm{r=" + "{" + rvalue + "}^" + "{\star}" + "}$"
    print(words)
             #         plt.text(1, 1, equation,ha='center', va='center',transform=plt.gca().transAxes)  
    plt.text(0.01, 0.88, words, fontsize=30, transform=plt.gca().transAxes)
    
def plot_rvalue_and_pvalue(rvalue,pval):
    if pval < 0.01:
        xing = "{**}"
    elif pval < 0.05:
        xing = "{*}"
    else:
        xing = " "
        
    text_plot(rvalue,xing)
    
def plt_config():
    """
    设置画图的参数，包括字体、字号等
    """  
    config = {
        "font.family": "serif",  # 使用衬线体
        "font.serif": ["SimHei"],  # 中易黑体
#         "font.serif": ["SimSun"],  # 中易宋体，在我电脑上不行
        "font.size": 25,  # 字号
        "axes.unicode_minus": False, #  # 解决保存图像负号'-'显示为方块的问题
        "mathtext.fontset": "stix",  # 设置 LaTeX 字体，stix 近似于 Times 字体
        }
    plt.rcParams.update(config)
        
def dropna_dropout_then_pair_cor(data,filepath,cols):
    """
    画两季性状两两之间的散点图，带相关系数
    :param data: 接收 pandas.DataFrame 数据格式
    :param filepath: 保存路径
    :param cols: 选择的性状
    """  
    from scipy.stats import pearsonr
    plt_config()
    
    tmp = data.copy()
    drop_dict = {}
    for col in cols:
        # 1、把同性状拿出来
        xname = col + "_x"
        yname = col + "_y"
        
        ind = data["品种编号"]
        x = data.loc[:,xname]
        y = data.loc[:,yname]
        xy = pd.concat([ind,x,y],axis=1) # 取出两年同性状的数据
        
        # 2、去掉缺失
        xna_list = data[data[xname].isnull()]["品种编号"].tolist() # xcol 的缺失样本
        yna_list = data[data[yname].isnull()]["品种编号"].tolist() # ycol 的缺失样本
        na_list = list(set(xna_list + yna_list))
        
        xy_dropna = xy[~xy["品种编号"].isin(na_list)] # 删掉缺失样本后
        
        # 3、计算同性状比值
        col_bi = col + "比"
        xy_dropna[col_bi] = xy_dropna[yname] / xy_dropna[xname]
        
        # 4、对比值进行离群点筛查
        minimun,maximun = get_outliers(xy_dropna[col_bi],1.5)
        pzbh_list = xy_dropna[(xy_dropna[col_bi] < minimun) | (xy_dropna[col_bi] > maximun)]["品种编号"].tolist() # 离群点的编号
        
        drop_dict[col] = {"xna_list": xna_list,
                          "xna_number":len(xna_list),
                         "yna_list":yna_list,
                          "yna_number":len(yna_list),
                         "na_list":na_list,
                          "na_number":len(na_list), 
                          "outliners": pzbh_list,
                         "outliners_number":len(pzbh_list),
                         "remain_number":data.shape[0] - len(na_list) - len(pzbh_list)       
                        } # 把缺失情况和离群点情况记录入字典
        
        # 5、在原数据上和同性状 xy 矩阵上删掉离群点
        tmp.loc[tmp["品种编号"].isin(pzbh_list),xname] = np.nan # 在原数据里把异常值删掉
        tmp.loc[tmp["品种编号"].isin(pzbh_list),yname] = np.nan
        xy_dropna_dropout = xy_dropna[~xy_dropna["品种编号"].isin(pzbh_list)] # 在单独拿出的同性状数据里删掉离群点
        xy_dropna_dropout = xy_dropna_dropout.set_index("品种编号")
        
        # 6、计算相关系数并画图
        sns.regplot(xname,yname,data=xy_dropna_dropout,color="b",scatter_kws={'alpha':0.3})
        
        xy_corr = xy_dropna_dropout.corr(method="pearson") # 计算相关系数矩阵
        rvalue = xy_corr.iloc[0,1]
        rvalue = round(rvalue,2) # 提取相关系数
        pval_matrix = xy_dropna_dropout.corr(method=lambda x, y: pearsonr(x, y)[1]) - np.eye(len(xy.columns)) # 计算 p 值矩阵
        pval = pval_matrix.iloc[0,1] # 提取 p 值
        plot_rvalue_and_pvalue(rvalue,pval) # 写上相关系数和显著性水平

        
        col = col.replace('/', 'sub')
        savepath = filepath + col + ".pdf" 
        plt.savefig(savepath, bbox_inches = 'tight') # 保存图片

        plt.show() # 展示图片
        
    drop_df = pd.DataFrame(drop_dict).T
    return tmp,drop_df
```

```python
# 20220812 更新
def data_standard(data):
    """
    标准化数据：减去均值再除以标准差
    data：有 index 的 DataFrame
    """  
    from sklearn.preprocessing import scale
    data_scaled = scale(data)
    data_scaled = pd.DataFrame(data_scaled, columns=data.columns,index=data.index)
    
    return data_scaled

def scale_and_cor_heatmap(data):
    # 标准化
    data_scale = data_standard(data)
    data_scale.head(2)

    #皮尔森相关系数
    pearson = data_scale.corr(method="pearson")

    sns.clustermap(pearson,
                   cmap="vlag",
    #                linewidths=.1,
                   figsize=(15, 15),               
                   annot=True, 
                   fmt="1.1f",                 
                   cbar_kws={"shrink": 0.25,"pad":0.01,}
                  )
```

```python
def get_outliner_index(data,index,cols):
    """
    获得异常值的索引名/编号
    :param data: 接收 pandas.DataFrame 数据格式
    :param index: 索引所在列的名字
    :param cols: 选择的性状
    """  
    out_dict = {}
    for col in cols:
        col = col + "比"
        minimun,maximun = get_outliers(data[col],1.2)
        pzbh_list = data[(data[col] < minimun) | (data[col] > maximun)][index].tolist() 
        out_dict[col] = pzbh_list
    #     out_dict[col] = {"outliners": pzbh_list}
    return out_dict
```

```python
def text_plot(rvalue,xing):
        """
        相关系数在图上的位置
        :param rvalue: 相关系数的数值
        :param xing: 星号
        """  
    rvalue = str(round(rvalue,2)) # 取两位小数
    words = "$r=" + "{" + rvalue + "}^" + xing + "$"
    #         words = "$\mathrm{r=" + "{" + rvalue + "}^" + xing + "}$"
    #         words = "$\mathrm{r=" + "{" + rvalue + "}^" + "{\star}" + "}$"
    print(words)
             #         plt.text(1, 1, equation,ha='center', va='center',transform=plt.gca().transAxes)  
    plt.text(0.01, 0.88, words, fontsize=33, transform=plt.gca().transAxes)
    
def plot_r(rvalue):
        """
        相关系数对应的星号个数
        :param rvalue: 相关系数的数值
        """  
    if abs(rvalue) > 0.402641: # 0.01 显著性水平的阈值
        xing = "{**}"
        text_plot(rvalue,xing)
    elif abs(rvalue) > 0.312006: # 0.05 显著性水平的阈值
        xing = "{*}"
        text_plot(rvalue,xing)
    else: # 不显著
        xing = " "
        text_plot(rvalue,xing)
        
def pair_cor(data,filepath,cols):
        """
        画两季性状两两之间的散点图，带相关系数
        :param data: 接收 pandas.DataFrame 数据格式
        :param filepath: 保存路径
        :param cols: 选择的性状
        """  
    for col in cols:
        xname = col + "_x"
        yname = col + "_y"
        x = data.loc[:,xname]
        y = data.loc[:,yname]

        sns.regplot(xname,yname,data=data,color="b",scatter_kws={'alpha':0.3})

        slope,intercept,rvalue,pvalue,stderr = scipy.stats.linregress(x = x, y = y)
        print(rvalue)
        plot_r(rvalue)

        col = col.replace('/', 'sub')
        savepath = filepath + col + ".pdf" 
        plt.savefig(savepath, bbox_inches = 'tight') # 保存图片

        plt.show() # 展示图片
```



```python
def box_plot_outliers(data_ser, box_scale):
        """
        利用箱线图去除异常值
        :param data_ser: 接收 pandas.Series 数据格式
        :param box_scale: 箱线图尺度，一般为 1.5
        """      
        iqr = data_ser.quantile(0.75) - data_ser.quantile(0.25)
        minimum = data_ser.quantile(0.25) - box_scale * iqr
        maximum = data_ser.quantile(0.75) + box_scale * iqr
        
#         rule_low = (data_ser < minimum)
#         rule_up = (data_ser > maximum)
        outliers = (data_ser < minimum) | (data_ser > maximum)
        
#         errors.append({'K': k, 'RMSE': error})
        threshold = {'minimum': minimum, 'maximun': maximum}
        return (outliers,threshold)
```
```python
def row_col_missing(df):
    """
    查看数据框行方向和列方向上的缺失情况
    df: 接收 pandas.DataFrame 数据格式，索引为样本，列为性状（一般来说）
    """ 
    rn = pd.DataFrame(df.isnull().sum(axis=1).sort_values(ascending=False))
    rn.columns = ["缺失数"]
    
    cn = pd.DataFrame(df.isnull().sum().sort_values(ascending=False))
    cn.columns = ["缺失数"]
    return rn,cn
    
rn,cn = row_col_missing(df)    
```

```python
def plt_config():
    """
    设置画图的参数，包括字体、字号等
    """  
    config = {
        "font.family": "serif",  # 使用衬线体
        "font.serif": ["SimHei"],  # 中易黑体
#         "font.serif": ["SimSun"],  # 中易宋体，在我电脑上不行
        "font.size": 25,  # 字号
        "axes.unicode_minus": False, #  # 解决保存图像负号'-'显示为方块的问题
        "mathtext.fontset": "stix",  # 设置 LaTeX 字体，stix 近似于 Times 字体
        }
    plt.rcParams.update(config)

def save_histgram(df,eng,filepath):  
    """
    绘制数据框每一列的直方图
    df: 接收 pandas.DataFrame 数据格式，没有索引，每一列均为变量
    eng: 性状表，包括性状对应的单位、英文全称、英文简称
    filepath: 保存图片的文件夹路径，以斜杠结尾
    """  
    plt_config() # 设置画图的全局参数

    cols = df.columns.tolist() # 画图
    for col in cols:
        sns.rugplot(df[col],color="k")
        plt.hist(df[col], facecolor = "w",edgecolor="k")


        t1 = eng.loc[col,"性状无单位"] # 标题三部分之一：性状，如株高
        t2 = eng.loc[col,"单位数学格式"] # 标题三部分之二：单位，格式为 $(cm)$
        t3 = eng.loc[col,"英文全称数学格式"] # 标题三部分之三：英文全称，格式为 $\mathrm{Plant\ height}$
        
        if col == "大小维管束平均面积比值":
            title = t1 + "\n" + t3 # 特殊情况，性状没有单位
        else:              
            title = t1 + t2 + "\n" + t3 # 一般情况性状都有单位
            
        plt.title(title) # 添加标题
    
        plt.xlabel(None) # 去掉 xlabel
        plt.xticks(fontproperties="Times new roman") # 设置 ticks 字体为 Times new roman
        plt.yticks(fontproperties="Times new roman")
       
        col = col.replace('/', 'sub')
        savepath = filepath + col + ".pdf" 
        plt.savefig(savepath, bbox_inches = 'tight') # 保存图片

        plt.show() # 展示图片
```


```python
def plt_config():
    """
    设置画图的参数，包括字体、字号等
    """  
    config = {
        "font.family": "serif",  # 使用衬线体
        "font.serif": ["SimHei"],  # 中易黑体
#         "font.serif": ["SimSun"],  # 中易宋体，在我电脑上不行
        "font.size": 25,  # 字号
        "axes.unicode_minus": False, #  # 解决保存图像负号'-'显示为方块的问题
        "mathtext.fontset": "stix",  # 设置 LaTeX 字体，stix 近似于 Times 字体
        }
    plt.rcParams.update(config)
    
def save_boxplot(df,eng,filepath):
    """
    绘制数据框每一列的箱线图
    df: 接收 pandas.DataFrame 数据格式，没有索引，每一列均为变量
    filepath: 保存图片的文件夹路径，以斜杠结尾
    """  
    plt_config() # 设置画图的全局参数
    
    cols = df.columns.tolist()  # 画图
    for col in cols:
        sns.boxplot(col,data=df,palette="Set2",showfliers=False)
        sns.swarmplot(col,data=df,color=sns.color_palette("Set2")[1])
        

        t1 = eng.loc[col,"性状无单位"] # 标题三部分之一：性状，如株高
        t2 = eng.loc[col,"单位数学格式"] # 标题三部分之二：单位，格式为 $(cm)$
        t3 = eng.loc[col,"英文全称数学格式"] # 标题三部分之三：英文全称，格式为 $\mathrm{Plant\ height}$
        
        if col == "大小维管束平均面积比值":
            title = t1 + "\n" + t3 # 特殊情况，性状没有单位
        else:              
            title = t1 + t2 + "\n" + t3 # 一般情况性状都有单位
            
        plt.title(title) # 设置 xlabel 字体
        plt.xlabel(None) # 去掉 xlabel
        plt.xticks(fontproperties="Times new roman") # 设置 ticks 字体为 Times new roman
        
        col = col.replace('/', 'sub') # 把性状名中的斜杠换成英文，以免被系统认为是路径
        savepath = filepath + col + ".png" 
        plt.savefig(savepath, bbox_inches = 'tight') # 保存图片

        plt.show() # 显示图片
```


```python
def string_split(string):
    """
    分割 df 列中格式为 字符+数字 的元素，比如 F18 将被分割成 F 和 8
    """  
    import re
    match = re.match(r"([a-z]+)([0-9]+)", string, re.I)
    if match:
        items = match.groups()
    return(items)
```

```python
def optimize_k(data,target,number):
    """
    用 KNN 填充缺失时选择最佳 K 值
    data: 接收 pandas.DataFrame 数据格式
    target: 用来当作因变量的变量
    number: N 的最大取值
    """   
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import mean_squared_error
    from sklearn.impute import KNNImputer
    
    rmse = lambda y, yhat: np.sqrt(mean_squared_error(y, yhat))

    errors = []
    for k in range(1, number, 1):
        imputer = KNNImputer(n_neighbors=k)
        imputed = imputer.fit_transform(data)
        df_imputed = pd.DataFrame(imputed, columns=data.columns)
        
        X = df_imputed.drop(target, axis=1)
        y = df_imputed[target]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model = RandomForestRegressor()
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        error = rmse(y_test, preds)
        errors.append({'K': k, 'RMSE': error})
        
    return errors
    
k_errors = optimize_k(data=d533, target='酸性洗涤纤维(%)_x',number=20)
k_errors
pf = pd.DataFrame(k_errors)
import matplotlib.pyplot as plt
plt.plot(pf["K"],pf["RMSE"])
    
def data_imputed(df,n):
    """
    用 KNN 填充缺失值
    df: 接收 pandas.DataFrame 数据格式
    """   
    from sklearn.impute import KNNImputer

    imputer = KNNImputer(n_neighbors=n)
    imputed = imputer.fit_transform(df)
    df_imputed = pd.DataFrame(imputed, columns=df.columns,index=df.index)
    return df_imputed
```

```python
# 20220729更新
def df_describe(df):
    """
    得出描述性统计
    :df: 接收 pandas.DataFrame 数据格式
    """      
    des = df.describe().T # 初步的描述性统计结果
    des["CV"] = 100 * des["std"]/ des["mean"] # 计算变异系数
    des = des.drop(["25%","75%"],axis=1) # 删掉四分位点
    
    skew = pd.DataFrame(df.skew(),columns={"skew"}) # 偏度计算
    kurt = pd.DataFrame(df.kurt(),columns={"kurt"}) # 峰度计算
    sk = pd.concat([skew,kurt],axis=1)
    
    des_final = pd.concat([des,sk],axis=1)
    
    des_final.columns = ["样本数","均值","标准差",
                   "最小值","中位数","最大值","变异系数 CV（%）",
                  "偏度","峰度"] # 进行中文命名
    
    return des_final
```

```python
def traits_hisgram(data,savepath):
    """
    对每一个变量画分布图
    data：没有 index 只有变量 cols 的 DataFrame
    savepath：图片保存的路径，以斜杠结尾
    """  
        
    import seaborn as sns
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    from pylab import mpl

    # style set 这里只是一些简单的style设置
    sns.set_style('darkgrid')
    sns.set_context(rc={'figure.figsize': (5, 4) } )
    
    cols = data.columns.tolist()
    for col in cols:

        sns.rugplot(data[col])
        plt.hist(data[col],alpha=0.5)

        #添加描述信息
        mpl.rcParams['font.sans-serif'] = ['SimHei'] # 指定默认字体
        xlabel = col + "分布图"
        plt.xlabel(xlabel,fontsize = 13)

        # 保存
        savename = savepath + col + ".png" 
        plt.savefig(savename)

        plt.show()
```

```python
def traits_boxplot(data,savepath):
    """
    对每一个变量箱线图图
    data：没有 index 只有变量 cols 的 DataFrame
    savepath：图片保存的路径，以斜杠结尾
    """  
    
    import seaborn as sns
    import matplotlib.pyplot as plt
    from pylab import mpl
    
    sns.set_style('white')
    
    cols = data.columns.tolist()
    for col in cols:
        sns.boxplot(data[col],palette="Set2",showfliers=False)
        sns.swarmplot(data[col],color=sns.color_palette("Set2")[1])
    
        mpl.rcParams['font.sans-serif'] = ['SimHei'] # 指定默认字体
        savename = savepath + col + ".png" 
        plt.savefig(savename)

        plt.show()
```

```python
def write2excel(df,savepath,sheet_name):
    """
    将数据写入已存在的 excel 表
    df：DataFrame
    savepath：excel 所在的路径
    sheet_name：表格名
    """
    import openpyxl

    wb = openpyxl.load_workbook(savepath)
    #如果有多个模块可以读写excel文件，这里要指定engine，否则可能会报错
    writer = pd.ExcelWriter(savepath,engine='openpyxl')
    #没有下面这个语句的话excel表将完全被覆盖
    writer.book = wb

    #如果有相同名字的工作表，新添加的将命名为Sheet21，如果Sheet21也有了就命名为Sheet22，不会覆盖原来的工作表
    df.to_excel(writer,sheet_name = sheet_name)
    writer.save()
    writer.close()
```

```python
def plot_dend(data,savepath):
    """
    绘制层次聚类图
    data：index 是 sample name，列是变量/性状的 DataFrame
    savepath：图片保存的路径
    """
    from pylab import mpl
    mpl.rcParams['font.sans-serif'] = ['SimHei'] # 指定默认字体

    from sklearn.preprocessing import normalize
    data_scaled = normalize(data)
    data_scaled = pd.DataFrame(data_scaled, columns=data.columns)

    import scipy.cluster.hierarchy as shc
    plt.figure(figsize=(20, 7))  
    plt.title("Dendrograms")  
    Z = shc.linkage(data_scaled,method='ward',metric='euclidean') 
    dend = shc.dendrogram(Z,labels=data.index)

    plt.xticks(fontsize=12)
    # plt.xticks(rotation=60)
    # plt.axhline(y=5.5, color='r', linestyle='--') # 画一条横线
    plt.savefig(savepath)
```

```python
def make_cluster(data,n_clusters):
    """
    聚类并将类别添加到 data 中为新一列
    data：index 是 sample name，列是变量/性状的 DataFrame
    n_clusters：想要聚成几类，数字
    """
    from sklearn.preprocessing import normalize
    data_scaled = normalize(data)
    data_scaled = pd.DataFrame(data_scaled, columns=data.columns)
    
    from sklearn.cluster import AgglomerativeClustering
    cluster = AgglomerativeClustering(n_clusters=n_clusters,linkage='ward',affinity='euclidean')  
    cluster_res = cluster.fit_predict(data_scaled)
    
    data["cluster"] = cluster_res
    data["cluster"] = data["cluster"] + 1
    
    return data
```

```python
def cluster_mean(data):
    """
    对聚类后的类别计算每类的样本数和性状均值
    data：index 是 sample name，最后一列是 cluster，中间列是变量/性状的 DataFrame
    """
    # 计算每类数量
    data_cc = pd.DataFrame(data["cluster"].value_counts())
    data_cc = data_cc.reset_index()
    data_cc.columns = ["cluster","数量"]
    data_cc = data_cc.sort_values("cluster").set_index("cluster")
    
    # 计算每类的性状均值
    data_cm = data.groupby("cluster").mean()
    # 合并数量与均值
    data_cf = pd.concat([data_cc,data_cm],axis=1)
    return data_cf
```

```python
def prebox(data):
    """
    画聚类类别箱线图之前的数据准备
    data：index 是 cluster，中间列是变量/性状的 DataFrame
    """  
    from sklearn.preprocessing import scale
    data_scaled = scale(data)
    data_scaled = pd.DataFrame(data_scaled, columns=data.columns,index=data.index)
    
    data_scaled_stack = data_scaled.stack().reset_index()
    data_scaled_stack.columns = ["cluster","traits","standardized_values"]
    
    return data_scaled_stack
```

```python
def cluster_boxplot(data,savepath):
    """
    画聚类类别箱线图
    data：经过 prebox 处理的 DataFrame
    """  
    from matplotlib.pyplot import figure
    # figure(figsize=(20, 6), dpi=100)
    figure(figsize=(20, 6))
    ax = sns.boxplot(x="traits", y="standardized_values", hue="cluster",
                       data=data, palette="Set2", dodge=True)
    plt.xticks(fontsize=12,rotation=40)
    plt.yticks(fontsize=12)

    ax.set(xlabel=None)
    ax.set_ylabel("standardized_values",fontsize=16)

    plt.savefig(savepath,bbox_inches = 'tight')

    plt.show()  
```

```python
def traits_cluster_boxplot(data,savepath):
    """
    对每一个变量按照类别画箱线图
    data：没有 index 只有变量 cols 的 DataFrame
    savepath：图片保存的路径，以斜杠结尾
    """  
    
    import seaborn as sns
    import matplotlib.pyplot as plt
    from pylab import mpl
    
    sns.set_style('white')
    
    cols = data.columns.tolist()
    for col in cols[1:]:
        sns.boxplot(x=data["cluster"],y=data[col],palette="Set2",showfliers=False)
        sns.swarmplot(x=data["cluster"],y=data[col],palette="Set2")
#         sns.swarmplot(x=data["cluster"],y=data[col],color=sns.color_palette("Set2")[1])
    
        mpl.rcParams['font.sans-serif'] = ['SimHei'] # 指定默认字体
        
        col = col.replace("/","sub")
        savename = savepath + col + ".png" 
        plt.savefig(savename)

        plt.show()
```


```python
def data_standard(data):
    """
    标准化数据：减去均值再除以标准差
    data：有 index 的 DataFrame
    """  
    from sklearn.preprocessing import scale
    data_scaled = scale(data)
    data_scaled = pd.DataFrame(data_scaled, columns=data.columns,index=data.index)
    
    return data_scaled
```    

```python
def cluster_plt(data,savepath):
    """
    画聚类类别性状均值的折线图
    data：经过标准化或其他 proprocessing 处理的 DataFrame，使不同量纲的数据处于可比范围
    savepath：图片保存的路径
    """  
    data = data_standard(data)
    
    import matplotlib.pyplot as plt
    colors = sns.color_palette() # 调色盘
    # 画每一行/类的折线图
    length = data.shape[0]
    for i in range(length):
        label = "cluster" + str(i+1)
        plt.plot(data.columns,data.iloc[i],c=colors[i],label=label)
    
    plt.legend()    
    plt.ylabel('standardized_values', fontsize=16)
    plt.xticks(fontsize=12,rotation=40)
    plt.savefig(savepath,bbox_inches = 'tight')
```
