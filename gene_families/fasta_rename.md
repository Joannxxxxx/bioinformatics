```python
def fasta2dict(inf):
    """
    作者：徐诗芬
    功能：将序列读取成字典，根据特定序列ID提取字典里的序列
    日期：2021.1.9
    """
    # 按行读取序列
    # 输入fasta文件，返回名称，序列
    global name
    dict = {}
    for line in inf:
        line = line.strip()
        if line.startswith('>'):
            name = line
            dict[name] = ''
        else:
            dict[name] += line
    return dict
    

def fasta_rename(wd,org_fasta_name,name_map_use,name_map_key,name_map_value,new_fasta_name):
    """
    函数功能：修改 fasta 文件里的基因名
    wd：工作目录
    org_fasta_name：原始 fasta 文件名
    name_map_use：原始基因名和新基因名的映射 dataframe
    name_map_key：映射表中的 key，也就是原始基因名
    name_map_value：映射表中的 value，也就是新基因名
    new_fasta_name：新 fasta 文件名
    """ 
    
    # 0、载入库
    import pandas as pd
    
    # 1、读入原始序列文件
    org_filepath = wd + org_fasta_name
    with open(org_filepath, "r", encoding="utf-8") as f:
        prot_seq = f.readlines()
        
    prot_dict = fasta2dict(prot_seq) # 将序列转为字典格式
    prot_df = pd.DataFrame([prot_dict]).T.reset_index() # 将字典转为 dataframe
    prot_df.columns = ["name","sequence"] # 将列重命名
   
    copy_df = prot_df.copy() # 复制一份，接下来在副本上修改
    copy_df['name'] = copy_df['name'].str[1:] # 将 name 列中的基因名拆解出来
#     copy_df.head()
    
    # 2、与 name_map 合并
    copy_map = pd.merge(copy_df,name_map_use,
                        left_on='name',right_on=name_map_key)
    
    # 3、用新名字书写序列
    new_filepath = wd + new_fasta_name
    with open(new_filepath, "a", encoding="utf-8") as f:
        for i in range(copy_map.shape[0]):
            rename = ">" + copy_map.loc[i,name_map_value]
            seq = copy_map.loc[i,'sequence']
            f.write(rename + "\n")
            f.write(seq + "\n")
```
