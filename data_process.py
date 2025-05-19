import json
import pandas as pd
import time

def read_json_file(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
            return data
    except FileNotFoundError:
        print(f"错误：文件 {file_path} 未找到")
        return None
    except json.JSONDecodeError:
        print(f"错误：文件 {file_path} 不是有效的JSON格式")
        return None

data=read_json_file("/home/ylqiu/datamining/product_catalog.json")
product_list=data['products']


def get_category(x):
    cate_list=set()
    for item in x:
        id=item['id']
        for j in product_list:
            if j['id']==id:
                category=j['category']
                if category in "智能手机、笔记本电脑、平板电脑、智能手表、耳机、音响、相机、摄像机、游戏机":
                    cate_list.add("电子产品")
                elif category in "上衣、裤子、裙子、内衣、鞋子、帽子、手套、围巾、外套":
                    cate_list.add("服装")
                elif category in "零食、饮料、调味品、米面、水产、肉类、蛋奶、水果、蔬菜":
                    cate_list.add("食品")
                elif category in "家具、床上用品、厨具、卫浴用品":
                    cate_list.add("家居")
                elif category in "文具、办公用品":
                    cate_list.add("办公")
                elif category in "健身器材、户外装备":
                    cate_list.add("运动户外")
                elif category in "玩具、模型、益智玩具":
                    cate_list.add("玩具")
                elif category in "婴儿用品、儿童课外读物":
                    cate_list.add("母婴")
                elif category in "车载电子、汽车装饰":
                    cate_list.add("汽车用品")
                break
    return list(cate_list)
time1=time.time()
data = pd.DataFrame()
file_path = "/home/ylqiu/datamining/part-00000.parquet"
df = pd.read_parquet(file_path, engine='pyarrow')
df['item_list'] = df['item_list'].apply(lambda x: get_category(x))
df.to_parquet("/home/ylqiu/datamining_filter/part-00000simpleRQ1.parquet")
print(time.time()-time1)