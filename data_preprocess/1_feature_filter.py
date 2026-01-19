import sys

# 從命令列參數取得資料集名稱
data_name = sys.argv[1]
#data_name = 'Appliances' #記得換其他資料集時要改回來
#data_name = 'Grocery_and_Gourmet_Food' #記得換其他資料集時要改回來
#data_name = 'Home_and_Kitchen' #記得換其他資料集時要改

# 開啟輸入的元資料檔案並讀取所有行 
#meta_file = open('/home/clara_r76121188/thesis/SR-Rec-master/data_preprocess/raw_data/meta_{}.json'.format(data_name)).readlines()
meta_file = open('./raw_data/meta_{}.json'.format(data_name)).readlines()

# 建立輸出檔案以儲存過濾後的元資料
#out_file = open('/home/clara_r76121188/thesis/SR-Rec-master/data_preprocess/tmp/filtered_meta_{}.json'.format(data_name), 'w')
out_file = open('./tmp/filtered_meta_{}.json'.format(data_name), 'w')

print("Filtering items with incomplete features...")

# 統計過濾後的商品總數
total_node_num = 0
# 保存cid2,cid3,price不同取值（儲存類別級別2、類別級別3和價格的唯一值集合）
feature_sets = [set() for _ in range(4)]

def is_float(s):
    """
    檢查字串是否可轉換為浮點數。
    
    參數:
        s: 要檢查的字串
        
    返回:
        如果字串是有效的浮點數則返回 True，否則返回 False
    """
    try:
        float(s)
        return True
    except ValueError:
        return False

# 過濾掉沒有完整價格資訊的商品
for eachline in meta_file:
    # 將每一行解析為字典
    data = eval(eachline)
    # 只處理至少有4級類別且具有價格欄位的商品
    if len(data['category']) >= 4 and 'price' in data:
        # 提取類別級別2和3（索引1和2）
        cid1, cid2, cid3 = data['category'][1:4]
        # 取得價格值
        price = data['price']
        # 跳過價格為空字串的商品
        if price== "":
            continue
        # 嘗試將價格轉換為浮點數（移除位置0的貨幣符號）
        if is_float(price[1:]):
            price = float(price[1:])
        else:
            # 跳過價格格式無效的商品
            continue
        
        #---------------疑似作者忘記刪除或輸出這個部分---------------------
        # 收集此商品的特徵
        features = [cid2, cid3, price]
        # 將特徵新增到對應的特徵集合
        for i in range(len(features)):
            feature_sets[i].add(features[i])
        #---------------疑似作者忘記刪除或輸出這個部分---------------------

        # 將有效的商品寫入輸出檔案
        out_file.write(eachline)
        # 遞增有效商品計數器
        total_node_num += 1

# 列印過濾後的商品總數
print('Total node num is {}'.format(total_node_num))

