import json
import numpy as np
import random
import sys

def pair_construct(item_dict, num_items, train_neg_num, test_neg_num, random_seed = 1):
    # {商品id:關聯buy商品id列表} ，商品數，訓練負樣本數2，測試負樣本數100
    """
    為訓練、驗證和測試集構造三元組樣本對。
    
    參數:
        item_dict: {商品id: 相關商品id列表} 的字典
        num_items: 總商品數
        train_neg_num: 訓練集中每個正樣本對應的負樣本數（預設為 2）
        test_neg_num: 測試/驗證集中每個樣本對應的負樣本數（預設為 100）
        random_seed: 隨機種子（預設為 1）
    
    返回:
        com_edge_index: 商品關聯邊列表 [[商品id, 相關商品id], ...]
        train_triple_pair: 訓練集三元組 [[商品id, 正樣本id, 負樣本id1, 負樣本id2], ...]
        val_triple_pair: 驗證集三元組 [[商品id, 正樣本id, 負樣本id1, ..., 負樣本id100], ...]
        test_triple_pair: 測試集三元組 格式同驗證集
    """
    random.seed(random_seed)
    all_items = np.arange(num_items)
    com_edge_index = []
    train_triple_pair = []
    val_triple_pair = []
    test_triple_pair = []

    # 為每個商品構造三元組樣本
    for key_item in item_dict.keys():
        # 此商品的相關商品列表（正樣本）
        pos_list = item_dict[key_item]
        # 此商品的非相關商品列表（負樣本）
        neg_list = list(set(all_items)-set(pos_list))
        # 採樣：訓練集每個正樣本 2 個負樣本，測試/驗證集各 100 個負樣本
        neg_sample = random.sample(neg_list, train_neg_num * len(pos_list) + 2 * test_neg_num)
        
        # 若正樣本數少於 3，則只放入訓練集（不放入驗證和測試集）
        if len(pos_list) < 3:
            for i in range(len(pos_list)):
                # 構造訓練三元組：[商品id, 正樣本id, 負樣本1, 負樣本2]
                tem = [int(key_item), int(pos_list[i])]
                com_edge_index.append([int(key_item), int(pos_list[i])])
                tem.extend(neg_sample[train_neg_num * i: train_neg_num * (i + 1)])
                train_triple_pair.append(tem)
            continue

        # 前 len(pos_list)-2 個正樣本放入訓練集
        for i in range(len(pos_list)-2):
            tem = [int(key_item), int(pos_list[i])]
            com_edge_index.append([int(key_item), int(pos_list[i])])
            tem.extend(neg_sample[train_neg_num * i: train_neg_num * (i+1)])
            train_triple_pair.append(tem)

        # 倒數第 2 個正樣本放入驗證集，對應 100 個負樣本
        tem = [ int(key_item), int(pos_list[len(pos_list)-2])]
        tem.extend(neg_sample[train_neg_num * len(pos_list):  train_neg_num * len(pos_list) + test_neg_num])
        val_triple_pair.append(tem)

        # 最後 1 個正樣本放入測試集，對應 100 個負樣本
        tem = [ int(key_item), int(pos_list[len(pos_list)-1])]
        tem.extend(neg_sample[train_neg_num * len(pos_list) + test_neg_num: ])
        test_triple_pair.append(tem)

    # 轉換為 numpy 陣列
    train_triple_pair = np.array(train_triple_pair)
    val_triple_pair = np.array(val_triple_pair)
    test_triple_pair = np.array(test_triple_pair)
    # com_edge_index: [[商品id1，关联的buy商品id1],[商品id1，关联的buy商品id2],...]
    # train_triple_pair: [[商品id1，关联的buy商品id1，非关联的buy商品id3，非关联的buy商品id4]，[商品id1，关联的buy商品id2，非关联的buy商品id5，非关联的buy商品id6]]
    # val_triple_pair: [[商品id1，关联的buy商品id1，非关联的buy商品id3,...,非关联的buy商品id102]，[商品id1，关联的buy商品id2，非关联的buy商品id5,...,非关联的buy商品id104]]
    # test_triple_pair和val_triple_pair的格式一致，不同的是对负样本的选择，且两者都不将关联buy样本数少于3的商品放入(但train中是放入的)
    return com_edge_index, train_triple_pair, val_triple_pair, test_triple_pair

def main():
    print("Generating training, validation and test sets...")

    # 儲存商品 ID 到相關商品列表的映射
    com_dict = {}
    # 儲存商品的特徵
    features = []

    # 讀取 JSON 格式的商品資料
    with open(dataset_path, encoding='utf-8') as f:
        while True:
            line = f.readline()
            if not line:
                break
            # 解析 JSON 行
            js = json.loads(line)
            # 獲得鄰接節點信息
            neighbors = js['neighbor']
            # 獲得商品 ID
            key_item = js['node_id']
            # 獲得相關商品列表（類型 1 表示 also_buy 邊）
            com_list = list(neighbors["1"].keys())
            # 儲存商品的相關商品列表
            # {商品id:關聯buy商品id列表}
            com_dict[key_item] = com_list
            # 儲存商品的特徵：[分類 2 ID, 分類 3 ID, 價格]
            # [標籤1下標，標籤2下標，價格下標]
            features.append(list(js['uint64_feature'].values()))

    # 將特徵列表轉換為 numpy 陣列
    features = np.squeeze(np.array(features))
    # 獲得商品總數
    num_items = features.shape[0]
    #{商品id:關聯buy商品id列表} ，商品數，訓練負樣本數2，測試負樣本數100
    # 構造三元組訓練、驗證和測試集
    train_com_edge_index, train_set, val_set, test_set = pair_construct(com_dict, num_items, train_neg_num, test_neg_num)

    # 儲存特徵、邊和各個資料集到 NPZ 檔案
    np.savez(save_path, features = features, com_edge_index = train_com_edge_index, train_set = train_set,
             val_set = val_set, test_set = test_set)

    # 列印各個資料集的大小統計
    print("train set num: {}; validation set num: {}; test set num: {}".
          format(len(train_set), len(val_set), len(test_set)))

if __name__ == '__main__':
    # 從命令列參數取得資料集名稱
    data_name = sys.argv[1]

    # 構造輸入資料檔案路徑
    dataset_path = "./data/{}.json".format(data_name)
    # 構造輸出檔案路徑
    save_path = "./processed/{}.npz".format(data_name)

    # 訓練集中每個正樣本對應的負樣本數
    train_neg_num = 2
    # 測試/驗證集中每個樣本對應的負樣本數
    test_neg_num = 100
    
    main()









