import json
import sys


def assign_num_id(u, node_to_num_id): 
  """為每個節點分配唯一的數字 ID"""
  if u not in node_to_num_id:
    node_to_num_id[u] = len(node_to_num_id)
  return node_to_num_id[u]


def init_node(num_id):
    """
    初始化節點結構。
    
    返回：包含節點資訊的字典，包括節點 ID、類型、權重、鄰居和邊
    """
    return {
        "node_id": num_id,
        "node_type": 0,
        "node_weight": 1.0,
        # neighbor 中 "0" 代表相似邊（view），"1" 代表相關邊（buy），"2"、"3" 預留
        "neighbor": {"0": [], "1": [], "2": [], "3": []},
        # 節點特徵：包含分類 ID 和價格
        "uint64_feature": {},
        "edge":[]
    }


def add_neighbor(u, v, edge_type, node_data):
    """將鄰接節點 v 新增到節點 u 的指定邊類型鄰接表中"""
    node_data[u]['neighbor'][str(edge_type)].append(v)


def add_edge(u, v, edge_type, node_data):
  """將邊資訊新增到源節點 u 的邊列表中"""
  uv_edge = {
            "src_id": u,
            "dst_id": v,
            "edge_type": edge_type,
            "weight": 1.0,
  }
  node_data[u]['edge'].append(uv_edge)


def fill_node_features(node_data, node_to_num_id, valid_node_asin):
  """
  為有效的節點填充特徵資訊（分類、價格）。
  
  同時輸出分類 ID 對應字典到檔案
  """
  def id_mapping(s, id_dict):
    """為元素建立或查詢 ID 映射"""
    if s not in id_dict:
        id_dict[s] = len(id_dict)
    return id_dict[s]

  # 分別儲存類別級別 1、2、3 的 ID 映射字典
  cid1_dict, cid2_dict, cid3_dict = {}, {}, {}

  # 從元資料檔案中提取每個商品的分類和價格特徵
  for eachline in meta_file:
    data = eval(eachline)
    asin = data['asin']
    # 只處理在有效節點集合中的商品
    if asin in valid_node_asin:
      c1, c2, c3 = data['category'][1:4]

      price = data['price']
      # 獲得商品分類的 ID 映射
      cid1, cid2, cid3 = id_mapping(c1, cid1_dict), id_mapping(c2, cid2_dict), \
                         id_mapping(c3, cid3_dict)

      # 獲得商品的數字 ID
      num_id = node_to_num_id[asin]

      # 存儲節點的特徵：分類 2 ID、分類 3 ID 和價格
      node_data[num_id]['uint64_feature'] = {"0": [cid2], "1": [cid3], "2": [float(price[1:])]}

  # 將分類 2 的 ID 映射寫入檔案
  for asin, num_id in cid2_dict.items():
    cid2_dict_file.write("{}\t{}\n".format(num_id, asin))
  # 將分類 3 的 ID 映射寫入檔案
  for asin, num_id in cid3_dict.items():
    cid3_dict_file.write("{}\t{}\n".format(num_id, asin))

  # 記錄特徵統計資訊
  feature_stats = "#cid2: {}; #cid3: {};".format(len(cid2_dict), len(cid3_dict))
  print(feature_stats)
  log_file.write(feature_stats + '\n')

def price_string(price):
  """將價格字串轉換為浮點數（移除貨幣符號）"""
  price = price[1:]
  return float(price)

def main():
  print("Converting node data to json format...")
  
  # 存儲 ASIN 到數字 ID 的映射
  node_to_num_id = {}
  # 存儲所有節點的資料結構
  node_data = {}

  # 收集所有在邊中出現的有效節點
  valid_node_asin = set()

  # 處理相似邊（also_view）
  for eachline in sim_edges_file:
    u, v, w = eachline.strip('\n').split('\t') 
    valid_node_asin.add(u)
    valid_node_asin.add(v)
    # 為節點分配數字 ID
    uid = assign_num_id(u, node_to_num_id) 
    vid = assign_num_id(v, node_to_num_id) 

    if uid not in node_data:
      # neighbor 中 "0" 代表相似邊（view），"1" 代表相關邊（buy），"2"、"3" 預留
      # uint64_feature 包含分類 ID 和價格特徵
      # 初始化節點資料結構（如果還未初始化）
      node_data[uid] = init_node(uid)
    if vid not in node_data:
      node_data[vid] = init_node(vid)


    # node_data[uid]['neighbor']['0'].append(vid)
    # 將view下的邊類別設為0
    # 新增相似邊的鄰接關係（邊類型 0 代表相似）
    add_neighbor(uid, vid, 0, node_data)
    add_neighbor(vid, uid, 0, node_data)

  # 處理相關邊（also_buy）
  for eachline in cor_edges_file:
    u, v, w = eachline.strip('\n').split('\t') 
    valid_node_asin.add(u)
    valid_node_asin.add(v)
    # 為節點分配數字 ID
    uid = assign_num_id(u, node_to_num_id) 
    vid = assign_num_id(v, node_to_num_id) 

    # 初始化節點資料結構（如果還未初始化）
    if uid not in node_data:
      node_data[uid] = init_node(uid) 
    if vid not in node_data:
      node_data[vid] = init_node(vid)

    #node_data[u]['neighbor'][str(edge_type)].append(v)
    # 將buy下的邊類別設為1
    # 新增相關邊的鄰接關係（邊類型 1 代表相關）
    add_neighbor(uid, vid, 1, node_data)
    add_neighbor(vid, uid, 1, node_data)

  # 填充節點的特徵資訊
  fill_node_features(node_data, node_to_num_id, valid_node_asin)

  # 統計相似邊和相關邊的數量
  sim_num, cor_num = 0, 0

  # 遍歷所有節點並生成最終的 JSON 格式輸出
  for u in sorted(node_data.keys()):
    u_data = node_data[u]
    u_neighbor = node_data[u]['neighbor']

    # 統計此節點的相似和相關邊數
    sim_num += len(u_neighbor['0']) #獲得商品的view邊數
    cor_num += len(u_neighbor['1']) #獲得商品的buy邊數

    for edge_type in [0, 1]:
        # 將node_data中的'edge'賦值為"src_id", "dst_id", "edge_type", "weight"
        # 根據鄰接表建立邊資訊
        for v in u_data['neighbor'][str(edge_type)]:
            uid, vid = u, v
            add_edge(uid, vid, edge_type, node_data)
        # 將鄰接列表轉換為字典格式（key 為鄰接節點，value 為邊權重）
        u_data['neighbor'][str(edge_type)] = {
                v: 1.0 for v in u_data['neighbor'][str(edge_type)]}
    # 將節點資料輸出為 JSON 格式
    json.dump(u_data, out_file)
    out_file.write('\n')


  # 統計並記錄圖的統計資訊
  node_stats = "Total node num: {}".format(len(node_data)) 
  edge_stats = "Total edge num: {}; sim: {}; cor: {}".format(sim_num + cor_num, sim_num, cor_num)
  data_stats = "{}\n{}".format(node_stats, edge_stats)

  print(data_stats)
  log_file.write(data_stats)

  # 將節點的 ASIN 與數字 ID 的對應關係寫入檔案
  for asin, num_id in node_to_num_id.items():
    id_dict_file.write("{}\t{}\n".format(asin, num_id))



if __name__ == '__main__':
  # 從命令列參數取得資料集名稱
  data_name = sys.argv[1]

  # 構造相似邊和相關邊的檔案名稱
  sim_filename = "filtered_{}_sim.edges".format(data_name)
  cor_filename = "filtered_{}_cor.edges".format(data_name)

  # 開啟相似邊和相關邊的檔案
  sim_edges_file = open("./tmp/" + sim_filename, 'r') 
  cor_edges_file = open("./tmp/" + cor_filename, 'r') 

  # 開啟過濾後的元資料檔案
  meta_file = open('./tmp/filtered_meta_{}.json'.format(data_name)).readlines()

  # 統計資訊輸出檔案（記錄分類數、節點數、邊數等）
  log_file = open("./stats/{}.log".format(data_name), 'w')
  
  # 商品 ASIN 與數字 ID 的對應檔案
  id_dict_file = open('./tmp/{}_id_dict.txt'.format(data_name), 'w')
  
  # 分類 2 的 ID 映射檔案
  cid2_dict_file = open('./tmp/{}_cid2_dict.txt'.format(data_name), 'w')
  
  # 分類 3 的 ID 映射檔案
  cid3_dict_file = open('./tmp/{}_cid3_dict.txt'.format(data_name), 'w')
  
  # 最終的 JSON 格式資料檔案（包含所有節點和邊資訊）
  out_file = open("./data/{}.json".format(data_name), 'w')
  # 執行主流程
  main()

