import sys
from collections import defaultdict


def main():
  # 從命令列參數取得資料集名稱
  data_name = sys.argv[1]
  
  # 開啟過濾後的元資料檔案
  f = open("./tmp/filtered_meta_{}.json".format(data_name), 'r').readlines()
  
  # 儲存相似商品邊及其權重
  sim_edges = defaultdict(int)
  
  # 儲存相關商品邊及其權重
  rel_edges = defaultdict(int)

  print("Filtering items with not sub/comp edges...")

  # 收集所有有效的商品 ASIN
  all_asin = set()
  for eachline in f:
    each_data = eval(eachline)
    asin = each_data['asin']
    all_asin.add(asin)

  # 處理每個商品的相似和相關關聯
  for eachline in f:
    each_data = eval(eachline)
    asin = each_data['asin']

    # 跳過沒有相似或相關商品資訊的商品
    if ('also_view' not in each_data.keys()) or ('also_buy' not in each_data.keys()):
      continue
    
    # 處理 also_view（相似商品）邊
    for rid in each_data['also_view']:
      # 只保留兩個端點都在資料集中的邊
      if rid not in all_asin: continue
      u, v = str(asin), str(rid)
      # 標準化邊的方向（較小的 ASIN 在前）
      if u > v: u, v = v, u
      edge = (u, v)
      # 增加邊的權重（表示關聯強度）
      sim_edges[edge] += 1

    # 處理 also_buy（相關商品）邊
    for rid in each_data['also_buy']:
      # 只保留兩個端點都在資料集中的邊
      if rid not in all_asin: continue
      u, v = str(asin), str(rid)
      # 標準化邊的方向（較小的 ASIN 在前）
      if u > v: u, v = v, u
      edge = (u, v)
      # 增加邊的權重（表示關聯強度）
      rel_edges[edge] += 1
    
  # 建立相似商品邊的輸出檔案
  fout_sim = open("./tmp/{}_sim.edges".format(data_name), 'w') 
  # 建立相關商品邊的輸出檔案
  fout_rel = open("./tmp/{}_cor.edges".format(data_name), 'w') 

  # 追蹤相似和相關邊的最大權重
  max_sim_weight = 0
  max_rel_weight = 0
  # 收集圖中的所有節點
  all_nodes = set()
  
  # 寫入相似商品邊
  for (u, v), w in sim_edges.items():
    fout_sim.write('\t'.join([u, v, str(w)]) + '\n')
    max_sim_weight = max(max_sim_weight, w)
    all_nodes.add(u)
    all_nodes.add(v)
  
  # 寫入相關商品邊
  for (u, v), w in rel_edges.items():
    fout_rel.write('\t'.join([u, v, str(w)]) + '\n')
    max_rel_weight = max(max_rel_weight, w)
    all_nodes.add(u)
    all_nodes.add(v)
  
  # 列印統計資訊
  print("max_sim_weight: {}; max_rel_weight: {}; node num: {}".format(
         max_sim_weight, max_rel_weight, len(all_nodes)))
    

if __name__ == "__main__":
  main()

