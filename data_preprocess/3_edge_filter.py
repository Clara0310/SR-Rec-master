import sys
from collections import defaultdict


def main():
  # 列印過濾閾值
  print("Filtering nodes with edge num threshold = {}".format(threshold))
  # 構造相似邊檔案名稱
  sim_filename = "{}_sim.edges".format(sys.argv[1])
  # 構造相關邊檔案名稱
  cor_filename = "{}_cor.edges".format(sys.argv[1])

  # 讀取相似邊檔案
  sim_edges = open("./tmp/" + sim_filename, 'r').readlines() 
  # 讀取相關邊檔案
  cor_edges = open("./tmp/" + cor_filename, 'r').readlines() 
  # 儲存將被過濾掉的無效節點
  invalid_nodes = set()

  # 進行多輪過濾，直到沒有節點被移除
  filter_round = 0
  while 1:  # 確保每個商品的相似和相關邊數（關聯view以及關聯buy）都不小於閾值
    filter_round += 1
    print("Round {}".format(filter_round))

    # 統計每個節點的相似邊數（also_view）
    node_sim_score = defaultdict(int)
    # 統計每個節點的相關邊數（also_buy）
    node_cor_score = defaultdict(int)
    # 本輪有效的節點集合
    valid_nodes = set()

    # 處理相似邊，計算每個節點的相似邊數
    for eachline in sim_edges:
      u, v, _ = eachline.strip('\n').split('\t')
      # 跳過已被標記為無效的節點
      if u in invalid_nodes or v in invalid_nodes:
        continue
      # 累計相似邊數
      node_sim_score[u] += 1
      node_sim_score[v] += 1
      # 初始化相關邊數計數
      node_cor_score[u] += 0
      node_cor_score[v] += 0
      # 將節點標記為有效
      valid_nodes.add(u)
      valid_nodes.add(v)
  
    # 處理相關邊，計算每個節點的相關邊數
    for eachline in cor_edges:
      u, v, _ = eachline.strip('\n').split('\t')
      # 跳過已被標記為無效的節點
      if u in invalid_nodes or v in invalid_nodes:
        continue
      # 累計相關邊數
      node_cor_score[u] += 1
      node_cor_score[v] += 1
      # 初始化相似邊數計數
      node_sim_score[u] += 0
      node_sim_score[v] += 0
      # 將節點標記為有效
      valid_nodes.add(u)
      valid_nodes.add(v)
  
    # 列印本輪有效節點數
    print("Num of valid nodes: {}".format(len(valid_nodes)))

    # 標誌位：若本輪沒有新節點被過濾，則停止迴圈
    stop_sign = True

    # 過濾相似邊數不足的節點
    for u, s in node_sim_score.items():
      if s < threshold:
        invalid_nodes.add(u) 
        stop_sign = False 

    # 過濾相關邊數不足的節點
    for u, s in node_cor_score.items():
      if s < threshold:
        invalid_nodes.add(u) 
        stop_sign = False
    
    # 列印本輪被過濾的節點總數
    print("Num of filtered nodes: {}".format(len(invalid_nodes)))
    # 若沒有新節點被過濾，則停止迴圈
    if stop_sign: break
    print("")

  # 建立過濾後的相似邊輸出檔案
  filtered_sim_file = open("./tmp/filtered_{}".format(sim_filename), 'w')
  # 建立過濾後的相關邊輸出檔案
  filtered_cor_file = open("./tmp/filtered_{}".format(cor_filename), 'w')

  # 寫入有效的相似邊
  for eachline in sim_edges:
    u, v, _ = eachline.strip('\n').split('\t')
    # 跳過包含無效節點的邊
    if u in invalid_nodes or v in invalid_nodes:
      continue
    filtered_sim_file.write(eachline)
  
  # 寫入有效的相關邊
  for eachline in cor_edges:
    u, v, _ = eachline.strip('\n').split('\t')
    # 跳過包含無效節點的邊
    if u in invalid_nodes or v in invalid_nodes:
      continue
    filtered_cor_file.write(eachline)


if __name__ == "__main__":
  # 設定邊數最少閾值
  threshold = 1
  main()

