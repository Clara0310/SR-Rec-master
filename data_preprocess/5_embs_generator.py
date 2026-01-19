import sys
from transformers import BertTokenizer, BertModel
import pandas as pd
import torch
import numpy as np

def main():
    print("Generating category embeddings with BERT...")
    # 從命令列參數取得資料集名稱
    data_name = sys.argv[1]

    # 構造分類 2 和分類 3 的字典檔案路徑
    data_path1 = "./tmp/{}_cid2_dict.txt".format(data_name)
    data_path2 = "./tmp/{}_cid3_dict.txt".format(data_name)
    
    # 載入預訓練的 BERT 模型和 tokenizer 分詞器
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased')
    
    # 儲存生成的嵌入矩陣
    embedding_matrix_list = []
    index = 2
    
    # 為兩個分類級別分別生成嵌入
    for data_path in [data_path1, data_path2]:
        # 讀取分類 ID 和對應的文本
        id_pd = pd.read_csv(data_path, sep='\t', header=None)
        # 按分類 ID 排序
        id_pd = id_pd.sort_values(by=0)
        # 獲得分類對應的文本
        categories = np.array(id_pd[1])

        # 儲存此分類級別的所有嵌入
        embeddings_matrix = []
        # 為每個分類文本生成嵌入
        for text in categories:
            # 使用tokenizer分詞器將文本轉換為 token
            tokens = tokenizer.encode(text, add_special_tokens=True)
            # 轉換為張量格式
            tokens_tensor = torch.tensor([tokens])

            # 使用 BERT 模型生成嵌入（不計算梯度）
            with torch.no_grad():
                outputs = model(tokens_tensor)

            # 獲得最後隱藏層的輸出
            last_hidden_states = outputs[0]
            # 對序列維度取平均以獲得固定大小的嵌入向量
            embeddings = torch.mean(last_hidden_states, dim=1)
            embeddings_matrix.append(embeddings)
        
        # 將所有嵌入向量連接為一個矩陣
        embeddings_matrix = torch.cat(embeddings_matrix, dim=0)
        embedding_matrix_list.append(embeddings_matrix)
        
        # 列印嵌入矩陣的形狀
        print("category {} embedding shape: {}".format(index, embeddings_matrix.shape))
        index += 1
    
    # 構造嵌入檔案的存儲路徑
    save_path = "./embs/{}_embeddings.npz".format(data_name)
    # 儲存分類 2 和分類 3 的嵌入矩陣到 NPZ 檔案
    np.savez(save_path, cid2_emb=embedding_matrix_list[0], cid3_emb=embedding_matrix_list[1])

if __name__ == '__main__':
    main()

