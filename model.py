# 匯入數學函式庫
import math
# 匯入 PyTorch 核心模組
import torch
# 匯入 PyTorch 神經網路模組
import torch.nn as nn
# 匯入 PyTorch 參數類
from torch.nn.parameter import Parameter
# 匯入 sklearn 指標計算函數（用於計算 NDCG）
from sklearn.metrics import ndcg_score

# 雙階段注意力機制模組：用於學習位置權重
class Twostage_Attention(nn.Module):
    # 初始化雙階段注意力模組
    # 參數: hidden_size - 隱藏層維度（預設 16）
    def __init__(self, hidden_size):    # 16
        super(Twostage_Attention, self).__init__()

        # 儲存隱藏層維度
        self.hidden_size = hidden_size

        # 第一階段：pair-wise 注意力機制
        # query、key、value 線性變換層（用於 transformer self-attention）
        self.query1 = nn.Linear(hidden_size, hidden_size)
        self.key1 = nn.Linear(hidden_size, hidden_size)
        self.value1 = nn.Linear(hidden_size, hidden_size)

        # 第二階段：self-attention 機制（目前未使用）
        self.query2 = nn.Linear(hidden_size, hidden_size)
        self.key2 = nn.Linear(hidden_size, hidden_size)
        self.value2 = nn.Linear(hidden_size, hidden_size)

        # softmax 層（用於計算注意力權重）
        self.softmax = nn.Softmax(dim=1)
        # 特徵連接後的線性投影層
        self.nn_cat = nn.Linear(2 * hidden_size, hidden_size)
        # 中間線性變換層
        self.nn_tmp = nn.Linear(hidden_size, hidden_size)
        # 輸出層（將隱藏向量映射到單一標量）
        self.nn_output = nn.Linear(hidden_size, 1)

    # 前向傳播
    # 參數: query_x - query 向量, x - key/value 向量
    def forward(self, query_x, x):
        # 取得批次大小
        batch_size = x.size(0)

        #pair-wise attention
        '''transformer block'''
        # 第一階段：pair-wise 注意力機制（類似 transformer 的自注意力塊）
        # 將 query 向量線性變換
        query = self.query1(query_x).view(batch_size, -1, self.hidden_size)  # [batch_size, seq_len, hidden_size]
        # 將 key 向量線性變換
        key = self.key1(x).view(batch_size, -1, self.hidden_size)  # [batch_size, seq_len, hidden_size]
        # 將 value 向量線性變換
        value = self.value1(x).view(batch_size, -1, self.hidden_size)  # [batch_size, seq_len, hidden_size]
        # 計算 query 與 key 的相似度分數（點積注意力）
        attention_scores = torch.bmm(query, key.transpose(1, 2))  # [batch_size, seq_len, seq_len]
        # 用 softmax 將相似度分數正規化為注意力權重（除以 sqrt(hidden_size) 進行尺度調整）
        attention_scores = self.softmax(attention_scores / (self.hidden_size ** 0.5))  # [batch_size, seq_len, seq_len]
        # 用注意力權重加權 value，得到輸出
        x = torch.bmm(attention_scores, value)  # [batch_size, seq_len, hidden_size]

        # 通過中間層進行特徵變換
        x = self.nn_tmp(x)
        # 通過輸出層得到標量權重，並去掉維度包裝
        x = torch.squeeze(self.nn_output(x))
        # 返回注意力權重向量
        return x

        # 第二階段：自注意力機制（注釋掉的程式碼顯示另一個選項）
        # 此部分程式碼不會執行（因為前面已經 return），保留作備註
        #self attention
        # '''transformer block'''
        # query = self.query2(x).view(batch_size, -1, self.hidden_size)
        # key = self.key2(x).view(batch_size, -1, self.hidden_size)
        # value = self.value2(x).view(batch_size, -1, self.hidden_size)
        # attention_scores = torch.bmm(query, key.transpose(1, 2))
        # attention_scores = self.softmax(attention_scores / (self.hidden_size ** 0.5))
        # x = torch.bmm(attention_scores, value)
        

        # 已棄用的特徵連接方式
        # 已棄用的特徵連接方式
        # x1 = x[:,0,:]
        # x2 = x[:,1,:]
        # x = torch.cat([x1, x2], dim=1)
        # x = self.nn_cat(x)

        #return x

class GCN_Low(nn.Module):

    # 初始化低階 GCN
    # 參數: features_size - 輸入特徵維度, embedding_size - 嵌入維度, low_k - 聚合階數, bias - 是否使用偏置項
    def __init__(self, features_size, embedding_size, low_k, bias=False):  # 16, 16

        super(GCN_Low, self).__init__()
        # 聚合階數（多少層鄰接矩陣運算）
        self.low_k = low_k
        # 線性變換權重矩陣（特徵維度 → 嵌入維度）
        self.weight = Parameter(torch.FloatTensor(features_size, embedding_size))   #
        # 初始化偏置項
        if bias:
            self.bias = Parameter(torch.FloatTensor(embedding_size))
        else:
            self.register_parameter('bias', None)
        # 重設參數初始值
        self.reset_parameters()

    # 重設參數初始值（使用均勻分佈）
    def reset_parameters(self):

        # 計算初始化範圍（基於權重維度的平方根倒數）
        stdv = 1. / math.sqrt(self.weight.size(1))
        # 權重均勻初始化
        self.weight.data.uniform_(-stdv, stdv)
        # 偏置項均勻初始化
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    # 前向傳播
    # 參數: feature - 節點特徵, adj_self - 包含自迴圈的鄰接矩陣
    def forward(self, feature, adj_self):

        # 初始化聚合操作
        # 未使用的替代方案：
        # output = torch.spmm(adj, feature)
        # output = 0.5 * output + 0.5 * feature
        
        # 0.5 倍的自迴圈鄰接矩陣（用於聚合鄰近節點與自身特徵）
        conv = 0.5 * adj_self
        # 第一次矩陣乘法：鄰接矩陣 × 特徵
        output = torch.spmm(conv, feature)
        # 重複聚合 low_k-1 次（共 low_k 次聚合階數）
        for i in range(self.low_k - 1):
            output = torch.spmm(conv, output)
        ''''''

        # 通過線性變換層投影至嵌入空間
        output = torch.mm(output, self.weight)

        # 加入偏置項
        if self.bias is not None:
            output += self.bias
        # 返回聚合後的嵌入
        return output

# 中階圖卷積網路模組：使用差分鄰接矩陣進行聚合（排除自迴圈差）
class GCN_Mid(nn.Module):

    # 初始化中階 GCN
    # 參數: features_size - 輸入特徵維度, embedding_size - 嵌入維度, mid_k - 聚合階數, bias - 是否使用偏置項
    def __init__(self, features_size, embedding_size, mid_k, bias=False):

        super(GCN_Mid, self).__init__()
        # 聚合階數（多少層鄰接矩陣運算）
        self.mid_k = mid_k
        # 線性變換權重矩陣（特徵維度 → 嵌入維度）
        self.weight = Parameter(torch.FloatTensor(features_size, embedding_size))
        # 初始化偏置項
        if bias:
            self.bias = Parameter(torch.FloatTensor(embedding_size))
        else:
            self.register_parameter('bias', None)
        # 重設參數初始值
        self.reset_parameters()

    # 重設參數初始值（使用均勻分佈）
    def reset_parameters(self):

        # 計算初始化範圍
        stdv = 1. / math.sqrt(self.weight.size(1))
        # 權重均勻初始化
        self.weight.data.uniform_(-stdv, stdv)
        # 偏置項均勻初始化
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    # 前向傳播
    # 參數: feature - 節點特徵, adj_self - 自迴圈鄰接矩陣, adj_dele - 去自迴圈鄰接矩陣
    def forward(self, feature, adj_self, adj_dele):

        # 初始化聚合操作
        # 未使用的替代方案：
        # output = torch.spmm(adj, feature)
        # output = torch.spmm(adj, output)
        # output = 0.5 * output - 0.5 * feature
        
        # 計算差分卷積矩陣：-(自迴圈鄰接矩陣 × 去自迴圈鄰接矩陣)
        conv = -torch.spmm(adj_self, adj_dele)
        # 第一次矩陣乘法：差分卷積矩陣 × 特徵
        output = torch.spmm(conv, feature)
        # 重複聚合 mid_k-1 次（共 mid_k 次聚合階數）
        for i in range(self.mid_k - 1):
            output = torch.spmm(conv, output)
        ''''''

        # 通過線性變換層投影至嵌入空間
        output = torch.mm(output, self.weight)
        # 加入偏置項
        if self.bias is not None:
            output += self.bias

        # 返回聚合後的嵌入
        return output

# 商品圖卷積模組：整合低階與中階 GCN 特徵
class Item_Graph_Convolution(nn.Module):

    # 初始化商品圖卷積模組
    # 參數: features_size - 輸入特徵維度, embedding_size - 嵌入維度, mode - 合併模式, low_k - 低階階數, mid_k - 中階階數
    def __init__(self, features_size, embedding_size, mode, low_k, mid_k):    # 16, 16, concat
        super(Item_Graph_Convolution, self).__init__()
        # 合併模式選擇（'att': 注意力, 'concat': 連接, 'mid': 中階, 'low': 低階）
        self.mode = mode
        # 低階 GCN（使用包含自迴圈的鄰接矩陣）
        self.gcn_low = GCN_Low(features_size, embedding_size, low_k)
        # 中階 GCN（使用差分鄰接矩陣）
        self.gcn_mid = GCN_Mid(features_size, embedding_size, mid_k)
        # 批次正規化層 1（用於低階 GCN 輸出）
        self.bn1 = nn.BatchNorm1d(embedding_size)
        # 批次正規化層 2（用於中階 GCN 輸出）
        self.bn2 = nn.BatchNorm1d(embedding_size)
        # 若選擇連接模式，初始化連接層以融合兩個特徵
        if mode == "concat":
            self.nn_cat = nn.Linear(2 * embedding_size, embedding_size)
        else:
            self.nn_cat = None

    # 前向傳播
    # 參數: feature - 節點特徵, adj - 鄰接矩陣, adj_self - 自迴圈鄰接矩陣, adj_dele - 去自迴圈鄰接矩陣
    def forward(self, feature, adj, adj_self, adj_dele):

        # 計算低階 GCN 輸出，並應用批次正規化
        output_low = self.bn1(self.gcn_low(feature, adj_self))
        # 計算中階 GCN 輸出，並應用批次正規化
        output_mid = self.bn2(self.gcn_mid(feature, adj_self, adj_dele))

        # 根據選定的模式融合低階與中階特徵
        if self.mode == "att":
            # 注意力模式：堆疊兩個向量，後續用注意力機制加權
            output = torch.cat([torch.unsqueeze(output_low, dim=1), torch.unsqueeze(output_mid, dim=1)], dim=1)
        elif self.mode == "concat":
            # 連接模式：沿著特徵維度連接，再通過線性層融合
            output = (self.nn_cat(torch.cat([output_low, output_mid], dim=1)))
        elif self.mode == "mid":
            # 中階模式：只使用中階 GCN 輸出
            output = output_mid
        else:
            # 低階模式：只使用低階 GCN 輸出
            output = output_low

        # 返回融合後的嵌入
        return output

# SR_Rec 主模型：整合圖卷積、嵌入與注意力機制的推薦模型
class SR_Rec(nn.Module):

    # 初始化 SR_Rec 模型
    # 參數: embedding_size - 嵌入維度, price_n_bins - 價格分位數, mode - 特徵融合模式, 
    # low_k/mid_k - 聚合階數, alpha/beta - 損失權重, dataset - 資料集名稱, category_emb_size - 分類嵌入維度
    def __init__(self, embedding_size, price_n_bins, mode, low_k, mid_k, alpha, beta, dataset, category_emb_size=768): # 16, 20, att, 768
        super(SR_Rec, self).__init__()
        # 分類嵌入維度（來自 BERT，預設 768）
        self.category_emb_size = category_emb_size  # 768
        # 特徵融合模式
        self.mode = mode
        # 低階特徵損失權重
        self.alpha = alpha
        # 中階特徵損失權重
        self.beta = beta
        # 資料集名稱（用於模式選擇）
        self.dataset = dataset

        # 將 cid2（分類 2）BERT 嵌入投影至嵌入空間
        self.embedding_cid2 = nn.Linear(category_emb_size, embedding_size, bias=True)    # 768 * 16
        # 將 cid3（分類 3）BERT 嵌入投影至嵌入空間
        self.embedding_cid3 = nn.Linear(category_emb_size, embedding_size, bias=True)    # 768 * 16
        # 價格嵌入層（將價格分位轉為嵌入向量）
        self.embedding_price = nn.Embedding(price_n_bins, embedding_size)
        # 融合三種特徵（cid2, cid3, price）的線性層
        self.nn_emb = nn.Linear(embedding_size * 3, embedding_size)                      # (16x3) * 16
        # 商品圖卷積模組
        self.item_gc = Item_Graph_Convolution(embedding_size, embedding_size, self.mode, low_k, mid_k) # 16 * 16, att
        # 超圖網路模組（目前未使用）
        # self.hyper_gnn = Hyper_Graph_Network(embedding_size, hylab, n_hyper_layer) # 16 * 16, att
        # 雙階段注意力機制
        self.two_att = Twostage_Attention(embedding_size)                                # 16
        # 雙線性層：計算兩個嵌入之間的關係分數
        self.rela = nn.Bilinear(embedding_size, embedding_size, 1)

    # 前向傳播（訓練與驗證）
    # 參數: features - 特徵矩陣, price - 價格張量, adj/adj_self/adj_dele - 各類型鄰接矩陣, 
    # train_set - 訓練三元組, mode - 執行模式（'train' 或 'infer'）
    def forward(self, features, price, adj, adj_self, adj_dele, train_set, mode='train'):

        # obtain item embeddings 提取與嵌入特徵
        # 分類 2 嵌入（前 category_emb_size 列）
        cid2 = features[:,:self.category_emb_size]
        # 分類 3 嵌入（後 category_emb_size 列）
        cid3 = features[:,self.category_emb_size:]
        # 投影 cid2 到嵌入空間
        embedded_cid2 = self.embedding_cid2(cid2)
        # 投影 cid3 到嵌入空間
        embedded_cid3 = self.embedding_cid3(cid3)
        # 嵌入價格
        embed_price = self.embedding_price(price)
        # 融合三種特徵，並通過 ReLU 啟動
        item_latent = torch.relu(self.nn_emb(torch.cat([embedded_cid2, embedded_cid3, embed_price], dim=1)))

        # 應用圖卷積，得到低階與中階融合特徵
        item_latent = self.item_gc(item_latent, adj, adj_self, adj_dele)
        # 未使用的超圖網路模組：
        # low_item = item_latent[:, 0, :]
        # mid_item = item_latent[:, 1, :]
        # low_emb = self.hyper_gnn(low_item)  # TODO
        # mid_emb = self.hyper_gnn(mid_item)  # TODO
        # hyper_emb = torch.stack([low_emb, mid_item], dim=1)
        # item_latent = item_latent + hyper_emb

        # 從訓練集中提取 query（key）、正樣本、負樣本的嵌入
        # key 嵌入 (batch_size, 2, embedding_size)
        key_emb = item_latent[train_set[:, 0]]       # (505, 2, 16)
        # 正樣本嵌入 (batch_size, 2, embedding_size)
        pos_emb = item_latent[train_set[:, 1]]       # (505, 2, 16)
        # 負樣本嵌入 (batch_size, num_neg, 2, embedding_size)
        neg_emb = item_latent[train_set[:, 2:]]      # (505, 2, 2, 16)

        # 分離低階與中階嵌入
        low_key_emb, mid_key_emb, low_pos_emb, mid_pos_emb = key_emb[:, 0, :], key_emb[:, 1, :], pos_emb[:, 0, :], pos_emb[:, 1, :]

        # 計算低階相似度分數（點積）
        low_pos = torch.sum(low_key_emb * low_pos_emb, dim=1, keepdim=True) # 6366 * 1
        # 計算中階相似度分數（雙線性）
        mid_pos = self.rela(mid_key_emb, mid_pos_emb)    # 6366 * 1
        # 合併低階與中階正樣本分數
        score_pos = torch.cat((low_pos, mid_pos), dim=1) # 6366 * 2

        # 計算所有負樣本的相似度分數
        for i in range(neg_emb.shape[1]):
            # 提取第 i 個負樣本的低階與中階嵌入
            low_neg_emb, mid_neg_emb = neg_emb[:, i, 0, :], neg_emb[:, i, 1, :]
            if i == 0:
                # 低階負樣本分數
                low_neg = torch.sum(low_key_emb * low_neg_emb, dim=1, keepdim=True) # 6366 * 1
                # 中階負樣本分數
                mid_neg = self.rela(mid_key_emb, mid_neg_emb)    # 6366 * 1
            else:
                # 沿著負樣本維度連接低階分數
                low_neg = torch.cat((low_neg, torch.sum(low_key_emb * low_neg_emb, dim=1, keepdim=True)), dim=1) # 6366 * 2
                # 沿著負樣本維度連接中階分數
                mid_neg = torch.cat((mid_neg, self.rela(mid_key_emb, mid_neg_emb)), dim=1)  # 6366 * 2
        # 堆疊低階與中階負樣本分數（用於分離兩個層級）
        score_neg = torch.stack((low_neg, mid_neg), dim=2)

        # 注意力加權與推薦分數計算（若選擇注意力模式）
        if self.mode == "att":
            # 特殊情況：若資料集為 'Toys_and_Games'，使用簡化的注意力計算
            if self.dataset == 'Toys_and_Games':
                weight_pos = self.two_att(key_emb, pos_emb)  # (505, 16)
            else:
                # 計算 key 相對於 pos 的注意力權重
                key_latent_pos = self.two_att(pos_emb, key_emb) # (505, 16)
                # 計算 pos 相對於 key 的注意力權重
                pos_latent = self.two_att(key_emb, pos_emb)     # (505, 16)
                # 平均兩個方向的注意力權重
                weight_pos = (key_latent_pos + pos_latent) / 2
            # L2 正規化權重向量
            weight_pos = nn.functional.normalize(weight_pos, p=2, dim=1)
            # 用權重加權正樣本分數
            weighted_score_pos = torch.sum(weight_pos * score_pos, dim=1)

            # 計算所有負樣本的注意力權重
            for i in range(neg_emb.shape[1]):
                # 提取第 i 個負樣本的嵌入
                neg_emb_tmp = neg_emb[:, i, :, :]
                if self.dataset != 'Toys_and_Games':
                    # 計算 key 相對於 neg 的注意力權重
                    key_latent_neg_tmp = self.two_att(neg_emb_tmp, key_emb)
                # 計算 neg 相對於 key 的注意力權重
                neg_latent_tmp = self.two_att(key_emb, neg_emb_tmp)
                if i == 0:
                    if self.dataset != 'Toys_and_Games':
                        # 初始化 key 方向的注意力權重張量
                        key_latent_neg = key_latent_neg_tmp.unsqueeze(dim=1)
                    # 初始化 neg 方向的注意力權重張量
                    neg_latent = neg_latent_tmp.unsqueeze(dim=1)

                else:
                    if self.dataset != 'Toys_and_Games':
                        # 沿著負樣本維度連接
                        key_latent_neg = torch.cat((key_latent_neg, key_latent_neg_tmp.unsqueeze(dim=1)), dim=1)
                    # 沿著負樣本維度連接
                    neg_latent = torch.cat((neg_latent, neg_latent_tmp.unsqueeze(dim=1)), dim=1)   # (505, 100, 16)

            # 平均兩個方向的注意力權重（若不是特殊資料集）
            weight_neg = (key_latent_neg + neg_latent) / 2 if self.dataset != 'Toys_and_Games' else neg_latent
            # L2 正規化權重矩陣
            weight_neg = nn.functional.normalize(weight_neg, p=2, dim=2)
            # 用權重加權負樣本分數
            weighted_score_neg = torch.sum(weight_neg * score_neg, dim=2)            # pos_scores = torch.sum(torch.mul(key_latent_pos, pos_latent), dim=1)   # (505,)
            # neg_scores = torch.sum(torch.mul(key_latent_neg, neg_latent), dim=2)   # (505,2)


        # 訓練模式：計算損失函數
        if mode == 'train':
            # 低階對比損失：正樣本分數 vs. 負樣本分數
            loss_low = -torch.mean(torch.log(torch.sigmoid(low_pos - low_neg) + 1e-9))
            # 中階對比損失
            loss_mid = -torch.mean(torch.log(torch.sigmoid(mid_pos - mid_neg) + 1e-9))

            # 加權融合損失：整合正負樣本的相似度對比
            loss = -torch.mean(torch.log(torch.sigmoid(weighted_score_pos.unsqueeze(1) - weighted_score_neg) + 1e-9))
            # 返回總損失：融合損失 + alpha*低階損失 + beta*中階損失
            return loss + self.alpha * loss_low + self.beta * loss_mid

        # 推論模式：計算評估指標
        hr5, hr10, ndcg = self.metrics(torch.unsqueeze(weighted_score_pos, 1), weighted_score_neg)
        return hr5, hr10, ndcg

    # 推論方法（目前未被使用，forward 已包含推論邏輯）
    def inference(self, features, price, adj, test_set):
        # 提取與嵌入特徵
        cid2 = features[:,:self.category_emb_size]
        cid3 = features[:,self.category_emb_size:]
        # 融合三種嵌入類型
        embedded_cid2 = self.embedding_cid2(cid2)
        embedded_cid3 = self.embedding_cid3(cid3)
        embed_price = self.embedding_price(price)
        item_latent = torch.relu(self.nn_emb(torch.cat([embedded_cid2, embedded_cid3, embed_price], dim=1)))
        # 應用圖卷積（注意：此處缺少某些參數，原代碼有誤）
        item_latent = self.item_gc(item_latent, adj)

        # 提取測試集三元組的嵌入
        key_emb = item_latent[test_set[:, 0]]
        pos_emb = item_latent[test_set[:, 1]]
        neg_emb = item_latent[test_set[:, 2:]]

        # 注意力模式：計算加權分數
        if self.mode == "att":
            # 計算雙方向注意力權重
            key_latent_pos = self.two_att(pos_emb, key_emb)
            pos_latent = self.two_att(key_emb, pos_emb)
            # 逐個處理負樣本
            for i in range(neg_emb.shape[1]):
                neg_emb_tmp = neg_emb[:, i, :, :]
                key_latent_neg_tmp = self.two_att(neg_emb_tmp, key_emb)
                neg_latent_tmp = self.two_att(key_emb, neg_emb_tmp)
                if i == 0:
                    # 初始化負樣本張量
                    key_latent_neg = key_latent_neg_tmp.unsqueeze(dim=1)
                    neg_latent = neg_latent_tmp.unsqueeze(dim=1)
                else:
                    # 沿著負樣本維度連接
                    key_latent_neg = torch.cat((key_latent_neg, key_latent_neg_tmp.unsqueeze(dim=1)), dim=1)
                    neg_latent = torch.cat((neg_latent, neg_latent_tmp.unsqueeze(dim=1)), dim=1)

            # 計算加權分數（點積）
            pos_scores = torch.sum(torch.mul(key_latent_pos, pos_latent), dim=1)
            neg_scores = torch.sum(torch.mul(key_latent_neg, neg_latent), dim=2)

        else:
            # 其他模式：簡單的點積相似度
            pos_scores = torch.sum(torch.mul(key_emb, pos_emb), dim=1)
            # 擴展 key 維度以匹配負樣本形狀
            key_emb = key_emb.unsqueeze(dim=1)
            neg_scores = torch.sum(torch.mul(key_emb, neg_emb), dim=2)

        # 計算評估指標
        hr5, hr10, ndcg = self.metrics(torch.unsqueeze(pos_scores, 1), neg_scores)

        return hr5, hr10, ndcg

    # 計算推薦評估指標
    # 參數: pos_scores - 正樣本分數, neg_scores - 負樣本分數
    def metrics(self, pos_scores, neg_scores):

        # concatenate the scores of both positive and negative samples
        # 連接正負樣本分數
        scores = torch.cat([pos_scores, neg_scores], dim=1)
        # 建立標籤（1 表示正樣本，0 表示負樣本）
        labels = torch.cat([torch.ones_like(pos_scores), torch.zeros_like(neg_scores)], dim=1).to(scores.device)
        # 去掉不必要的維度
        scores = torch.squeeze(scores)
        labels = torch.squeeze(labels)
        # 按分數從高到低排序，取得排名索引
        ranking = torch.argsort(scores, dim=1, descending=True)

        #obtain ndcg scores
        ndcg = ndcg_score(labels.cpu(), scores.cpu())

        #obtain hr scores
        #計算 NDCG 指標（需搬至 CPU 計算，因為 sklearn 不支援 GPU）
        # 計算 HR@k 指標（k = 5, 10）
        k_list = [5, 10]
        hr_list = []
        for k in k_list:
            # 取前 k 的排名
            ranking_k = ranking[:, :k]
            # 計算命中率：前 k 中有多少個正樣本被正確排列
            hr = torch.mean(torch.sum(torch.gather(labels, 1, ranking_k), dim=1))
            hr_list.append(hr)

        return hr_list[0], hr_list[1], ndcg
