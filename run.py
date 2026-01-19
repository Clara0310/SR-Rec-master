# 匯入自定義模型與實用工具函數
from model import SR_Rec
from utils import *
# 匯入標準庫：隨機、系統操作、命令列參數解析、時間
import random
import os
import argparse
import time
from time import strftime, gmtime

# 環境變數設定，解決 Intel MKL 與 OpenMP 多執行緒衝突
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ['OMP_NUM_THREADS'] = '1'
 
# 建立命令列參數解析器
parser = argparse.ArgumentParser(description='SR_Rec')
# 檢查點路徑（已訓練模型的儲存位置）
parser.add_argument('--ckpt_path', type=str, default='None')
# 計算設備（GPU: 'cuda:0' 或 CPU: 'cpu'）
parser.add_argument('--device', type=str, default='cuda:0')
# 資料集名稱（可選: Appliances、Grocery_and_Gourmet_Food、Home_and_Kitchen）
parser.add_argument('--dataset', type=str, default='Grocery_and_Gourmet_Food')#記得改datasets Appliances Grocery_and_Gourmet_Food Home_and_Kitchen
# 優化器學習率
parser.add_argument('--lr', type=float, default=0.005)
# L2 正則化權重衰減係數
parser.add_argument('--weight_decay', type=float, default=5e-8)
# 獨立訓練運行次數（用於多次實驗統計）
parser.add_argument('--runs', type=int, default=10)
# 嵌入向量維度
parser.add_argument('--embedding_dim', type=int, default=16)
# 早停耐心值（驗證性能無改善時，等待多少個 epoch 後停止訓練）
parser.add_argument('--patience', type=int, default=200)
# 最大訓練 epoch 數
parser.add_argument('--num_epoch', type=int, default=200)
# 驗證頻率（每隔多少個 epoch 進行一次驗證）
parser.add_argument('--val_epoch', type=int, default=1)

# 低階 top-k 參數（用於多層次的推薦排序）
parser.add_argument('--low_k', type=int, default=5) # Appliances:5, Grocery:5, Home:4
# 中階 top-k 參數
parser.add_argument('--mid_k', type=int, default=5) # Appliances:5, Grocery:5, Home:4
# 低階特徵權重因子
parser.add_argument('--alpha', type=int, default=1)
# 中階特徵權重因子
parser.add_argument('--beta', type=int, default=1)
# 模型變體選擇（'att': 注意力、'concat': 連接、'mid': 中階、'low': 低階）
parser.add_argument('--mode', choices=["att", "concat", "mid", "low"], help='the version of models', default='att')

# 解析命令列參數
args = parser.parse_args()

if __name__ == '__main__':
    # 記錄訓練開始時間
    ttt = time.time()

    # 印出所選資料集名稱
    print('Dataset: {}'.format(args.dataset), flush=True)
    # 設定計算設備（GPU 或 CPU）
    device = torch.device(args.device)

    # 建立檢查點儲存目錄結構
    ckpt_path = "checkpoints/"
    if not os.path.exists(ckpt_path):
        os.mkdir(ckpt_path)

    # 依據資料集名稱或自訂路徑設定檢查點子目錄
    if args.ckpt_path == 'None':
        ckpt_path = "checkpoints/{}".format(args.dataset)
    else:
        ckpt_path = "checkpoints/{}".format(args.ckpt_path)
    if not os.path.exists(ckpt_path):
        os.mkdir(ckpt_path)

    # 設定 PyTorch 後端為確定性（可重現），關閉自動調優
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # 生成隨機種子列表（1 到 args.runs）
    seeds = [i + 1 for i in range(args.runs)]

    # 載入資料集：特徵、價格分位、邊索引、訓練/驗證/測試集
    features, price_bin, com_edge_index, train_set, val_set, test_set = load_dataset(args.dataset)

    # 取得商品總數
    num_items = features.shape[0]
    # 從邊索引生成各類型鄰接矩陣（包含自迴圈、去除自迴圈等變體）
    adj, adj_self, adj_dele = generate_adj(com_edge_index, num_items)

    # 將所有資料轉為 PyTorch 張量並搬至指定設備
    features = torch.FloatTensor(features).to(device)
    price_bin = torch.LongTensor(price_bin).to(device)
    adj = adj.to(device)
    adj_self = adj_self.to(device)
    adj_dele = adj_dele.to(device)
    train_set = torch.LongTensor(train_set).to(device)
    val_set = torch.LongTensor(val_set).to(device)
    test_set = torch.LongTensor(test_set).to(device)

    # 初始化結果儲存列表（跨越多次運行的評估指標）
    mean_hr5 = []
    mean_hr10 = []
    mean_ndcg = []

    
    # train the model
    # 開始多次獨立訓練迴圈
    for run in range(args.runs):
        # 取得此次運行的隨機種子
        seed = seeds[run]
        print('\n# Run:{} with random seed:{}'.format(run, seed), flush=True)
        # 為 numpy、torch 等多個隨機庫設定相同種子以確保可重現性
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        random.seed(seed)
        torch.backends.cudnn.deterministic = True
        os.environ['PYTHONHASHSEED'] = str(seed)

        # 建立 SR_Rec 模型實例，傳入超參數與資料集名稱
        model = SR_Rec(args.embedding_dim, 20, args.mode, args.low_k, args.mid_k, args.alpha, args.beta, args.dataset)

        # 將模型搬至指定設備
        model = model.to(device)
        # 建立 Adam 優化器，設定學習率與權重衰減
        optimiser = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

        # 初始化早停相關變數
        cnt_wait = 0  # 驗證性能無改善計數
        best_epoch = 0  # 最佳驗證性能對應的 epoch
        best_hr5 = 0  # 最佳 HR@5 分數
        best_hr10 = 0  # 最佳 HR@10 分數
        best_ndcg = 0  # 最佳 NDCG 分數

        # 訓練迴圈
        for epoch in range(args.num_epoch):

            # 設定模型為訓練模式（啟用 dropout 等）
            model.train()
            # 清空優化器的梯度
            optimiser.zero_grad()
            # 前向傳播，計算訓練集上的損失
            loss = model(features, price_bin, adj, adj_self,adj_dele, train_set)
            # 反向傳播，計算梯度
            loss.backward()
            # 優化器步進，更新模型參數
            optimiser.step()
            # 印出當前 epoch 的損失值
            print('Epoch:{} Loss:{:.8f}'.format(epoch, loss.item()), flush=True)

            
            #validation
            # 驗證階段（根據驗證頻率進行）
            if 1:
                if epoch % args.val_epoch == 0:
                    # 禁用梯度計算以加速推論
                    with torch.no_grad():
                        # 設定模型為評估模式（關閉 dropout 等）
                        model.eval()
                        # 在驗證集上進行推論，取得 HR@5、HR@10、NDCG 分數
                        hr5_score, hr10_score, ndcg_score= model(features, price_bin, adj, adj_self, adj_dele, val_set, 'infer')

                        # 將分數搬至 CPU 便於印出
                        hr5_score = hr5_score.to(torch.device('cpu'))
                        hr10_score = hr10_score.to(torch.device('cpu'))

                    # 印出驗證結果
                    print('Epoch:{} Val HR@5:{:.4f}, HR@10:{:.4f} NDCG:{:.4f}'.format(epoch, hr5_score, hr10_score, ndcg_score), flush=True)

                    # 檢查是否達到新的最佳 NDCG（目標指標）
                    if ndcg_score > best_ndcg:
                        # 更新最佳指標與 epoch
                        best_hr5 = hr5_score
                        best_hr10 = hr10_score
                        best_ndcg = ndcg_score
                        best_run = run
                        best_epoch = epoch
                        # 重置早停計數
                        cnt_wait = 0
                        # 儲存當前最佳模型
                        torch.save(model.state_dict(), '{}/model.pkl'.format(ckpt_path))
                    else:
                        # 性能無改善，增加等待計數
                        cnt_wait += 1

                    # 若等待次數超過耐心值，執行早停
                    if cnt_wait == args.patience:
                        print('Early stopping!', flush=True)
                        break

        # Testing
        # 測試階段
        if 1:
            # 印出最佳模型所在的 epoch
            print('Loading {}th epoch'.format(best_epoch), flush=True)
            # 載入儲存的最佳模型權重
            model.load_state_dict(torch.load('{}/model.pkl'.format(ckpt_path)))
            print('Testing AUC!', flush=True)
            # 禁用梯度計算
            with torch.no_grad():
                # 設定模型為評估模式
                model.eval()
                # 在測試集上進行推論，取得 HR@5、HR@10、NDCG 分數
                hr5_score, hr10_score, ndcg_score= model(features, price_bin, adj, adj_self, adj_dele, test_set, 'infer')
                # 將分數搬至 CPU
                hr5_score = hr5_score.to(torch.device('cpu'))
                hr10_score = hr10_score.to(torch.device('cpu'))

            # 將測試分數加入結果列表（跨越多次運行）
            mean_hr5.append(hr5_score)
            mean_hr10.append(hr10_score)
            mean_ndcg.append(ndcg_score)
            # 印出此次運行的測試結果
            print('Testing HR@5:{:.4f}, HR@10:{:.4f}, NDCG:{:.4f}'.format(hr5_score, hr10_score, ndcg_score),flush=True)


    # 輸出最終彙總結果
    print("--------------------final results--------------------")
    # 計算並印出 HR@5 的平均值、標準差、最大與最小值
    print('Test HR@5: mean{:.4f}, std{:.4f}, max{:.4f}, min{:.4f}'.format(sum(mean_hr5)/len(mean_hr5), np.std(mean_hr5),
        max(mean_hr5), min(mean_hr5)), flush=True)
    # 計算並印出 HR@10 的平均值、標準差、最大與最小值
    print('Test HR@10: mean{:.4f}, std{:.4f}, max{:.4f}, min{:.4f}'.format(sum(mean_hr10)/len(mean_hr10), np.std(mean_hr10),
        max(mean_hr10), min(mean_hr10)), flush=True)
    # 計算並印出 NDCG 的平均值、標準差、最大與最小值
    print('Test NDCG: mean{:.4f}, std{:.4f}, max{:.4f}, min{:.4f}'.format(sum(mean_ndcg)/len(mean_ndcg), np.std(mean_ndcg),
        max(mean_ndcg), min(mean_ndcg)), flush=True)

    # 計算並轉換總執行時間為 HH:MM:SS 格式
    run_time = strftime("%H:%M:%S", gmtime(time.time() - ttt))
    # 印出執行時間
    print('Running time',run_time)