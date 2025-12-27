import json
import time
import hashlib
from client.bcosclient import BcosClient
from client.datatype_parser import DatatypeParser

class BlockchainLogger:
    def __init__(self, contract_address):
        """
        初始化区块链连接
        contract_address: 你在 WeBASE 部署后得到的 0x... 地址
        """
        self.client = BcosClient()
        self.contract_address = contract_address
        self.abi_file = "contracts/FLAudit.abi" # 确保你把 ABI 文件放到了这里
        
        try:
            with open(self.abi_file, 'r') as f:
                self.contract_abi = json.load(f)
        except Exception as e:
            print(f"[Blockchain] Error loading ABI: {e}")

    def log_round(self, round_idx, server, clients, rs, q_ratios):
        """
        核心函数：提取数据 -> 打包 -> 上链
        """
        print(f"\n[Blockchain] Preparing data for Round {round_idx}...")
        
        # 1. 准备时间戳
        timestamp = str(int(time.time()))
        
        # 2. 计算全局模型 Hash (防篡改指纹)
        # 将 state_dict 转为字符串后计算 SHA256
        model_str = str(server.model.state_dict())
        global_hash = hashlib.sha256(model_str.encode('utf-8')).hexdigest()
        
        # 3. 准备贡献数据 (Contribution JSON)
        # 包含：Reputation, Cosine Similarity 等
        contrib_list = []
        for i, client in enumerate(clients):
            # 获取余弦相似度 (假设你代码里有 self.simi 或直接用 rs 估算)
            # 这里直接存最终的 Reputation
            record = {
                "client_id": client.id,
                "reputation": round(rs[i].item(), 6),
                "data_size": client.train_size,
                # 如果有 FedCorr 的噪声率，可以在这里加:
                # "noise_rate": client.estimated_noise
            }
            contrib_list.append(record)
        contrib_json = json.dumps(contrib_list)
        
        # 4. 准备激励数据 (Incentive JSON)
        incentive_list = []
        for i, client in enumerate(clients):
            record = {
                "client_id": client.id,
                "incentive_ratio": round(q_ratios[i].item(), 4),
                "status": "Full" if q_ratios[i]>0.9 else ("Sparse" if q_ratios[i]>0.05 else "Zero")
            }
            incentive_list.append(record)
        incentive_json = json.dumps(incentive_list)

        # 5. 发送交易调用合约
        try:
            # 调用 recordRound 函数
            # 参数顺序必须与 Solidity 一致: roundId, timestamp, globalModelHash, contribJson, incentiveJson
            receipt = self.client.sendRawTransactionGetReceipt(
                self.contract_address,
                self.contract_abi,
                "recordRound",
                [round_idx, timestamp, global_hash, contrib_json, incentive_json]
            )
            
            # 检查结果
            if receipt['status'] == '0x0':
                print(f"[Blockchain] ✅ Round {round_idx} saved successfully!")
                print(f"[Blockchain] Tx Hash: {receipt['transactionHash']}")
            else:
                print(f"[Blockchain] ❌ Transaction failed. Status: {receipt['status']}")
                
        except Exception as e:
            print(f"[Blockchain] ❌ Error uploading to chain: {e}")