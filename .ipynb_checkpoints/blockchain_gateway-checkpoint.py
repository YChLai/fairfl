import requests
import json
import hashlib
import time

class FLBlockchainGateway:
    def __init__(self, gateway_url="http://localhost:8080/api/fl/recordRound"):
        self.gateway_url = gateway_url

    def upload_round_data(self, round_idx, server, clients, 
                          reputations, incentive_ratios):
        """
        发送数据给 Java 网关
        """
        try:
            # 放大倍数 (合约中涉及金额或分数通常需要放大)
            SCALE = 1000000 
            
            # 1. 准备数据列表
            client_ids = []
            rep_list = []
            ratio_list = []

            for i, client in enumerate(clients):
                client_ids.append(f"client_{i}")
                
                # 转换 Reputation
                rep_val = float(reputations[i])
                rep_list.append(int(rep_val * SCALE))
                
                # 转换 Incentive Ratio
                ratio_val = float(incentive_ratios[i])
                ratio_list.append(int(ratio_val * SCALE))

            # 2. 全局模型 Hash (简化版，仅Hash一部分以提高速度)
            first_layer_key = list(server.model.state_dict().keys())[0]
            first_layer_data = str(server.model.state_dict()[first_layer_key])
            global_hash = hashlib.sha256(first_layer_data.encode()).hexdigest()

            # 3. 准备贡献数据 (Contribution JSON)
            contrib_list = []
            for i, client in enumerate(clients):
                # 计算局部模型 Hash (防篡改)
                local_model_str = str(client.model.state_dict())
                local_hash = hashlib.sha256(local_model_str.encode('utf-8')).hexdigest()
        
                record = {
                    "client_id": client.id,
                    "reputation": round(rs[i].item(), 6), # 最终声誉
                    "cosine_sim": round(phis[i].item(), 6), # 梯度对齐度 (新增)
                    "data_size": client.train_size,
                    "class_diversity": len(client.prototype.keys()), # 类别多样性 (新增)
                    "est_noise_rate": round(noise_levels.get(client.id, 0.0), 4), # 噪声率 (新增)
                    "local_model_hash": local_hash # 局部梯度哈希 (新增)
                }
                contrib_list.append(record)

            # 4. 发送
            print(f">>> [Blockchain] Uploading Round {round_idx}...")
            response = requests.post(self.gateway_url, json=payload)
            
            if response.status_code == 200:
                print(f">>> [Blockchain] Response: {response.text}")
            else:
                print(f">>> [Blockchain] Failed: {response.status_code} - {response.text}")

        except Exception as e:
            print(f">>> [Blockchain] Error: {e}")