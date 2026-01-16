import requests
import json
import time

class Gateway:
    def __init__(self, base_url="http://localhost:8080"):
        self.base_url = base_url
        self.headers = {'Content-Type': 'application/json'}
        self.scale_factor = 10000  # 用于将浮点数转为整数 (保留4位小数)

    def upload_round_data(self, round_idx, server, clients, reputations, incentive_ratios):
        """
        封装完整的上链流程：开启轮次 -> 上传各客户端数据 -> 结算轮次
        """
        print(f"\n[Gateway] 正在将第 {round_idx} 轮数据上链...")

        # 1. 开启新一轮 (startRound)
        # -------------------------------------------------
        # 实际项目中可以使用 server.model 的 hash，这里简化处理
        global_model_hash = f"round_{round_idx}_model_hash" 
        total_budget = 500000 # 示例预算

        start_payload = {
            "_roundId": round_idx,
            "_globalModelHash": global_model_hash,
            "_totalBudget": total_budget
        }
        
        if not self._send_request("/startRound", start_payload, "开启轮次"):
            return # 如果开启失败，后续无需执行

        # 2. 上传每个客户端的贡献 (uploadClientContributions)
        # -------------------------------------------------
        success_count = 0
        for i, client in enumerate(clients):
            # 获取缓存的指标 (如果在 fl_client.py 中没初始化，默认为0)
            lid = getattr(client, 'lid_score_cache', 0.0)
            cosine = getattr(client, 'cosine_sim_cache', 0.0)
            is_noisy = getattr(client, 'is_noisy', False)
            
            # 确保 reputation 和 incentive_ratio 不越界
            rep = reputations[i] if i < len(reputations) else 0.0
            inc = incentive_ratios[i] if i < len(incentive_ratios) else 0.0

            # 构造参数 (浮点数 -> 整数)
            contrib_payload = {
                "_roundId": round_idx,
                "_clientIds": str(client.id), # 这里的key是复数s，对应Java BO
                "_gradientHash": f"grad_hash_c{client.id}",
                "_sampleSize": client.train_size,
                "_lidScore": int(lid * self.scale_factor),
                "_cosineSim": int(cosine * self.scale_factor),
                "_isNoisy": bool(is_noisy),
                "_reputScore": int(rep * self.scale_factor),
                "_incentiveRatio": int(inc * self.scale_factor)
            }
            
            if self._send_request("/uploadClientContributions", contrib_payload, f"客户端{client.id}上传"):
                success_count += 1

        # 3. 结束本轮 (finalizeRound)
        # -------------------------------------------------
        finalize_payload = {
            "_roundId": round_idx
        }
        self._send_request("/finalizeRound", finalize_payload, "结算轮次")
        
        print(f"[Gateway] 上链完成. 成功上传 {success_count}/{len(clients)} 个客户端数据.\n")

    def _send_request(self, endpoint, data, action_name=""):
        url = self.base_url + endpoint
        try:
            response = requests.post(url, headers=self.headers, json=data)
            if response.status_code == 200:
                # 检查区块链交易状态
                res_json = response.json()
                status = res_json.get('status', 'unknown')
                tx_receipt = res_json.get('transactionReceipt', {})
                tx_status = tx_receipt.get('status') if tx_receipt else None
                
                # 适配：直接返回0x0 或 receipt里是0x0 都算成功
                if status == "0x0" or tx_status == "0x0" or status == 0:
                    return True
                else:
                    print(f"  [Error] {action_name} 交易失败: {res_json}")
            else:
                print(f"  [Error] {action_name} HTTP错误 {response.status_code}: {response.text}")
        except Exception as e:
            print(f"  [Error] {action_name} 连接异常: {e}")
        return False