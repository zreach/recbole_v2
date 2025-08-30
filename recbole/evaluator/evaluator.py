# -*- encoding: utf-8 -*-
# @Time    :   2021/6/25
# @Author  :   Zhichao Feng
# @email   :   fzcbupt@gmail.com

"""
recbole.evaluator.evaluator
#####################################
"""

from recbole.evaluator.register import metrics_dict
from recbole.evaluator.collector import DataStruct
from collections import OrderedDict
import os


class Evaluator(object):
    """Evaluator is used to check parameter correctness, and summarize the results of all metrics."""

    def __init__(self, config):
        self.config = config
        self.metrics = [metric.lower() for metric in self.config["metrics"]]
        self.metric_class = {}

        for metric in self.metrics:
            self.metric_class[metric] = metrics_dict[metric](self.config)

    def evaluate(self, dataobject: DataStruct, flag=None):
        """calculate all the metrics. It is called at the end of each epoch

        Args:
            dataobject (DataStruct): It contains all the information needed for metrics.

        Returns:
            collections.OrderedDict: such as ``{'hit@20': 0.3824, 'recall@20': 0.0527, 'hit@10': 0.3153, 'recall@10': 0.0329, 'gauc': 0.9236}``

        """
        # import pdb;pdb.set_trace()
        result_dict = OrderedDict()
        for metric in self.metrics:
            metric_val = self.metric_class[metric].calculate_metric(dataobject)
            result_dict.update(metric_val)
        
        if flag == "test" and "rec.score" in dataobject:
            import pickle
            save_root = os.path.join(self.config["data_path"], f"results/{self.config['model']}/{self.config['task_name']}")
            os.makedirs(save_root, exist_ok=True)
        
            # 获取预测分数、物品ID和真实标签
            pred_scores = dataobject['rec.score'].cpu().numpy()  # 预测分数
            item_ids = dataobject['data.items'].cpu().numpy()    # 物品ID
            true_labels = dataobject['data.label'].cpu().numpy() # 真实标签
            user_ids = dataobject['data.users'].cpu().numpy()  # 用户ID
            
            # 使用阈值（可以根据需要调整）来将预测分数转换为二分类预测
            threshold = 0.5  # 或者使用其他合适的阈值
            pred_labels = (pred_scores > threshold).astype(int)
            
            # 计算TP, FP, TN, FN
            tp_mask = (true_labels == 1) & (pred_labels == 1)  # 真阳性
            fp_mask = (true_labels == 0) & (pred_labels == 1)  # 假阳性
            tn_mask = (true_labels == 0) & (pred_labels == 0)  # 真阴性
            fn_mask = (true_labels == 1) & (pred_labels == 0)  # 假阴性
            
            # 保存所有样本的分类结果
            classification_results = {
                'true_positive': {
                    'item_ids': item_ids[tp_mask].tolist(),
                    'pred_scores': pred_scores[tp_mask].tolist(),
                    'true_labels': true_labels[tp_mask].tolist(),
                    'count': int(tp_mask.sum())
                },
                'false_positive': {
                    'item_ids': item_ids[fp_mask].tolist(),
                    'pred_scores': pred_scores[fp_mask].tolist(),
                    'true_labels': true_labels[fp_mask].tolist(),
                    'count': int(fp_mask.sum())
                },
                'true_negative': {
                    'item_ids': item_ids[tn_mask].tolist(),
                    'pred_scores': pred_scores[tn_mask].tolist(),
                    'true_labels': true_labels[tn_mask].tolist(),
                    'count': int(tn_mask.sum())
                },
                'false_negative': {
                    'item_ids': item_ids[fn_mask].tolist(),
                    'pred_scores': pred_scores[fn_mask].tolist(),
                    'true_labels': true_labels[fn_mask].tolist(),
                    'count': int(fn_mask.sum())
                },
                'threshold': threshold,
                'total_samples': len(true_labels)
            }

            user_item_scores = {}
            for user_id, item_id, pred_score in zip(user_ids, item_ids, pred_scores):
                if user_id not in user_item_scores:
                    user_item_scores[user_id] = []
                user_item_scores[user_id].append((int(item_id), float(pred_score)))
            
            # 保存用户-物品预测分数字典
            user_scores_path = os.path.join(save_root, "user_item_scores.pkl")
            with open(user_scores_path, 'wb') as f:
                pickle.dump(user_item_scores, f)
            
            # 保存分类结果字典
            save_path = os.path.join(save_root, "classification_results.pkl")
            with open(save_path, 'wb') as f:
                pickle.dump(classification_results, f)
        return result_dict
