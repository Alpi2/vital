"""
A/B Testing Framework for ML Models
"""

import random
from typing import Dict, List
from datetime import datetime
import logging
from collections import defaultdict

class ABTestFramework:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.experiments = {}
        self.results = defaultdict(lambda: defaultdict(list))
    
    def create_experiment(self, experiment_id: str, models: List[Dict], traffic_split: Dict = None) -> Dict:
        if traffic_split is None:
            split_value = 1.0 / len(models)
            traffic_split = {m['id']: split_value for m in models}
        
        experiment = {
            'id': experiment_id,
            'models': models,
            'traffic_split': traffic_split,
            'created_at': datetime.utcnow().isoformat(),
            'status': 'active'
        }
        
        self.experiments[experiment_id] = experiment
        return experiment
    
    def route_request(self, experiment_id: str) -> str:
        experiment = self.experiments[experiment_id]
        rand = random.random()
        cumulative = 0.0
        
        for model_id, split in experiment['traffic_split'].items():
            cumulative += split
            if rand <= cumulative:
                return model_id
        
        return list(experiment['traffic_split'].keys())[0]
    
    def record_result(self, experiment_id: str, model_id: str, metrics: Dict):
        result = {
            'timestamp': datetime.utcnow().isoformat(),
            'model_id': model_id,
            'metrics': metrics
        }
        self.results[experiment_id][model_id].append(result)
