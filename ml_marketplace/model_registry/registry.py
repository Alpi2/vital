"""
ML Model Registry
Manage model versions, metadata, and lifecycle
"""

import json
import hashlib
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime
import logging

class ModelRegistry:
    """Central registry for ML models"""
    
    def __init__(self, registry_path: str = './model_registry'):
        self.registry_path = Path(registry_path)
        self.registry_path.mkdir(parents=True, exist_ok=True)
        self.metadata_file = self.registry_path / 'registry.json'
        self.logger = logging.getLogger(__name__)
        self._load_registry()
    
    def _load_registry(self):
        if self.metadata_file.exists():
            with open(self.metadata_file, 'r') as f:
                self.registry = json.load(f)
        else:
            self.registry = {'models': {}}
            self._save_registry()
    
    def _save_registry(self):
        with open(self.metadata_file, 'w') as f:
            json.dump(self.registry, f, indent=2)
    
    def register_model(self, name: str, version: str, model_path: str, metadata: Dict) -> str:
        """Register a new model or version"""
        model_id = f"{name}:{version}"
        model_info = {
            'name': name,
            'version': version,
            'model_path': model_path,
            'metadata': metadata,
            'registered_at': datetime.utcnow().isoformat()
        }
        
        if name not in self.registry['models']:
            self.registry['models'][name] = {}
        
        self.registry['models'][name][version] = model_info
        self._save_registry()
        return model_id
