#!/usr/bin/env python3
"""
ML Model Registry Backend
Manages model versioning, metadata, and lifecycle
"""

import os
import json
import hashlib
from datetime import datetime
from typing import Dict, List, Optional
import boto3
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Text, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

Base = declarative_base()

class ModelVersion(Base):
    __tablename__ = 'model_versions'
    
    id = Column(Integer, primary_key=True)
    model_name = Column(String(200), nullable=False)
    version = Column(String(50), nullable=False)
    model_hash = Column(String(64), unique=True)
    storage_path = Column(String(500))
    metadata = Column(JSON)
    created_at = Column(DateTime, default=datetime.utcnow)
    created_by = Column(String(100))
    status = Column(String(50), default='registered')
    
class ModelRegistry:
    def __init__(self, db_url: str, storage_backend: str = 's3'):
        self.engine = create_engine(db_url)
        Base.metadata.create_all(self.engine)
        Session = sessionmaker(bind=self.engine)
        self.session = Session()
        
        self.storage_backend = storage_backend
        if storage_backend == 's3':
            self.s3_client = boto3.client('s3')
            self.bucket_name = 'vital-ml-models'
    
    def register_model(
        self,
        model_name: str,
        version: str,
        model_path: str,
        metadata: Dict,
        created_by: str
    ) -> int:
        """
        Register a new model version
        """
        # Calculate model hash
        model_hash = self._calculate_file_hash(model_path)
        
        # Check if model already exists
        existing = self.session.query(ModelVersion).filter_by(model_hash=model_hash).first()
        if existing:
            raise ValueError(f"Model with hash {model_hash} already registered")
        
        # Upload model to storage
        storage_path = self._upload_model(model_name, version, model_path)
        
        # Create registry entry
        model_version = ModelVersion(
            model_name=model_name,
            version=version,
            model_hash=model_hash,
            storage_path=storage_path,
            metadata=metadata,
            created_by=created_by,
            status='registered'
        )
        
        self.session.add(model_version)
        self.session.commit()
        
        print(f"✅ Model registered: {model_name} v{version} (ID: {model_version.id})")
        return model_version.id
    
    def get_model(
        self,
        model_name: str,
        version: Optional[str] = None
    ) -> Optional[ModelVersion]:
        """
        Get model by name and version (latest if version not specified)
        """
        query = self.session.query(ModelVersion).filter_by(model_name=model_name)
        
        if version:
            query = query.filter_by(version=version)
        else:
            query = query.order_by(ModelVersion.created_at.desc())
        
        return query.first()
    
    def list_models(
        self,
        model_name: Optional[str] = None,
        status: Optional[str] = None
    ) -> List[ModelVersion]:
        """
        List all models or filter by name/status
        """
        query = self.session.query(ModelVersion)
        
        if model_name:
            query = query.filter_by(model_name=model_name)
        if status:
            query = query.filter_by(status=status)
        
        return query.order_by(ModelVersion.created_at.desc()).all()
    
    def update_model_status(
        self,
        model_id: int,
        status: str
    ) -> None:
        """
        Update model status (registered, validated, deployed, deprecated)
        """
        model = self.session.query(ModelVersion).get(model_id)
        if not model:
            raise ValueError(f"Model ID {model_id} not found")
        
        model.status = status
        self.session.commit()
        
        print(f"✅ Model {model_id} status updated to: {status}")
    
    def download_model(
        self,
        model_id: int,
        download_path: str
    ) -> str:
        """
        Download model from storage
        """
        model = self.session.query(ModelVersion).get(model_id)
        if not model:
            raise ValueError(f"Model ID {model_id} not found")
        
        if self.storage_backend == 's3':
            self.s3_client.download_file(
                self.bucket_name,
                model.storage_path,
                download_path
            )
        else:
            # Local filesystem
            import shutil
            shutil.copy(model.storage_path, download_path)
        
        print(f"✅ Model downloaded to: {download_path}")
        return download_path
    
    def compare_models(
        self,
        model_id_a: int,
        model_id_b: int
    ) -> Dict:
        """
        Compare two model versions
        """
        model_a = self.session.query(ModelVersion).get(model_id_a)
        model_b = self.session.query(ModelVersion).get(model_id_b)
        
        if not model_a or not model_b:
            raise ValueError("One or both models not found")
        
        comparison = {
            'model_a': {
                'id': model_a.id,
                'name': model_a.model_name,
                'version': model_a.version,
                'metadata': model_a.metadata,
                'created_at': model_a.created_at.isoformat()
            },
            'model_b': {
                'id': model_b.id,
                'name': model_b.model_name,
                'version': model_b.version,
                'metadata': model_b.metadata,
                'created_at': model_b.created_at.isoformat()
            },
            'differences': self._calculate_differences(model_a.metadata, model_b.metadata)
        }
        
        return comparison
    
    def _calculate_file_hash(self, file_path: str) -> str:
        """
        Calculate SHA256 hash of file
        """
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()
    
    def _upload_model(self, model_name: str, version: str, model_path: str) -> str:
        """
        Upload model to storage backend
        """
        storage_key = f"models/{model_name}/{version}/{os.path.basename(model_path)}"
        
        if self.storage_backend == 's3':
            self.s3_client.upload_file(model_path, self.bucket_name, storage_key)
        else:
            # Local filesystem
            storage_dir = f"./model_storage/{model_name}/{version}"
            os.makedirs(storage_dir, exist_ok=True)
            import shutil
            shutil.copy(model_path, os.path.join(storage_dir, os.path.basename(model_path)))
        
        return storage_key
    
    def _calculate_differences(self, metadata_a: Dict, metadata_b: Dict) -> Dict:
        """
        Calculate differences between two metadata dictionaries
        """
        differences = {}
        
        all_keys = set(metadata_a.keys()) | set(metadata_b.keys())
        
        for key in all_keys:
            val_a = metadata_a.get(key)
            val_b = metadata_b.get(key)
            
            if val_a != val_b:
                differences[key] = {
                    'model_a': val_a,
                    'model_b': val_b
                }
        
        return differences

# Example usage
if __name__ == '__main__':
    registry = ModelRegistry(
        db_url='postgresql://vital_user:vital_password@localhost/ml_marketplace',
        storage_backend='local'
    )
    
    # Register a model
    model_id = registry.register_model(
        model_name='ecg_arrhythmia_classifier',
        version='2.1.0',
        model_path='./models/ecg_classifier.h5',
        metadata={
            'framework': 'TensorFlow',
            'accuracy': 0.967,
            'latency_ms': 8.7,
            'model_size_mb': 256,
            'training_date': '2026-01-04',
            'dataset_size': 25000
        },
        created_by='ml_team@vital.ai'
    )
    
    # List all models
    models = registry.list_models()
    print(f"\nTotal models: {len(models)}")
    
    for model in models:
        print(f"  - {model.model_name} v{model.version} ({model.status})")
