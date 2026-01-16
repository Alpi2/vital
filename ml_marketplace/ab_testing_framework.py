#!/usr/bin/env python3
"""
A/B Testing Framework for ML Models
Enables controlled experiments to compare model performance
"""

import random
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import numpy as np
from scipy import stats
import json

class TestStatus(Enum):
    DRAFT = "draft"
    ACTIVE = "active"
    PAUSED = "paused"
    COMPLETED = "completed"
    CANCELLED = "cancelled"

@dataclass
class ModelVariant:
    model_id: int
    model_name: str
    version: str
    traffic_allocation: float
    endpoint_url: str

@dataclass
class TestMetrics:
    variant_id: str
    total_requests: int
    successful_requests: int
    failed_requests: int
    avg_latency_ms: float
    avg_accuracy: float
    total_cost: float

class ABTest:
    def __init__(
        self,
        test_id: str,
        test_name: str,
        variant_a: ModelVariant,
        variant_b: ModelVariant,
        success_metric: str = 'accuracy',
        min_sample_size: int = 1000,
        confidence_level: float = 0.95
    ):
        self.test_id = test_id
        self.test_name = test_name
        self.variant_a = variant_a
        self.variant_b = variant_b
        self.success_metric = success_metric
        self.min_sample_size = min_sample_size
        self.confidence_level = confidence_level
        
        self.status = TestStatus.DRAFT
        self.start_time = None
        self.end_time = None
        
        # Metrics storage
        self.metrics_a = []
        self.metrics_b = []
        
        # Results
        self.winner = None
        self.statistical_significance = None
    
    def start(self):
        """Start the A/B test"""
        self.status = TestStatus.ACTIVE
        self.start_time = datetime.now()
        print(f"✅ A/B Test '{self.test_name}' started")
        print(f"   Variant A: {self.variant_a.model_name} ({self.variant_a.traffic_allocation*100}% traffic)")
        print(f"   Variant B: {self.variant_b.model_name} ({self.variant_b.traffic_allocation*100}% traffic)")
    
    def route_request(self) -> str:
        """
        Route incoming request to variant A or B based on traffic allocation
        """
        if self.status != TestStatus.ACTIVE:
            raise ValueError("Test is not active")
        
        rand = random.random()
        if rand < self.variant_a.traffic_allocation:
            return 'A'
        else:
            return 'B'
    
    def record_result(
        self,
        variant: str,
        latency_ms: float,
        accuracy: float,
        success: bool,
        cost: float = 0.0
    ):
        """
        Record result for a variant
        """
        result = {
            'timestamp': datetime.now(),
            'latency_ms': latency_ms,
            'accuracy': accuracy,
            'success': success,
            'cost': cost
        }
        
        if variant == 'A':
            self.metrics_a.append(result)
        elif variant == 'B':
            self.metrics_b.append(result)
        else:
            raise ValueError(f"Invalid variant: {variant}")
    
    def get_current_metrics(self) -> Tuple[TestMetrics, TestMetrics]:
        """
        Get current metrics for both variants
        """
        metrics_a = self._calculate_metrics(self.metrics_a, 'A')
        metrics_b = self._calculate_metrics(self.metrics_b, 'B')
        
        return metrics_a, metrics_b
    
    def _calculate_metrics(self, results: List[Dict], variant_id: str) -> TestMetrics:
        """
        Calculate aggregated metrics from results
        """
        if not results:
            return TestMetrics(
                variant_id=variant_id,
                total_requests=0,
                successful_requests=0,
                failed_requests=0,
                avg_latency_ms=0.0,
                avg_accuracy=0.0,
                total_cost=0.0
            )
        
        total = len(results)
        successful = sum(1 for r in results if r['success'])
        failed = total - successful
        
        avg_latency = np.mean([r['latency_ms'] for r in results])
        avg_accuracy = np.mean([r['accuracy'] for r in results])
        total_cost = sum(r['cost'] for r in results)
        
        return TestMetrics(
            variant_id=variant_id,
            total_requests=total,
            successful_requests=successful,
            failed_requests=failed,
            avg_latency_ms=avg_latency,
            avg_accuracy=avg_accuracy,
            total_cost=total_cost
        )
    
    def check_statistical_significance(self) -> Dict:
        """
        Check if results are statistically significant using t-test
        """
        if len(self.metrics_a) < self.min_sample_size or len(self.metrics_b) < self.min_sample_size:
            return {
                'significant': False,
                'reason': f'Insufficient samples (need {self.min_sample_size}, have A:{len(self.metrics_a)}, B:{len(self.metrics_b)})'
            }
        
        # Extract success metric values
        values_a = [r[self.success_metric] for r in self.metrics_a]
        values_b = [r[self.success_metric] for r in self.metrics_b]
        
        # Perform t-test
        t_statistic, p_value = stats.ttest_ind(values_a, values_b)
        
        # Calculate effect size (Cohen's d)
        mean_a = np.mean(values_a)
        mean_b = np.mean(values_b)
        std_a = np.std(values_a, ddof=1)
        std_b = np.std(values_b, ddof=1)
        pooled_std = np.sqrt(((len(values_a)-1)*std_a**2 + (len(values_b)-1)*std_b**2) / (len(values_a) + len(values_b) - 2))
        cohens_d = (mean_a - mean_b) / pooled_std
        
        alpha = 1 - self.confidence_level
        is_significant = p_value < alpha
        
        # Determine winner
        if is_significant:
            if mean_a > mean_b:
                winner = 'A'
                improvement = ((mean_a - mean_b) / mean_b) * 100
            else:
                winner = 'B'
                improvement = ((mean_b - mean_a) / mean_a) * 100
        else:
            winner = None
            improvement = 0
        
        return {
            'significant': is_significant,
            'p_value': p_value,
            'confidence_level': self.confidence_level,
            't_statistic': t_statistic,
            'cohens_d': cohens_d,
            'winner': winner,
            'improvement_percent': improvement,
            'mean_a': mean_a,
            'mean_b': mean_b,
            'std_a': std_a,
            'std_b': std_b
        }
    
    def stop(self, reason: str = 'completed'):
        """
        Stop the A/B test
        """
        self.end_time = datetime.now()
        
        if reason == 'completed':
            self.status = TestStatus.COMPLETED
            
            # Check statistical significance
            sig_results = self.check_statistical_significance()
            self.statistical_significance = sig_results
            self.winner = sig_results.get('winner')
            
            print(f"\n✅ A/B Test '{self.test_name}' completed")
            print(f"   Duration: {self.end_time - self.start_time}")
            print(f"   Winner: {self.winner if self.winner else 'No significant difference'}")
            if sig_results['significant']:
                print(f"   Improvement: {sig_results['improvement_percent']:.2f}%")
                print(f"   P-value: {sig_results['p_value']:.4f}")
        else:
            self.status = TestStatus.CANCELLED
            print(f"⚠️ A/B Test '{self.test_name}' cancelled: {reason}")
    
    def get_report(self) -> Dict:
        """
        Generate comprehensive test report
        """
        metrics_a, metrics_b = self.get_current_metrics()
        
        report = {
            'test_id': self.test_id,
            'test_name': self.test_name,
            'status': self.status.value,
            'start_time': self.start_time.isoformat() if self.start_time else None,
            'end_time': self.end_time.isoformat() if self.end_time else None,
            'duration_hours': (self.end_time - self.start_time).total_seconds() / 3600 if self.end_time and self.start_time else None,
            'variant_a': {
                'model_name': self.variant_a.model_name,
                'version': self.variant_a.version,
                'traffic_allocation': self.variant_a.traffic_allocation,
                'metrics': {
                    'total_requests': metrics_a.total_requests,
                    'success_rate': metrics_a.successful_requests / metrics_a.total_requests if metrics_a.total_requests > 0 else 0,
                    'avg_latency_ms': metrics_a.avg_latency_ms,
                    'avg_accuracy': metrics_a.avg_accuracy,
                    'total_cost': metrics_a.total_cost
                }
            },
            'variant_b': {
                'model_name': self.variant_b.model_name,
                'version': self.variant_b.version,
                'traffic_allocation': self.variant_b.traffic_allocation,
                'metrics': {
                    'total_requests': metrics_b.total_requests,
                    'success_rate': metrics_b.successful_requests / metrics_b.total_requests if metrics_b.total_requests > 0 else 0,
                    'avg_latency_ms': metrics_b.avg_latency_ms,
                    'avg_accuracy': metrics_b.avg_accuracy,
                    'total_cost': metrics_b.total_cost
                }
            },
            'statistical_analysis': self.statistical_significance,
            'winner': self.winner,
            'recommendation': self._get_recommendation()
        }
        
        return report
    
    def _get_recommendation(self) -> str:
        """
        Get deployment recommendation based on test results
        """
        if not self.statistical_significance:
            return "Test not yet completed or insufficient data"
        
        if not self.statistical_significance['significant']:
            return "No statistically significant difference detected. Either variant can be used."
        
        winner = self.statistical_significance['winner']
        improvement = self.statistical_significance['improvement_percent']
        
        if winner == 'A':
            return f"Deploy Variant A ({self.variant_a.model_name}). Shows {improvement:.2f}% improvement over Variant B."
        else:
            return f"Deploy Variant B ({self.variant_b.model_name}). Shows {improvement:.2f}% improvement over Variant A."

# Example usage
if __name__ == '__main__':
    # Create variants
    variant_a = ModelVariant(
        model_id=1,
        model_name='ecg_classifier_v1',
        version='1.0.0',
        traffic_allocation=0.5,
        endpoint_url='https://api.vital.ai/models/1/predict'
    )
    
    variant_b = ModelVariant(
        model_id=2,
        model_name='ecg_classifier_v2',
        version='2.0.0',
        traffic_allocation=0.5,
        endpoint_url='https://api.vital.ai/models/2/predict'
    )
    
    # Create A/B test
    ab_test = ABTest(
        test_id='test_001',
        test_name='ECG Classifier v1 vs v2',
        variant_a=variant_a,
        variant_b=variant_b,
        success_metric='accuracy',
        min_sample_size=1000,
        confidence_level=0.95
    )
    
    # Start test
    ab_test.start()
    
    # Simulate requests
    print("\nSimulating 2000 requests...")
    for i in range(2000):
        variant = ab_test.route_request()
        
        # Simulate model performance (variant B is slightly better)
        if variant == 'A':
            latency = random.gauss(10, 2)
            accuracy = random.gauss(0.92, 0.05)
        else:
            latency = random.gauss(9, 2)
            accuracy = random.gauss(0.95, 0.04)
        
        success = accuracy > 0.8
        cost = latency * 0.001
        
        ab_test.record_result(variant, latency, accuracy, success, cost)
    
    # Check significance
    sig_results = ab_test.check_statistical_significance()
    print(f"\nStatistical Significance: {sig_results['significant']}")
    print(f"P-value: {sig_results['p_value']:.4f}")
    print(f"Winner: {sig_results.get('winner', 'None')}")
    
    # Stop test
    ab_test.stop()
    
    # Get report
    report = ab_test.get_report()
    print(f"\n{json.dumps(report, indent=2)}")
