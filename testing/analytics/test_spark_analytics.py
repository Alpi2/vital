#!/usr/bin/env python3
"""
Comprehensive tests for Spark Analytics Engine
Tests: Patient flow analysis, real-time streaming, data aggregation, performance
"""

import pytest
import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql.types import *
from datetime import datetime, timedelta
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../analytics/platform')))

from spark_analytics_engine import HealthcareAnalyticsEngine

class TestSparkAnalyticsEngine:
    
    @pytest.fixture(scope="class")
    def spark_session(self):
        """Create Spark session for testing"""
        spark = SparkSession.builder \
            .appName("VitalAnalyticsTest") \
            .master("local[2]") \
            .config("spark.sql.shuffle.partitions", "2") \
            .getOrCreate()
        yield spark
        spark.stop()
    
    @pytest.fixture
    def analytics_engine(self, spark_session):
        """Create analytics engine instance"""
        return HealthcareAnalyticsEngine(app_name="TestAnalytics")
    
    @pytest.fixture
    def sample_admissions_data(self, spark_session):
        """Create sample admissions data"""
        data = [
            ("P001", "2024-01-01 08:00:00", "Emergency", "Cardiology"),
            ("P002", "2024-01-01 09:30:00", "Elective", "Orthopedics"),
            ("P003", "2024-01-01 10:15:00", "Emergency", "Neurology"),
            ("P004", "2024-01-01 11:00:00", "Elective", "Cardiology"),
            ("P005", "2024-01-01 12:30:00", "Emergency", "Emergency"),
        ]
        
        schema = StructType([
            StructField("patient_id", StringType(), False),
            StructField("admission_time", StringType(), False),
            StructField("admission_type", StringType(), False),
            StructField("department", StringType(), False),
        ])
        
        return spark_session.createDataFrame(data, schema)
    
    @pytest.fixture
    def sample_discharges_data(self, spark_session):
        """Create sample discharges data"""
        data = [
            ("P001", "2024-01-02 14:00:00", "Home"),
            ("P002", "2024-01-03 10:00:00", "Home"),
            ("P003", "2024-01-02 08:00:00", "Transfer"),
            ("P004", "2024-01-04 16:00:00", "Home"),
            ("P005", "2024-01-01 20:00:00", "Home"),
        ]
        
        schema = StructType([
            StructField("patient_id", StringType(), False),
            StructField("discharge_time", StringType(), False),
            StructField("discharge_disposition", StringType(), False),
        ])
        
        return spark_session.createDataFrame(data, schema)
    
    def test_engine_initialization(self, analytics_engine):
        """Test analytics engine initializes correctly"""
        assert analytics_engine is not None
        assert analytics_engine.spark is not None
        assert analytics_engine.spark.sparkContext.appName == "TestAnalytics"
    
    def test_patient_flow_analysis(self, analytics_engine, sample_admissions_data, sample_discharges_data):
        """Test patient flow analysis"""
        result = analytics_engine.analyze_patient_flow(
            sample_admissions_data,
            sample_discharges_data
        )
        
        assert result is not None
        assert result.count() == 5
        
        # Check columns exist
        assert "patient_id" in result.columns
        assert "los_hours" in result.columns
        assert "department" in result.columns
        
        # Verify length of stay calculations
        los_data = result.select("patient_id", "los_hours").collect()
        assert all(row.los_hours > 0 for row in los_data)
    
    def test_department_statistics(self, analytics_engine, sample_admissions_data, sample_discharges_data):
        """Test department-level statistics"""
        flow_df = analytics_engine.analyze_patient_flow(
            sample_admissions_data,
            sample_discharges_data
        )
        
        dept_stats = analytics_engine.calculate_department_statistics(flow_df)
        
        assert dept_stats is not None
        assert "department" in dept_stats.columns
        assert "avg_los_hours" in dept_stats.columns
        assert "patient_count" in dept_stats.columns
        
        # Verify Cardiology has 2 patients
        cardiology_stats = dept_stats.filter(dept_stats.department == "Cardiology").collect()
        assert len(cardiology_stats) == 1
        assert cardiology_stats[0].patient_count == 2
    
    def test_admission_type_analysis(self, analytics_engine, sample_admissions_data, sample_discharges_data):
        """Test admission type analysis"""
        flow_df = analytics_engine.analyze_patient_flow(
            sample_admissions_data,
            sample_discharges_data
        )
        
        admission_stats = analytics_engine.analyze_admission_types(flow_df)
        
        assert admission_stats is not None
        
        # Verify Emergency and Elective types exist
        types = [row.admission_type for row in admission_stats.collect()]
        assert "Emergency" in types
        assert "Elective" in types
    
    def test_real_time_streaming_setup(self, analytics_engine):
        """Test real-time streaming configuration"""
        # This tests the streaming setup without actually running a stream
        stream_config = analytics_engine.get_streaming_config()
        
        assert stream_config is not None
        assert "checkpointLocation" in stream_config
        assert stream_config["checkpointLocation"] == "/tmp/spark-checkpoint"
    
    def test_data_quality_checks(self, analytics_engine, sample_admissions_data):
        """Test data quality validation"""
        quality_report = analytics_engine.validate_data_quality(sample_admissions_data)
        
        assert quality_report is not None
        assert quality_report["total_records"] == 5
        assert quality_report["null_count"] == 0
        assert quality_report["duplicate_count"] == 0
    
    def test_time_series_aggregation(self, analytics_engine, sample_admissions_data):
        """Test time-series aggregation"""
        hourly_admissions = analytics_engine.aggregate_by_time(
            sample_admissions_data,
            time_column="admission_time",
            interval="1 hour"
        )
        
        assert hourly_admissions is not None
        assert hourly_admissions.count() > 0
    
    def test_outlier_detection(self, analytics_engine, sample_admissions_data, sample_discharges_data):
        """Test outlier detection in length of stay"""
        flow_df = analytics_engine.analyze_patient_flow(
            sample_admissions_data,
            sample_discharges_data
        )
        
        outliers = analytics_engine.detect_los_outliers(flow_df, threshold=2.0)
        
        assert outliers is not None
        # Outliers should be a subset of total patients
        assert outliers.count() <= flow_df.count()
    
    def test_performance_metrics_calculation(self, analytics_engine, sample_admissions_data, sample_discharges_data):
        """Test performance metrics calculation"""
        flow_df = analytics_engine.analyze_patient_flow(
            sample_admissions_data,
            sample_discharges_data
        )
        
        metrics = analytics_engine.calculate_performance_metrics(flow_df)
        
        assert metrics is not None
        assert "avg_los" in metrics
        assert "median_los" in metrics
        assert "total_patients" in metrics
        assert metrics["total_patients"] == 5
    
    def test_readmission_analysis(self, analytics_engine, spark_session):
        """Test readmission analysis"""
        # Create sample readmission data
        data = [
            ("P001", "2024-01-01", "2024-01-15", True),
            ("P002", "2024-01-02", None, False),
            ("P003", "2024-01-03", "2024-02-01", True),
        ]
        
        schema = StructType([
            StructField("patient_id", StringType(), False),
            StructField("first_admission", StringType(), False),
            StructField("readmission_date", StringType(), True),
            StructField("is_readmission", BooleanType(), False),
        ])
        
        readmission_df = spark_session.createDataFrame(data, schema)
        
        readmission_rate = analytics_engine.calculate_readmission_rate(
            readmission_df,
            days_threshold=30
        )
        
        assert readmission_rate is not None
        assert 0 <= readmission_rate <= 100
    
    def test_resource_utilization(self, analytics_engine, spark_session):
        """Test resource utilization analysis"""
        # Create sample resource data
        data = [
            ("Bed_001", "ICU", "2024-01-01", 20, 18),
            ("Bed_002", "ICU", "2024-01-01", 20, 20),
            ("Bed_003", "Ward", "2024-01-01", 30, 25),
        ]
        
        schema = StructType([
            StructField("resource_id", StringType(), False),
            StructField("department", StringType(), False),
            StructField("date", StringType(), False),
            StructField("capacity", IntegerType(), False),
            StructField("occupied", IntegerType(), False),
        ])
        
        resource_df = spark_session.createDataFrame(data, schema)
        
        utilization = analytics_engine.calculate_resource_utilization(resource_df)
        
        assert utilization is not None
        assert "utilization_rate" in utilization.columns
        
        # Verify utilization rates are between 0 and 100
        rates = [row.utilization_rate for row in utilization.collect()]
        assert all(0 <= rate <= 100 for rate in rates)
    
    def test_predictive_capacity_planning(self, analytics_engine, sample_admissions_data):
        """Test predictive capacity planning"""
        forecast = analytics_engine.forecast_admissions(
            sample_admissions_data,
            forecast_days=7
        )
        
        assert forecast is not None
        assert "predicted_admissions" in forecast.columns
        assert forecast.count() == 7  # 7 days forecast
    
    def test_data_aggregation_performance(self, analytics_engine, spark_session):
        """Test performance of large data aggregation"""
        import time
        
        # Create large dataset (10,000 records)
        large_data = [
            (f"P{i:05d}", f"2024-01-{(i % 30) + 1:02d} {(i % 24):02d}:00:00", 
             "Emergency" if i % 2 == 0 else "Elective", 
             ["Cardiology", "Neurology", "Orthopedics"][i % 3])
            for i in range(10000)
        ]
        
        schema = StructType([
            StructField("patient_id", StringType(), False),
            StructField("admission_time", StringType(), False),
            StructField("admission_type", StringType(), False),
            StructField("department", StringType(), False),
        ])
        
        large_df = spark_session.createDataFrame(large_data, schema)
        
        start_time = time.time()
        result = analytics_engine.aggregate_by_department(large_df)
        end_time = time.time()
        
        execution_time = end_time - start_time
        
        assert result is not None
        assert result.count() == 3  # 3 departments
        assert execution_time < 5.0  # Should complete in less than 5 seconds
    
    def test_null_handling(self, analytics_engine, spark_session):
        """Test handling of null values"""
        data = [
            ("P001", "2024-01-01 08:00:00", None, "Cardiology"),
            ("P002", None, "Elective", "Neurology"),
            ("P003", "2024-01-01 10:00:00", "Emergency", None),
        ]
        
        schema = StructType([
            StructField("patient_id", StringType(), False),
            StructField("admission_time", StringType(), True),
            StructField("admission_type", StringType(), True),
            StructField("department", StringType(), True),
        ])
        
        df_with_nulls = spark_session.createDataFrame(data, schema)
        
        cleaned_df = analytics_engine.handle_missing_values(df_with_nulls)
        
        assert cleaned_df is not None
        # Verify nulls are handled (either filled or removed)
        null_counts = cleaned_df.select([
            sum(col(c).isNull().cast("int")).alias(c) 
            for c in cleaned_df.columns
        ]).collect()[0]
        
        # At least some nulls should be handled
        assert sum(null_counts) < 3
    
    def test_concurrent_queries(self, analytics_engine, sample_admissions_data):
        """Test concurrent query execution"""
        import concurrent.futures
        
        def run_query(query_id):
            return analytics_engine.aggregate_by_department(sample_admissions_data)
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            futures = [executor.submit(run_query, i) for i in range(3)]
            results = [f.result() for f in concurrent.futures.as_completed(futures)]
        
        assert len(results) == 3
        assert all(r is not None for r in results)
    
    def test_memory_efficiency(self, analytics_engine, spark_session):
        """Test memory efficiency with large datasets"""
        # Create dataset with 50,000 records
        large_data = [
            (f"P{i:06d}", f"2024-01-{(i % 30) + 1:02d}", i % 100)
            for i in range(50000)
        ]
        
        schema = StructType([
            StructField("patient_id", StringType(), False),
            StructField("date", StringType(), False),
            StructField("value", IntegerType(), False),
        ])
        
        large_df = spark_session.createDataFrame(large_data, schema)
        
        # Should not cause memory errors
        result = analytics_engine.process_large_dataset(large_df)
        assert result is not None
    
    def test_error_handling(self, analytics_engine, spark_session):
        """Test error handling for invalid data"""
        # Create invalid data
        invalid_data = [
            ("P001", "invalid_date", "Emergency", "Cardiology"),
        ]
        
        schema = StructType([
            StructField("patient_id", StringType(), False),
            StructField("admission_time", StringType(), False),
            StructField("admission_type", StringType(), False),
            StructField("department", StringType(), False),
        ])
        
        invalid_df = spark_session.createDataFrame(invalid_data, schema)
        
        # Should handle errors gracefully
        with pytest.raises(Exception):
            analytics_engine.analyze_patient_flow(invalid_df, invalid_df)

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
