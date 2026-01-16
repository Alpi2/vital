"""
ONNX Export and Optimization for CNN-LSTM Models

Comprehensive ONNX export with quantization, validation,
and performance benchmarking for production deployment.
"""

import torch
import onnx
import onnxruntime as ort
import numpy as np
from pathlib import Path
import logging
from typing import Dict, Any, Optional, Tuple
import time
import json

logger = logging.getLogger(__name__)

class ONNXExporter:
    """
    Advanced ONNX exporter with optimization and validation.
    
    Features:
    - Dynamic batch size support
    - INT8 quantization
    - Performance benchmarking
    - Model validation
    - Multiple export formats
    """
    
    def __init__(self, model: torch.nn.Module, input_shape: Tuple[int, ...]):
        """
        Initialize ONNX exporter.
        
        Args:
            model: PyTorch model to export
            input_shape: Input tensor shape (batch_size, channels, length)
        """
        self.model = model
        self.input_shape = input_shape
        self.device = next(model.parameters()).device
        
        # Export configuration
        self.export_config = {
            'opset_version': 14,
            'do_constant_folding': True,
            'dynamic_axes': {
                'input': {0: 'batch_size'},
                'output': {0: 'batch_size'}
            },
            'export_params': True,
            'keep_initializers_as_inputs': False,
            'verbose': False
        }
    
    def export_model(
        self,
        output_path: str,
        model_name: str = "model",
        validate: bool = True,
        optimize: bool = True
    ) -> Dict[str, Any]:
        """
        Export PyTorch model to ONNX format.
        
        Args:
            output_path: Output directory path
            model_name: Model name
            validate: Whether to validate exported model
            optimize: Whether to optimize exported model
            
        Returns:
            Export results dictionary
        """
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Create dummy input
        dummy_input = torch.randn(self.input_shape, device=self.device)
        
        # Set model to evaluation mode
        self.model.eval()
        
        # Export paths
        onnx_path = output_path / f"{model_name}.onnx"
        optimized_path = output_path / f"{model_name}_optimized.onnx"
        
        export_results = {
            'model_name': model_name,
            'input_shape': self.input_shape,
            'onnx_path': str(onnx_path),
            'optimized_path': str(optimized_path),
            'validation_passed': False,
            'optimization_applied': False,
            'export_time': 0.0,
            'validation_time': 0.0,
            'optimization_time': 0.0
        }
        
        try:
            # Export model
            start_time = time.time()
            
            torch.onnx.export(
                self.model,
                dummy_input,
                str(onnx_path),
                export_params=self.export_config['export_params'],
                opset_version=self.export_config['opset_version'],
                do_constant_folding=self.export_config['do_constant_folding'],
                input_names=['input'],
                output_names=['output'],
                dynamic_axes=self.export_config['dynamic_axes'],
                keep_initializers_as_inputs=self.export_config['keep_initializers_as_inputs'],
                verbose=self.export_config['verbose']
            )
            
            export_time = time.time() - start_time
            export_results['export_time'] = export_time
            
            logger.info(f"Model exported to ONNX in {export_time:.2f}s: {onnx_path}")
            
            # Validate exported model
            if validate:
                validation_results = self.validate_onnx_model(str(onnx_path))
                export_results.update(validation_results)
                
                if not validation_results['validation_passed']:
                    logger.error("ONNX model validation failed")
                    return export_results
            
            # Optimize model
            if optimize:
                optimization_results = self.optimize_onnx_model(str(onnx_path), str(optimized_path))
                export_results.update(optimization_results)
            
            # Get model information
            model_info = self.get_model_info(str(optimized_path if optimize else str(onnx_path)))
            export_results['model_info'] = model_info
            
            logger.info(f"ONNX export completed successfully")
            return export_results
            
        except Exception as e:
            logger.error(f"ONNX export failed: {e}")
            export_results['error'] = str(e)
            return export_results
    
    def validate_onnx_model(self, onnx_path: str) -> Dict[str, Any]:
        """
        Validate ONNX model.
        
        Args:
            onnx_path: Path to ONNX model
            
        Returns:
            Validation results
        """
        validation_results = {
            'validation_passed': False,
            'validation_time': 0.0,
            'onnx_checker_errors': [],
            'inference_errors': []
        }
        
        try:
            start_time = time.time()
            
            # Load and check ONNX model
            onnx_model = onnx.load(onnx_path)
            
            # Check model
            onnx.checker.check_model(onnx_model)
            
            # Test inference
            ort_session = ort.InferenceSession(onnx_path)
            
            # Create dummy input
            dummy_input = torch.randn(self.input_shape, device=self.device)
            input_dict = {'input': dummy_input.cpu().numpy()}
            
            # Run inference
            ort_outputs = ort_session.run(None, input_dict)
            
            validation_time = time.time() - start_time
            validation_results['validation_time'] = validation_time
            validation_results['validation_passed'] = True
            
            logger.info(f"ONNX model validation passed in {validation_time:.2f}s")
            
        except Exception as e:
            logger.error(f"ONNX model validation failed: {e}")
            validation_results['onnx_checker_errors'].append(str(e))
            validation_results['inference_errors'].append(str(e))
        
        return validation_results
    
    def optimize_onnx_model(self, input_path: str, output_path: str) -> Dict[str, Any]:
        """
        Optimize ONNX model.
        
        Args:
            input_path: Input ONNX model path
            output_path: Output optimized ONNX model path
            
        Returns:
            Optimization results
        """
        optimization_results = {
            'optimization_applied': False,
            'optimization_time': 0.0,
            'optimization_errors': []
        }
        
        try:
            start_time = time.time()
            
            # Load ONNX model
            onnx_model = onnx.load(input_path)
            
            # Apply optimizations
            from onnx import optimizer
            
            # Optimize model
            optimized_model = optimizer.optimize(
                onnx_model,
                fixed_point=False,
                optimize_for_mobile=False,
                optimize_for_runtime=True
            )
            
            # Save optimized model
            onnx.save(optimized_model, output_path)
            
            optimization_time = time.time() - start_time
            optimization_results['optimization_time'] = optimization_time
            optimization_results['optimization_applied'] = True
            
            logger.info(f"ONNX model optimization completed in {optimization_time:.2f}s")
            
        except Exception as e:
            logger.error(f"ONNX model optimization failed: {e}")
            optimization_results['optimization_errors'].append(str(e))
        
        return optimization_results
    
    def quantize_model(
        self,
        onnx_path: str,
        calibration_data: Optional[np.ndarray] = None,
        output_path: Optional[str] = None,
        quantization_type: str = 'int8'
    ) -> Dict[str, Any]:
        """
        Quantize ONNX model to INT8 for better performance.
        
        Args:
            onnx_path: Path to ONNX model
            calibration_data: Calibration dataset
            output_path: Output quantized model path
            quantization_type: Type of quantization ('int8', 'uint8', 'dynamic')
            
        Returns:
            Quantization results
        """
        if output_path is None:
            output_path = onnx_path.replace('.onnx', f'_{quantization_type}.onnx')
        
        quantization_results = {
            'quantization_applied': False,
            'quantization_type': quantization_type,
            'quantization_time': 0.0,
            'quantization_errors': []
        }
        
        try:
            start_time = time.time()
            
            if quantization_type == 'dynamic':
                # Dynamic quantization
                from onnxruntime.quantization import quantize_dynamic
                
                quantized_model = quantize_dynamic(
                    onnx_path,
                    weight_type=quantization_type
                )
                
            else:
                # Static quantization with calibration data
                if calibration_data is None:
                    # Generate synthetic calibration data
                    calibration_data = self._generate_calibration_data()
                
                from onnxruntime.quantization import quantize_static
                
                quantized_model = quantize_static(
                    onnx_path,
                    calibration_data,
                    weight_type=quantization_type
                )
            
            # Save quantized model
            quantized_model.save_model_to_file(output_path)
            
            quantization_time = time.time() - start_time
            quantization_results['quantization_time'] = quantization_time
            quantization_results['quantization_applied'] = True
            
            logger.info(f"Model quantized to {quantization_type} in {quantization_time:.2f}s")
            
        except Exception as e:
            logger.error(f"Model quantization failed: {e}")
            quantization_results['quantization_errors'].append(str(e))
        
        return quantization_results
    
    def _generate_calibration_data(self, num_samples: int = 100) -> np.ndarray:
        """
        Generate synthetic calibration data for quantization.
        
        Args:
            num_samples: Number of calibration samples
            
        Returns:
            Calibration data array
        """
        self.model.eval()
        calibration_data = []
        
        with torch.no_grad():
            for _ in range(num_samples):
                # Generate random input
                dummy_input = torch.randn(self.input_shape, device=self.device)
                
                # Get model output
                output = self.model(dummy_input)
                
                # Store input
                calibration_data.append(dummy_input.cpu().numpy())
        
        return np.array(calibration_data)
    
    def benchmark_models(
        self,
        original_path: str,
        optimized_path: Optional[str] = None,
        quantized_path: Optional[str] = None,
        num_iterations: int = 100
    ) -> Dict[str, Any]:
        """
        Benchmark different model versions.
        
        Args:
            original_path: Original ONNX model path
            optimized_path: Optimized ONNX model path
            quantized_path: Quantized ONNX model path
            num_iterations: Number of benchmark iterations
            
        Returns:
            Benchmark results
        """
        benchmark_results = {
            'original_model': {},
            'optimized_model': {},
            'quantized_model': {},
            'comparison': {}
        }
        
        # Prepare test data
        test_inputs = [torch.randn(self.input_shape, device=self.device) for _ in range(num_iterations)]
        test_inputs_np = [inp.cpu().numpy() for inp in test_inputs]
        
        # Benchmark original model
        if original_path:
            benchmark_results['original_model'] = self._benchmark_single_model(
                original_path, test_inputs_np, "Original"
            )
        
        # Benchmark optimized model
        if optimized_path:
            benchmark_results['optimized_model'] = self._benchmark_single_model(
                optimized_path, test_inputs_np, "Optimized"
            )
        
        # Benchmark quantized model
        if quantized_path:
            benchmark_results['quantized_model'] = self._benchmark_single_model(
                quantized_path, test_inputs_np, "Quantized"
            )
        
        # Calculate comparisons
        if (optimized_path and 'optimized_model' in benchmark_results and 
            'original_model' in benchmark_results):
            original_time = benchmark_results['original_model']['avg_inference_time']
            optimized_time = benchmark_results['optimized_model']['avg_inference_time']
            speedup = original_time / optimized_time if optimized_time > 0 else 0
            benchmark_results['comparison']['optimized_speedup'] = speedup
        
        if (quantized_path and 'quantized_model' in benchmark_results and 
            'original_model' in benchmark_results):
            original_time = benchmark_results['original_model']['avg_inference_time']
            quantized_time = benchmark_results['quantized_model']['avg_inference_time']
            speedup = original_time / quantized_time if quantized_time > 0 else 0
            benchmark_results['comparison']['quantized_speedup'] = speedup
        
        return benchmark_results
    
    def _benchmark_single_model(
        self,
        model_path: str,
        test_inputs: List[np.ndarray],
        model_name: str
    ) -> Dict[str, Any]:
        """
        Benchmark a single model.
        
        Args:
            model_path: Path to model
            test_inputs: Test input data
            model_name: Model name for logging
            
        Returns:
            Benchmark results
        """
        try:
            # Create inference session
            session = ort.InferenceSession(model_path)
            
            # Warm up
            session.run(None, {'input': test_inputs[0]})
            
            # Benchmark
            inference_times = []
            
            for test_input in test_inputs:
                start_time = time.time()
                outputs = session.run(None, {'input': test_input})
                end_time = time.time()
                inference_times.append(end_time - start_time)
            
            # Calculate statistics
            avg_time = np.mean(inference_times)
            std_time = np.std(inference_times)
            min_time = np.min(inference_times)
            max_time = np.max(inference_times)
            
            results = {
                'model_name': model_name,
                'model_path': model_path,
                'num_iterations': len(test_inputs),
                'avg_inference_time': avg_time,
                'std_inference_time': std_time,
                'min_inference_time': min_time,
                'max_inference_time': max_time,
                'throughput': 1.0 / avg_time if avg_time > 0 else 0
            }
            
            logger.info(f"{model_name} benchmark: avg={avg_time*1000:.2f}ms, "
                       f"std={std_time*1000:.2f}ms, throughput={results['throughput']:.2f} req/s")
            
            return results
            
        except Exception as e:
            logger.error(f"Benchmark failed for {model_name}: {e}")
            return {'error': str(e)}
    
    def get_model_info(self, onnx_path: str) -> Dict[str, Any]:
        """
        Get detailed information about ONNX model.
        
        Args:
            onnx_path: Path to ONNX model
            
        Returns:
            Model information dictionary
        """
        try:
            # Load ONNX model
            onnx_model = onnx.load(onnx_path)
            
            # Create inference session to get details
            session = ort.InferenceSession(onnx_path)
            
            # Get input/output information
            inputs = session.get_inputs()
            outputs = session.get_outputs()
            
            # Calculate model size
            model_size = Path(onnx_path).stat().st_size / (1024 * 1024)  # MB
            
            # Get graph information
            graph = onnx_model.graph
            num_nodes = len(graph.node)
            num_initializers = len(graph.initializer)
            
            model_info = {
                'model_path': onnx_path,
                'model_size_mb': model_size,
                'opset_version': onnx_model.opset_import[0].version if onnx_model.opset_import else 0,
                'producer_name': onnx_model.producer_name or 'Unknown',
                'producer_version': onnx_model.producer_version or 'Unknown',
                'domain': onnx_model.domain or '',
                'num_nodes': num_nodes,
                'num_initializers': num_initializers,
                'inputs': [
                    {
                        'name': inp.name,
                        'type': str(inp.type),
                        'shape': list(inp.shape) if inp.shape else 'dynamic'
                    }
                    for inp in inputs
                ],
                'outputs': [
                    {
                        'name': out.name,
                        'type': str(out.type),
                        'shape': list(out.shape) if out.shape else 'dynamic'
                    }
                    for out in outputs
                ]
            }
            
            return model_info
            
        except Exception as e:
            logger.error(f"Failed to get model info: {e}")
            return {'error': str(e)}
    
    def export_complete_pipeline(
        self,
        output_dir: str,
        model_name: str = "cnn_lstm_model",
        calibration_data: Optional[np.ndarray] = None,
        benchmark_iterations: int = 100
    ) -> Dict[str, Any]:
        """
        Run complete export pipeline.
        
        Args:
            output_dir: Output directory
            model_name: Model name
            calibration_data: Calibration data for quantization
            benchmark_iterations: Number of benchmark iterations
            
        Returns:
            Complete pipeline results
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        pipeline_results = {
            'model_name': model_name,
            'output_dir': str(output_dir),
            'export_results': {},
            'quantization_results': {},
            'benchmark_results': {},
            'total_time': 0.0
        }
        
        start_time = time.time()
        
        try:
            # Step 1: Export to ONNX
            logger.info("Step 1: Exporting model to ONNX...")
            export_results = self.export_model(
                str(output_dir),
                model_name,
                validate=True,
                optimize=True
            )
            pipeline_results['export_results'] = export_results
            
            if not export_results.get('validation_passed', False):
                logger.error("Model export validation failed, stopping pipeline")
                return pipeline_results
            
            # Step 2: Quantize model
            logger.info("Step 2: Quantizing model...")
            onnx_path = export_results['onnx_path']
            quantization_results = self.quantize_model(
                onnx_path,
                calibration_data=calibration_data,
                output_path=str(output_dir / f"{model_name}_quantized.onnx"),
                quantization_type='int8'
            )
            pipeline_results['quantization_results'] = quantization_results
            
            # Step 3: Benchmark models
            logger.info("Step 3: Benchmarking models...")
            optimized_path = export_results.get('optimized_path')
            quantized_path = None
            
            if quantization_results.get('quantization_applied', False):
                quantized_path = str(output_dir / f"{model_name}_quantized.onnx")
            
            benchmark_results = self.benchmark_models(
                onnx_path,
                optimized_path,
                quantized_path,
                benchmark_iterations
            )
            pipeline_results['benchmark_results'] = benchmark_results
            
            # Step 4: Generate report
            logger.info("Step 4: Generating export report...")
            report_path = output_dir / f"{model_name}_export_report.json"
            self._generate_export_report(pipeline_results, str(report_path))
            
            total_time = time.time() - start_time
            pipeline_results['total_time'] = total_time
            
            logger.info(f"Complete export pipeline finished in {total_time:.2f}s")
            
        except Exception as e:
            logger.error(f"Export pipeline failed: {e}")
            pipeline_results['error'] = str(e)
        
        return pipeline_results
    
    def _generate_export_report(self, results: Dict[str, Any], report_path: str):
        """
        Generate comprehensive export report.
        
        Args:
            results: Pipeline results
            report_path: Path to save report
        """
        report = {
            'export_summary': {
                'model_name': results['model_name'],
                'output_directory': results['output_dir'],
                'total_time': results['total_time'],
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
            },
            'export_results': results['export_results'],
            'quantization_results': results['quantization_results'],
            'benchmark_results': results['benchmark_results']
        }
        
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Export report saved to {report_path}")

def export_cnn_lstm_model(
    model: torch.nn.Module,
    output_dir: str,
    model_name: str = "cnn_lstm_ecg",
    input_shape: Tuple[int, ...] = (1, 1, 1000),
    calibration_data: Optional[np.ndarray] = None
) -> Dict[str, Any]:
    """
    Export CNN-LSTM model to ONNX with full pipeline.
    
    Args:
        model: PyTorch model
        output_dir: Output directory
        model_name: Model name
        input_shape: Input tensor shape
        calibration_data: Calibration data for quantization
        
    Returns:
        Export results
    """
    exporter = ONNXExporter(model, input_shape)
    
    return exporter.export_complete_pipeline(
        output_dir=output_dir,
        model_name=model_name,
        calibration_data=calibration_data,
        benchmark_iterations=100
    )

if __name__ == "__main__":
    # Test ONNX export
    import sys
    sys.path.append('../models')
    from cnn_lstm_attention import CNNLSTMAttention
    
    # Create model
    model = CNNLSTMAttention(num_classes=5, input_channels=1, sequence_length=1000)
    
    # Export model
    results = export_cnn_lstm_model(
        model=model,
        output_dir="onnx_exports",
        model_name="test_cnn_lstm",
        input_shape=(1, 1, 1000)
    )
    
    print("ONNX Export Results:")
    print(json.dumps(results, indent=2))
