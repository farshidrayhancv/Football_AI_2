#!/usr/bin/env python3
"""
Benchmark script to compare Roboflow vs Custom YOLO field detection performance
"""

import time
import cv2
import numpy as np
import yaml
import argparse
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
from inference import get_model
from ultralytics import YOLO
import supervision as sv


class FieldDetectionBenchmark:
    def __init__(self, config_path):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.results = {
            'roboflow': {'times': [], 'success': [], 'keypoints': []},
            'custom': {'times': [], 'success': [], 'keypoints': []}
        }
        
    def load_roboflow_model(self):
        """Load Roboflow model."""
        try:
            self.roboflow_model = get_model(
                model_id=self.config['models']['field_detection_model_id'],
                api_key=self.config['api_keys']['roboflow_api_key']
            )
            return True
        except Exception as e:
            print(f"Failed to load Roboflow model: {e}")
            return False
    
    def load_custom_model(self):
        """Load custom YOLO model."""
        try:
            model_path = self.config['models'].get('custom_field_model', 'yolo11n-pose-pitch.pt')
            if not Path(model_path).exists():
                print(f"Custom model not found at: {model_path}")
                return False
            
            self.custom_model = YOLO(model_path)
            # Warm up
            dummy = np.zeros((640, 640, 3), dtype=np.uint8)
            _ = self.custom_model(dummy, verbose=False)
            return True
        except Exception as e:
            print(f"Failed to load custom model: {e}")
            return False
    
    def benchmark_roboflow(self, frame):
        """Benchmark Roboflow model."""
        start_time = time.time()
        
        try:
            result = self.roboflow_model.infer(
                frame,
                confidence=self.config['detection']['keypoint_confidence_threshold']
            )[0]
            
            keypoints = sv.KeyPoints.from_inference(result)
            
            elapsed = time.time() - start_time
            
            # Count high-confidence keypoints
            if keypoints and len(keypoints.confidence) > 0:
                conf_threshold = self.config['detection']['keypoint_confidence_threshold']
                num_keypoints = np.sum(keypoints.confidence[0] > conf_threshold)
                success = num_keypoints >= 4
            else:
                num_keypoints = 0
                success = False
            
            self.results['roboflow']['times'].append(elapsed)
            self.results['roboflow']['success'].append(success)
            self.results['roboflow']['keypoints'].append(num_keypoints)
            
        except Exception as e:
            print(f"Roboflow error: {e}")
            self.results['roboflow']['times'].append(None)
            self.results['roboflow']['success'].append(False)
            self.results['roboflow']['keypoints'].append(0)
    
    def benchmark_custom(self, frame):
        """Benchmark custom YOLO model."""
        start_time = time.time()
        
        try:
            results = self.custom_model(frame, verbose=False)
            
            elapsed = time.time() - start_time
            
            # Extract keypoints
            if (results and len(results) > 0 and 
                hasattr(results[0], 'keypoints') and 
                results[0].keypoints is not None and
                hasattr(results[0].keypoints, 'data')):
                
                kpts_data = results[0].keypoints.data[0].cpu().numpy()
                confidences = kpts_data[:, 2]
                
                conf_threshold = self.config['detection']['keypoint_confidence_threshold']
                num_keypoints = np.sum(confidences > conf_threshold)
                success = num_keypoints >= 4
            else:
                num_keypoints = 0
                success = False
            
            self.results['custom']['times'].append(elapsed)
            self.results['custom']['success'].append(success)
            self.results['custom']['keypoints'].append(num_keypoints)
            
        except Exception as e:
            print(f"Custom model error: {e}")
            self.results['custom']['times'].append(None)
            self.results['custom']['success'].append(False)
            self.results['custom']['keypoints'].append(0)
    
    def run_benchmark(self, video_path, num_frames=100, skip_frames=10):
        """Run benchmark on video frames."""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Cannot open video: {video_path}")
            return
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Load models
        has_roboflow = self.load_roboflow_model()
        has_custom = self.load_custom_model()
        
        if not has_roboflow and not has_custom:
            print("No models available for benchmarking!")
            return
        
        print(f"\nBenchmarking on {num_frames} frames (skipping {skip_frames} between samples)")
        print("-" * 60)
        
        frame_count = 0
        processed = 0
        
        with tqdm(total=num_frames, desc="Benchmarking") as pbar:
            while processed < num_frames and cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                if frame_count % skip_frames == 0:
                    # Resize frame to field detection resolution
                    field_res = self.config.get('resolution', {}).get('field_detection', None)
                    if field_res:
                        frame = cv2.resize(frame, tuple(field_res))
                    
                    # Benchmark both models
                    if has_roboflow:
                        self.benchmark_roboflow(frame)
                    
                    if has_custom:
                        self.benchmark_custom(frame)
                    
                    processed += 1
                    pbar.update(1)
                
                frame_count += 1
        
        cap.release()
        
        # Generate report
        self.generate_report()
    
    def generate_report(self):
        """Generate benchmark report."""
        print("\n" + "="*60)
        print("BENCHMARK RESULTS")
        print("="*60)
        
        for model_name in ['roboflow', 'custom']:
            if not self.results[model_name]['times']:
                continue
            
            times = [t for t in self.results[model_name]['times'] if t is not None]
            if not times:
                continue
            
            success_rate = sum(self.results[model_name]['success']) / len(self.results[model_name]['success']) * 100
            avg_keypoints = np.mean(self.results[model_name]['keypoints'])
            
            print(f"\n{model_name.upper()} Model:")
            print(f"  Average time: {np.mean(times)*1000:.1f} ms")
            print(f"  Min time: {np.min(times)*1000:.1f} ms")
            print(f"  Max time: {np.max(times)*1000:.1f} ms")
            print(f"  Std dev: {np.std(times)*1000:.1f} ms")
            print(f"  Success rate: {success_rate:.1f}%")
            print(f"  Avg keypoints: {avg_keypoints:.1f}")
            print(f"  FPS: {1/np.mean(times):.1f}")
        
        # Create visualization
        self.visualize_results()
    
    def visualize_results(self):
        """Create performance visualization."""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
        
        # 1. Inference time distribution
        for model_name, color in [('roboflow', 'blue'), ('custom', 'green')]:
            times = [t*1000 for t in self.results[model_name]['times'] if t is not None]
            if times:
                ax1.hist(times, bins=30, alpha=0.6, label=model_name, color=color)
        
        ax1.set_xlabel('Inference Time (ms)')
        ax1.set_ylabel('Frequency')
        ax1.set_title('Inference Time Distribution')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Time series of inference times
        for model_name, color in [('roboflow', 'blue'), ('custom', 'green')]:
            times = [t*1000 if t is not None else None for t in self.results[model_name]['times']]
            valid_times = [(i, t) for i, t in enumerate(times) if t is not None]
            if valid_times:
                indices, values = zip(*valid_times)
                ax2.plot(indices, values, label=model_name, color=color, alpha=0.7)
        
        ax2.set_xlabel('Frame Index')
        ax2.set_ylabel('Inference Time (ms)')
        ax2.set_title('Inference Time Over Frames')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Success rate comparison
        model_names = []
        success_rates = []
        colors = []
        
        for model_name, color in [('roboflow', 'blue'), ('custom', 'green')]:
            if self.results[model_name]['success']:
                model_names.append(model_name.capitalize())
                success_rate = sum(self.results[model_name]['success']) / len(self.results[model_name]['success']) * 100
                success_rates.append(success_rate)
                colors.append(color)
        
        ax3.bar(model_names, success_rates, color=colors, alpha=0.7)
        ax3.set_ylabel('Success Rate (%)')
        ax3.set_title('Detection Success Rate')
        ax3.set_ylim(0, 105)
        ax3.grid(True, alpha=0.3, axis='y')
        
        # 4. Keypoints detected
        for model_name, color in [('roboflow', 'blue'), ('custom', 'green')]:
            keypoints = self.results[model_name]['keypoints']
            if keypoints:
                ax4.plot(keypoints, label=model_name, color=color, alpha=0.7)
        
        ax4.set_xlabel('Frame Index')
        ax4.set_ylabel('Number of Keypoints')
        ax4.set_title('Keypoints Detected per Frame')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        ax4.axhline(y=4, color='red', linestyle='--', alpha=0.5, label='Min required')
        
        plt.tight_layout()
        plt.savefig('field_detection_benchmark.png', dpi=150, bbox_inches='tight')
        print(f"\nBenchmark visualization saved to: field_detection_benchmark.png")
        
        # Summary comparison
        print("\n" + "="*60)
        print("PERFORMANCE COMPARISON")
        print("="*60)
        
        if 'roboflow' in self.results and 'custom' in self.results:
            rf_times = [t for t in self.results['roboflow']['times'] if t is not None]
            custom_times = [t for t in self.results['custom']['times'] if t is not None]
            
            if rf_times and custom_times:
                speedup = np.mean(rf_times) / np.mean(custom_times)
                print(f"\nCustom YOLO is {speedup:.2f}x {'faster' if speedup > 1 else 'slower'} than Roboflow")
                
                # Latency comparison
                rf_p95 = np.percentile(rf_times, 95) * 1000
                custom_p95 = np.percentile(custom_times, 95) * 1000
                print(f"\n95th percentile latency:")
                print(f"  Roboflow: {rf_p95:.1f} ms")
                print(f"  Custom:   {custom_p95:.1f} ms")
                
                # Consistency
                rf_std = np.std(rf_times) * 1000
                custom_std = np.std(custom_times) * 1000
                print(f"\nLatency consistency (std dev):")
                print(f"  Roboflow: {rf_std:.1f} ms")
                print(f"  Custom:   {custom_std:.1f} ms")


def main():
    parser = argparse.ArgumentParser(description='Benchmark field detection models')
    parser.add_argument('--config', type=str, default='config.yaml',
                        help='Path to config file')
    parser.add_argument('--video', type=str, required=True,
                        help='Path to test video')
    parser.add_argument('--frames', type=int, default=100,
                        help='Number of frames to benchmark')
    parser.add_argument('--skip', type=int, default=10,
                        help='Skip frames between samples')
    args = parser.parse_args()
    
    # Run benchmark
    benchmark = FieldDetectionBenchmark(args.config)
    benchmark.run_benchmark(args.video, args.frames, args.skip)


if __name__ == "__main__":
    main()