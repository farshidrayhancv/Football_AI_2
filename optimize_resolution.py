#!/usr/bin/env python3
"""
Football AI V2 - Processing Resolution Optimizer
Tool to help users choose the optimal processing resolution for their system and video
"""

import argparse
import cv2
import time
import numpy as np
import os
import yaml
import torch
from processor import FootballProcessor


def get_video_info(video_path):
    """Get detailed video information."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None
    
    info = {
        'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        'fps': cap.get(cv2.CAP_PROP_FPS),
        'total_frames': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        'duration': cap.get(cv2.CAP_PROP_FRAME_COUNT) / cap.get(cv2.CAP_PROP_FPS)
    }
    
    cap.release()
    return info


def benchmark_resolution(config, video_path, resolution, test_frames=10):
    """Benchmark processing speed at a specific resolution."""
    # Temporarily set the resolution
    original_resolution = config.get('processing', {}).get('resolution', None)
    config['processing']['resolution'] = resolution
    
    # Create processor
    try:
        processor = FootballProcessor(config, config['performance']['device'])
    except Exception as e:
        print(f"Failed to initialize processor: {e}")
        return None
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None
    
    # Benchmark on sample frames
    times = []
    
    for i in range(test_frames):
        # Seek to a different position each time
        frame_pos = (i * 100) % int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_pos)
        
        ret, frame = cap.read()
        if not ret:
            break
        
        # Time the processing
        start_time = time.time()
        try:
            _ = processor.process_frame(frame)
            end_time = time.time()
            times.append(end_time - start_time)
        except Exception as e:
            print(f"Error processing frame: {e}")
            continue
    
    cap.release()
    
    # Restore original resolution
    config['processing']['resolution'] = original_resolution
    
    if not times:
        return None
    
    return {
        'avg_time': np.mean(times),
        'min_time': np.min(times),
        'max_time': np.max(times),
        'std_time': np.std(times)
    }


def suggest_resolutions(video_width, video_height):
    """Suggest appropriate processing resolutions based on video size."""
    video_pixels = video_width * video_height
    
    suggestions = []
    
    # Common resolutions with their characteristics
    resolutions = [
        (640, 360, "Ultra Fast", "4-6x speedup, may reduce accuracy"),
        (854, 480, "Very Fast", "3-4x speedup, good for distant cameras"),
        (1280, 720, "Fast", "2-3x speedup, balanced quality/speed"),
        (1600, 900, "Moderate", "1.5-2x speedup, high quality"),
        (1920, 1080, "Quality", "Minor speedup, near-native quality"),
    ]
    
    for width, height, speed_desc, quality_desc in resolutions:
        res_pixels = width * height
        if res_pixels <= video_pixels:  # Only suggest smaller or equal resolutions
            speedup = video_pixels / res_pixels
            suggestions.append({
                'resolution': [width, height],
                'description': speed_desc,
                'details': quality_desc,
                'speedup': speedup,
                'pixels': res_pixels
            })
    
    # Add native resolution
    suggestions.append({
        'resolution': None,
        'description': "Native",
        'details': "Original video resolution, highest quality",
        'speedup': 1.0,
        'pixels': video_pixels
    })
    
    return suggestions


def main():
    parser = argparse.ArgumentParser(description='Optimize processing resolution for Football AI V2')
    parser.add_argument('--config', type=str, default='config.yaml',
                        help='Path to config file')
    parser.add_argument('--video', type=str, required=True,
                        help='Path to test video file')
    parser.add_argument('--benchmark', action='store_true',
                        help='Run detailed benchmarks (slower but more accurate)')
    parser.add_argument('--frames', type=int, default=10,
                        help='Number of frames to test for benchmarking')
    args = parser.parse_args()
    
    # Load config
    if not os.path.exists(args.config):
        print(f"❌ Config file not found: {args.config}")
        return 1
    
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Check video
    if not os.path.exists(args.video):
        print(f"❌ Video file not found: {args.video}")
        return 1
    
    # Get video info
    video_info = get_video_info(args.video)
    if video_info is None:
        print(f"❌ Cannot read video file: {args.video}")
        return 1
    
    print(f"📹 Video Analysis: {os.path.basename(args.video)}")
    print(f"   Resolution: {video_info['width']}x{video_info['height']}")
    print(f"   FPS: {video_info['fps']:.2f}")
    print(f"   Duration: {video_info['duration']:.1f} seconds")
    print(f"   Total Frames: {video_info['total_frames']}")
    
    # Get system info
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\n🖥️ System Information:")
    print(f"   Device: {device.upper()}")
    if device == 'cuda':
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
        memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"   GPU Memory: {memory:.1f} GB")
    
    # Get resolution suggestions
    suggestions = suggest_resolutions(video_info['width'], video_info['height'])
    
    print(f"\n🎯 Resolution Recommendations:")
    print(f"{'Resolution':<12} {'Mode':<12} {'Speedup':<10} {'Description'}")
    print("-" * 60)
    
    for suggestion in suggestions:
        if suggestion['resolution'] is None:
            res_str = "Native"
        else:
            res_str = f"{suggestion['resolution'][0]}x{suggestion['resolution'][1]}"
        
        speedup_str = f"{suggestion['speedup']:.1f}x" if suggestion['speedup'] > 1 else "1.0x"
        
        print(f"{res_str:<12} {suggestion['description']:<12} {speedup_str:<10} {suggestion['details']}")
    
    # Run benchmarks if requested
    if args.benchmark:
        print(f"\n⏱️ Running benchmarks...")
        print("This may take a few minutes...")
        
        benchmark_results = []
        
        for suggestion in suggestions:
            resolution = suggestion['resolution']
            
            if resolution is None:
                res_str = "Native"
            else:
                res_str = f"{resolution[0]}x{resolution[1]}"
            
            print(f"   Testing {res_str}...", end=" ", flush=True)
            
            result = benchmark_resolution(config, args.video, resolution, args.frames)
            
            if result is not None:
                fps = 1.0 / result['avg_time']
                print(f"{result['avg_time']:.3f}s avg ({fps:.1f} FPS)")
                benchmark_results.append({
                    'resolution': resolution,
                    'resolution_str': res_str,
                    'avg_time': result['avg_time'],
                    'fps': fps,
                    'suggestion': suggestion
                })
            else:
                print("Failed")
        
        # Display benchmark results
        if benchmark_results:
            print(f"\n📊 Benchmark Results:")
            print(f"{'Resolution':<12} {'Avg Time':<10} {'FPS':<8} {'Real-time':<12} {'Recommendation'}")
            print("-" * 80)
            
            # Sort by FPS (fastest first)
            benchmark_results.sort(key=lambda x: x['fps'], reverse=True)
            
            for result in benchmark_results:
                avg_time = result['avg_time']
                fps = result['fps']
                
                # Calculate real-time factor
                video_fps = video_info['fps']
                realtime_factor = fps / video_fps
                
                if realtime_factor >= 1.0:
                    realtime_str = f"{realtime_factor:.1f}x real-time"
                else:
                    realtime_str = f"{1/realtime_factor:.1f}x slower"
                
                # Recommendation
                if realtime_factor >= 2.0:
                    recommendation = "Excellent"
                elif realtime_factor >= 1.0:
                    recommendation = "Good"
                elif realtime_factor >= 0.5:
                    recommendation = "Acceptable"
                else:
                    recommendation = "Too slow"
                
                print(f"{result['resolution_str']:<12} {avg_time:.3f}s{'':<2} {fps:.1f}{'':<3} {realtime_str:<12} {recommendation}")
            
            # Provide specific recommendation
            best_realtime = None
            best_balanced = None
            
            for result in benchmark_results:
                realtime_factor = result['fps'] / video_info['fps']
                
                if realtime_factor >= 1.0 and best_realtime is None:
                    best_realtime = result
                
                if realtime_factor >= 0.8 and best_balanced is None:
                    best_balanced = result
            
            print(f"\n💡 Recommendations:")
            
            if best_realtime:
                print(f"   ⚡ For real-time processing: {best_realtime['resolution_str']}")
            
            if best_balanced and best_balanced != best_realtime:
                print(f"   ⚖️ For balanced quality/speed: {best_balanced['resolution_str']}")
            
            # Find the highest quality that's still reasonable
            quality_choice = None
            for result in reversed(benchmark_results):  # Start from slowest/highest quality
                realtime_factor = result['fps'] / video_info['fps']
                if realtime_factor >= 0.3:  # At least 30% of real-time
                    quality_choice = result
                    break
            
            if quality_choice and quality_choice not in [best_realtime, best_balanced]:
                print(f"   🎯 For best quality: {quality_choice['resolution_str']}")
    
    # Provide configuration suggestions
    print(f"\n⚙️ Configuration Suggestions:")
    print(f"Add to your config.yaml:")
    print(f"")
    print(f"processing:")
    
    if args.benchmark and benchmark_results:
        # Use benchmark results for recommendation
        best = benchmark_results[0]  # Fastest
        for result in benchmark_results:
            realtime_factor = result['fps'] / video_info['fps']
            if realtime_factor >= 1.0:  # Real-time capable
                best = result
                break
        
        if best['resolution'] is None:
            print(f"  resolution: null  # Native resolution")
        else:
            print(f"  resolution: [{best['resolution'][0]}, {best['resolution'][1]}]  # {best['resolution_str']}")
    else:
        # Use heuristic recommendation
        video_pixels = video_info['width'] * video_info['height']
        
        if video_pixels > 1920 * 1080:
            print(f"  resolution: [1920, 1080]  # Recommended for 4K+ videos")
        elif video_pixels > 1280 * 720:
            print(f"  resolution: [1280, 720]   # Recommended for HD videos")
        else:
            print(f"  resolution: null           # Native resolution for smaller videos")
    
    print(f"\n✅ Analysis complete!")
    return 0


if __name__ == "__main__":
    exit_code = main()
    exit(exit_code)