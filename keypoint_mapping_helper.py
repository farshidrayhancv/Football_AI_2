#!/usr/bin/env python3
"""
Helper script to create keypoint mapping between custom YOLO model and pitch vertices
This helps ensure your custom model's keypoints align with the expected pitch points
"""

import cv2
import numpy as np
import yaml
from ultralytics import YOLO
from sports.configs.soccer import SoccerPitchConfiguration
from sports.annotators.soccer import draw_pitch
import matplotlib.pyplot as plt
from pathlib import Path


def analyze_keypoint_order(model_path, test_image_path):
    """Analyze the keypoint order from your custom YOLO model."""
    print(f"Analyzing keypoint order from: {model_path}")
    
    # Load model
    model = YOLO(model_path)
    
    # Load test image
    if test_image_path:
        frame = cv2.imread(test_image_path)
    else:
        # Create a dummy frame
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    
    # Run inference
    results = model(frame, verbose=False)
    
    if (results and len(results) > 0 and 
        hasattr(results[0], 'keypoints') and 
        results[0].keypoints is not None):
        
        if hasattr(results[0].keypoints, 'data'):
            num_keypoints = results[0].keypoints.data.shape[1]
        else:
            num_keypoints = results[0].keypoints.shape[1]
        
        print(f"Model outputs {num_keypoints} keypoints")
        
        # Get model's class names if available
        if hasattr(model, 'names') and model.names:
            print("\nModel keypoint names:")
            for i, name in enumerate(model.names):
                print(f"  {i}: {name}")
        
        return num_keypoints
    else:
        print("Could not determine number of keypoints")
        return 0


def show_pitch_vertices():
    """Display the expected pitch vertices from SoccerPitchConfiguration."""
    config = SoccerPitchConfiguration()
    vertices = np.array(config.vertices)
    
    print(f"\nSoccerPitchConfiguration has {len(vertices)} vertices:")
    print("-" * 60)
    
    # Common pitch keypoint descriptions (adjust based on your dataset)
    vertex_descriptions = [
        "Pitch corner (top-left)",
        "Pitch corner (top-right)", 
        "Pitch corner (bottom-right)",
        "Pitch corner (bottom-left)",
        "Penalty area (top-left)",
        "Penalty area (top-right)",
        "Penalty area (bottom-right)",
        "Penalty area (bottom-left)",
        "Goal area (top-left)",
        "Goal area (top-right)",
        "Goal area (bottom-right)",
        "Goal area (bottom-left)",
        "Center circle (top)",
        "Center circle (right)",
        "Center circle (bottom)",
        "Center circle (left)",
        "Center spot",
        "Penalty spot (left)",
        "Penalty spot (right)",
        # Add more based on your specific model
    ]
    
    for i, vertex in enumerate(vertices):
        desc = vertex_descriptions[i] if i < len(vertex_descriptions) else f"Point {i}"
        print(f"{i:2d}: ({vertex[0]:6.1f}, {vertex[1]:6.1f}) - {desc}")
    
    # Visualize pitch with numbered vertices
    pitch = draw_pitch(config)
    
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.imshow(pitch)
    
    # Plot and number each vertex
    for i, vertex in enumerate(vertices):
        # Scale vertex to image coordinates (pitch image is typically 1050x680)
        scale_x = pitch.shape[1] / config.length
        scale_y = pitch.shape[0] / config.width
        x = vertex[0] * scale_x
        y = vertex[1] * scale_y
        
        ax.plot(x, y, 'ro', markersize=8)
        ax.text(x+5, y+5, str(i), fontsize=10, color='red', weight='bold')
    
    ax.set_title("Pitch Vertices (Numbered)")
    ax.axis('off')
    
    plt.tight_layout()
    plt.savefig('pitch_vertices.png', dpi=150, bbox_inches='tight')
    print(f"\nPitch vertices visualization saved to: pitch_vertices.png")
    
    return vertices


def create_mapping_template(num_model_keypoints, num_pitch_vertices):
    """Create a template mapping configuration."""
    print(f"\n{'='*60}")
    print("KEYPOINT MAPPING TEMPLATE")
    print(f"{'='*60}")
    print(f"Your model outputs {num_model_keypoints} keypoints")
    print(f"Pitch configuration has {num_pitch_vertices} vertices")
    
    template = {
        'keypoint_mapping': {
            'description': 'Maps model keypoint indices to pitch vertex indices',
            'mapping': {}
        }
    }
    
    # Create a suggested mapping (you'll need to adjust this)
    if num_model_keypoints >= 4:
        # Basic corner mapping as a starting point
        template['keypoint_mapping']['mapping'] = {
            0: 0,   # Model keypoint 0 -> Pitch vertex 0 (top-left corner)
            1: 1,   # Model keypoint 1 -> Pitch vertex 1 (top-right corner)
            2: 2,   # Model keypoint 2 -> Pitch vertex 2 (bottom-right corner)
            3: 3,   # Model keypoint 3 -> Pitch vertex 3 (bottom-left corner)
        }
        
        # Add more mappings if model has more keypoints
        for i in range(4, min(num_model_keypoints, num_pitch_vertices)):
            template['keypoint_mapping']['mapping'][i] = i
    
    print("\nSuggested mapping configuration to add to your config.yaml:")
    print(yaml.dump(template, default_flow_style=False))
    
    print("\nIMPORTANT: You need to manually verify and adjust this mapping!")
    print("1. Test your model on a video frame")
    print("2. Visualize which keypoints are detected where")
    print("3. Match them to the corresponding pitch vertices")
    print("4. Update the mapping in your config.yaml")
    
    return template


def test_mapping(config_path, model_path, test_frame):
    """Test a keypoint mapping configuration."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    mapping = config.get('keypoint_mapping', {}).get('mapping', {})
    
    if not mapping:
        print("No keypoint mapping found in config!")
        return
    
    print(f"\nTesting mapping with {len(mapping)} keypoint pairs")
    
    # Load model and test
    model = YOLO(model_path)
    results = model(test_frame, verbose=False)
    
    if results and len(results) > 0 and hasattr(results[0], 'keypoints'):
        kpts = results[0].keypoints.data[0].cpu().numpy()
        
        print("\nMapping verification:")
        print("-" * 60)
        
        pitch_config = SoccerPitchConfiguration()
        pitch_vertices = np.array(pitch_config.vertices)
        
        for model_idx, pitch_idx in mapping.items():
            model_idx = int(model_idx)
            pitch_idx = int(pitch_idx)
            
            if model_idx < len(kpts):
                x, y, conf = kpts[model_idx]
                pitch_vertex = pitch_vertices[pitch_idx]
                
                print(f"Model KP {model_idx} ({x:.1f}, {y:.1f}, conf={conf:.2f}) "
                      f"-> Pitch vertex {pitch_idx} ({pitch_vertex[0]:.1f}, {pitch_vertex[1]:.1f})")
            else:
                print(f"Model KP {model_idx} -> Not detected in this frame")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Helper for mapping YOLO keypoints to pitch vertices')
    parser.add_argument('--model', type=str, required=True,
                        help='Path to custom YOLO pose model')
    parser.add_argument('--image', type=str, default=None,
                        help='Test image path (optional)')
    parser.add_argument('--config', type=str, default='config.yaml',
                        help='Config file to test mapping')
    parser.add_argument('--test-mapping', action='store_true',
                        help='Test existing mapping from config')
    args = parser.parse_args()
    
    # Check model exists
    if not Path(args.model).exists():
        print(f"Error: Model not found at {args.model}")
        return
    
    print("=" * 60)
    print("YOLO PITCH KEYPOINT MAPPING HELPER")
    print("=" * 60)
    
    # Step 1: Analyze model
    num_model_kpts = analyze_keypoint_order(args.model, args.image)
    
    # Step 2: Show pitch vertices
    vertices = show_pitch_vertices()
    
    # Step 3: Create or test mapping
    if args.test_mapping and args.image:
        frame = cv2.imread(args.image)
        test_mapping(args.config, args.model, frame)
    else:
        # Create mapping template
        mapping = create_mapping_template(num_model_kpts, len(vertices))
        
        # Save template
        with open('keypoint_mapping_template.yaml', 'w') as f:
            yaml.dump(mapping, f, default_flow_style=False)
        print(f"\nMapping template saved to: keypoint_mapping_template.yaml")
        
        print("\n" + "="*60)
        print("NEXT STEPS:")
        print("="*60)
        print("1. Run the test_field_detection.py script to visualize your keypoints")
        print("2. Compare detected keypoints with pitch_vertices.png")
        print("3. Update the mapping in keypoint_mapping_template.yaml")
        print("4. Add the mapping to your config.yaml")
        print("5. Test again with --test-mapping flag")


if __name__ == "__main__":
    main()