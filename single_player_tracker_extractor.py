#!/usr/bin/env python3
"""
Football Player Tracking Data Extractor
Finds the player with highest pose confidence and exports their complete tracking data + ball data
"""

import json
import argparse
from pathlib import Path
from datetime import datetime


def find_best_player(data):
    """
    Find the player with the highest pose confidence across all frames
    """
    best_player = {
        'tracking_id': None,
        'max_confidence': 0.0,
        'first_frame': None
    }
    
    print("🔍 Analyzing pose confidence across all frames...")
    
    for frame_index, frame in enumerate(data['frames']):
        if 'poses' not in frame or 'players' not in frame['poses']:
            continue
            
        for player_index, player in enumerate(frame['poses']['players']):
            if player and 'pose_confidence' in player and 'tracking_id' in player:
                confidence = player['pose_confidence']
                tracking_id = player['tracking_id']
                
                if confidence > best_player['max_confidence']:
                    best_player['max_confidence'] = confidence
                    best_player['tracking_id'] = tracking_id
                    best_player['first_frame'] = frame_index
                    
                    print(f"   New best: Player ID {tracking_id} with confidence {confidence:.4f} in frame {frame_index}")
    
    return best_player


def extract_player_and_ball_data(data, target_tracking_id):
    """
    Extract all data for the selected player and ball across all frames
    """
    extracted_frames = []
    stats = {
        'total_frames': len(data['frames']),
        'frames_with_player': 0,
        'frames_with_ball': 0,
        'frames_with_both': 0
    }
    
    print(f"📊 Extracting data for player ID {target_tracking_id}...")
    
    for frame in data['frames']:
        frame_data = {
            'frame_number': frame['frame_number'],
            'timestamp': frame['timestamp'],
            'field_keypoints': frame.get('field_keypoints'),
            'player_data': None,
            'ball_data': None
        }
        
        player_found = False
        ball_found = False
        
        # Extract player detection data
        if 'detections' in frame and 'players' in frame['detections']:
            for player_detection in frame['detections']['players']:
                if player_detection.get('tracking_id') == target_tracking_id:
                    if not frame_data['player_data']:
                        frame_data['player_data'] = {}
                    frame_data['player_data']['detection'] = player_detection
                    player_found = True
                    break
        
        # Extract player pose data
        if 'poses' in frame and 'players' in frame['poses']:
            for player_pose in frame['poses']['players']:
                if player_pose and player_pose.get('tracking_id') == target_tracking_id:
                    if not frame_data['player_data']:
                        frame_data['player_data'] = {}
                    frame_data['player_data']['pose'] = player_pose
                    player_found = True
                    break
        
        # Extract ball data
        if 'detections' in frame and 'ball' in frame['detections']:
            if frame['detections']['ball']:  # Check if ball list is not empty
                frame_data['ball_data'] = frame['detections']['ball']
                ball_found = True
        
        # Update statistics
        if player_found:
            stats['frames_with_player'] += 1
        if ball_found:
            stats['frames_with_ball'] += 1
        if player_found and ball_found:
            stats['frames_with_both'] += 1
        
        # Only include frames where we have player or ball data
        if player_found or ball_found:
            extracted_frames.append(frame_data)
    
    return extracted_frames, stats


def create_export_data(original_data, best_player, extracted_frames, stats):
    """
    Create the final export data structure
    """
    export_data = {
        'metadata': {
            'export_timestamp': datetime.now().isoformat(),
            'source_metadata': original_data.get('metadata', {}),
            'source_config': original_data.get('config', {}),
            'selected_player': {
                'tracking_id': best_player['tracking_id'],
                'max_pose_confidence': best_player['max_confidence'],
                'first_appearance_frame': best_player['first_frame']
            },
            'export_statistics': {
                'total_source_frames': stats['total_frames'],
                'exported_frames': len(extracted_frames),
                'frames_with_player': stats['frames_with_player'],
                'frames_with_ball': stats['frames_with_ball'],
                'frames_with_both_player_and_ball': stats['frames_with_both']
            }
        },
        'frames': extracted_frames
    }
    
    return export_data


def main():
    parser = argparse.ArgumentParser(description='Extract single player tracking data from football analysis JSON')
    parser.add_argument('input_file', help='Input JSON file path')
    parser.add_argument('-o', '--output', help='Output JSON file path (default: player_ID_tracking.json)')
    parser.add_argument('--player-id', type=int, help='Specific player tracking ID to extract (optional)')
    
    args = parser.parse_args()
    
    # Load input data
    input_path = Path(args.input_file)
    if not input_path.exists():
        print(f"❌ Input file not found: {input_path}")
        return 1
    
    print(f"📖 Loading data from: {input_path}")
    try:
        with open(input_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        print(f"❌ Failed to load JSON: {e}")
        return 1
    
    print(f"✅ Loaded {len(data.get('frames', []))} frames")
    
    # Find best player or use specified player ID
    if args.player_id:
        best_player = {
            'tracking_id': args.player_id,
            'max_confidence': 'N/A (manually specified)',
            'first_frame': 'N/A'
        }
        print(f"🎯 Using manually specified player ID: {args.player_id}")
    else:
        best_player = find_best_player(data)
        if not best_player['tracking_id']:
            print("❌ No players with pose data found in the JSON file")
            return 1
        
        print(f"🏆 Best player: ID {best_player['tracking_id']} with confidence {best_player['max_confidence']:.4f}")
    
    # Extract player and ball data
    extracted_frames, stats = extract_player_and_ball_data(data, best_player['tracking_id'])
    
    # Create export data
    export_data = create_export_data(data, best_player, extracted_frames, stats)
    
    # Determine output filename
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = input_path.parent / f"player_{best_player['tracking_id']}_tracking.json"
    
    # Save exported data
    print(f"💾 Saving to: {output_path}")
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2)
    except Exception as e:
        print(f"❌ Failed to save output: {e}")
        return 1
    
    # Print summary
    print("\n📈 Export Summary:")
    print(f"   Selected Player ID: {best_player['tracking_id']}")
    if isinstance(best_player['max_confidence'], float):
        print(f"   Max Pose Confidence: {best_player['max_confidence']:.4f}")
    print(f"   Total Source Frames: {stats['total_frames']}")
    print(f"   Exported Frames: {len(extracted_frames)}")
    print(f"   Frames with Player: {stats['frames_with_player']}")
    print(f"   Frames with Ball: {stats['frames_with_ball']}")
    print(f"   Frames with Both: {stats['frames_with_both']}")
    
    file_size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"   Output File Size: {file_size_mb:.2f} MB")
    
    print(f"\n✅ Export complete! Saved to: {output_path}")
    return 0


if __name__ == "__main__":
    exit(main())