# ⚽ Football AI V2 - Simplified Edition

A streamlined football analysis system using supervision, Roboflow, and Ultralytics models.

## 🚀 Quick Start

1. **Clone and setup**:
```bash
git clone https://github.com/farshidrayhancv/Football_AI_2.git
cd football_ai_v2
pip install -r requirements.txt
```

2. **Configure API keys** in `config.yaml`:
```yaml
api_keys:
  roboflow_api_key: "YOUR_KEY_HERE"
  huggingface_token: "YOUR_TOKEN_HERE"  # Optional
```

3. **Set video paths** in `config.yaml`:
```yaml
video:
  input_path: "path/to/your/video.mp4"
  output_path: "output/result.mp4"
```

4. **Run**:
```bash
python main.py
```

## 📁 Simple Structure

```
football_ai_v2/
├── main.py           # Entry point
├── processor.py      # Core processing logic
├── models.py         # Model management
├── visualizer.py     # Supervision-based visualization
├── config.yaml       # Configuration
├── requirements.txt  # Dependencies
└── README.md         # This file
```

## 📊 **V1 vs V2 Comparison**

### **🏗️ Architecture & Structure**

| Aspect | V1 | V2 |
|--------|----|----|
| **File Structure** | Complex hierarchy with 8+ folders, 30+ files | Flat structure: 4 main Python files |
| **Entry Point** | `python main.py --config config.yaml --output output.mp4` | `python main.py` |
| **Configuration** | 100+ configuration options across multiple sections | Streamlined 20+ essential options |
| **Dependencies** | 15+ complex dependencies with version constraints | 8 core dependencies, auto-resolved |

### **🚀 Ease of Use**

| Feature | V1 | V2 |
|---------|----|----|
| **Setup Complexity** | Manual config of API keys, paths, advanced settings | Just API keys and video paths |
| **Running** | Multiple command-line arguments needed | Single command: `python main.py` |
| **Debugging** | 6+ test scripts, debug modes, extensive logging | Simple error messages, clean output |
| **Docker Support** | Full containerization with GPU support | Native Python execution |

### **🧠 AI Features Comparison**

| Feature | V1 | V2 | Status |
|---------|----|----|--------|
| **Object Detection** | Roboflow + SAHI slicing + adaptive padding | Roboflow models | ✅ **Core maintained** |
| **Field Detection** | Advanced keypoint detection + transformation | Same field detection + transformation | ✅ **Identical** |
| **Team Classification** | SigLIP + sports.common.team.TeamClassifier | Same stable classifier | ✅ **Identical** |
| **Pose Estimation** | YOLO pose + adaptive padding + size-aware crops | YOLO pose + standard padding | ✅ **Core maintained** |
| **Player Segmentation** | SAM + advanced prompting + size-adaptive boxes | SAM + box prompts | ✅ **Core maintained** |
| **Object Tracking** | ByteTrack + smoothing + motion trails | ByteTrack | ✅ **Core maintained** |
| **Possession Detection** | Advanced proximity + coordinate systems + duration tracking | Simple proximity-based | ⚠️ **Simplified** |
| **Tactical View** | Full pitch rendering + Voronoi + ball trails + statistics | Pitch view + ball trails | ✅ **Core maintained** |

### **⚙️ Advanced Features**

| Feature | V1 | V2 | Notes |
|---------|----|----|-------|
| **SAHI Integration** | 2x2 slicing for enhanced small object detection | Not included | Removed for simplicity |
| **Adaptive Padding** | Size-aware padding based on player distance | Fixed 10% padding | Simplified approach |
| **Smart Caching** | Model caching, team classifier persistence | No caching | Removed complexity |
| **Processing Resolution** | Configurable processing vs output resolution | Native resolution processing | Simplified |
| **Batch Video Processing** | `batch_videos.py` for multiple videos | Single video processing | Manual batch processing |
| **Annotation Export** | JSON annotation export for later viewing | Real-time processing only | Removed feature |
| **Streamlit Viewer** | `app.py` for viewing saved annotations | No separate viewer | Simplified workflow |

### **🎨 Visualization & Output**

| Feature | V1 | V2 | Status |
|---------|----|----|--------|
| **Team Coloring** | Consistent across all visualizations | Same consistent coloring | ✅ **Identical** |
| **Pose Visualization** | Team-colored poses + confidence thresholds | Team-colored poses | ✅ **Core maintained** |
| **Segmentation Overlay** | Alpha-blended team-colored masks | Same segmentation overlay | ✅ **Identical** |
| **Possession Highlighting** | Advanced possession indicators + statistics | Simple possession highlighting | ⚠️ **Simplified** |
| **Side-by-Side Layout** | Live view + tactical view + statistics | Live view + tactical view | ✅ **Core maintained** |
| **Ball Tracking** | Advanced trail + trajectory + speed analysis | Basic ball trail | ⚠️ **Simplified** |

### **📝 Configuration Complexity**

**V1 Config (100+ options):**
```yaml
detection:
  confidence_threshold: 0.3
  nms_threshold: 0.5
  padding_ratio: 0.1
  pose_bbox_padding: 50
  pose_bbox_padding_ratio: 0.5
  segmentation_padding: 30
  segmentation_padding_ratio: 0.3
  keypoint_confidence_threshold: 0.5

sahi:
  enable: false
  slice_rows: 2
  slice_cols: 2
  overlap_ratio: 0.2
  postprocess_match_threshold: 0.5

possession_detection:
  coordinate_system: frame
  proximity_threshold: 250
  frame_proximity_threshold: 30
  possession_frames: 1
  possession_duration: 3
  no_possession_frames: 15

processing:
  resolution: [1080, 1080]
  
performance:
  batch_size: 32
  use_gpu: true
  device: cuda
```

**V2 Config (20+ options):**
```yaml
detection:
  confidence_threshold: 0.3
  keypoint_confidence_threshold: 0.5

display:
  show_pose: true
  show_segmentation: true
  team_colors:
    team_1: "#0066FF"
    team_2: "#FF0066"

possession_detection:
  enable: true
  proximity_threshold: 100

performance:
  device: "cuda"
```

### **🎯 When to Use Which Version**

**Choose V1 if you need:**
- ✅ Maximum detection accuracy (SAHI, adaptive padding)
- ✅ Advanced possession analytics with statistics
- ✅ Batch processing of multiple videos
- ✅ Annotation export and later viewing
- ✅ Docker deployment and production scaling
- ✅ Extensive debugging and testing tools
- ✅ Research-grade configurability

**Choose V2 if you want:**
- ✅ Quick setup and immediate results
- ✅ Simple, maintainable codebase
- ✅ Core AI functionality without complexity
- ✅ Easy integration into other projects
- ✅ Learning/educational purposes
- ✅ Prototype development
- ✅ Resource-constrained environments

### **🔄 Migration Path**

**From V1 to V2:**
1. Copy your V1 `config.yaml` and simplify to V2 format
2. Replace complex command: `python main.py --config config.yaml --video input.mp4 --output output.mp4`
3. With simple command: `python main.py`
4. Core AI results will be nearly identical

**From V2 to V1:**
1. Use V2 for prototyping and proof-of-concept
2. Migrate to V1 when you need advanced features or production deployment
3. All AI models and approaches are compatible

---

## ✨ Features

- **Object Detection**: Players, goalkeepers, referees, ball (Roboflow)
- **Field Detection**: Keypoints for pitch transformation (Roboflow)
- **Pose Estimation**: Human poses using YOLO (Ultralytics)
- **Segmentation**: Player segmentation using SAM (Ultralytics)
- **Team Classification**: Stable team assignment using sports library (SigLIP + advanced clustering)
- **Tracking**: ByteTrack for consistent player IDs
- **Possession Detection**: Simple proximity-based possession
- **Tactical View**: Side-by-side live and tactical views

## ⚙️ Configuration

All settings are in `config.yaml`:

### Models
```yaml
models:
  player_detection_model_id: "football-players-detection-3zvbc/11"
  field_detection_model_id: "football-field-detection-f07vi/14"
  pose_model: "yolo11n-pose.pt"
  sam_model: "sam2.1_s.pt"
```

### Display
```yaml
display:
  show_pose: true
  show_segmentation: true
  team_colors:
    team_1: "#00BFFF"
    team_2: "#FF1493"
```

### Performance
```yaml
performance:
  device: "cuda"  # or "cpu"
```

## 🔧 API Keys

### Roboflow (Required)
1. Sign up at [roboflow.com](https://roboflow.com)
2. Get API key from workspace settings
3. Add to `config.yaml`

### Hugging Face (Optional)
1. Sign up at [huggingface.co](https://huggingface.co)
2. Create token at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
3. Add to `config.yaml` (for team classification)

## 🎯 Command Line Usage

```bash
# Basic usage
python main.py

# Custom paths
python main.py --input video.mp4 --output result.mp4

# Custom config
python main.py --config my_config.yaml
```

## 📊 Output

Creates a side-by-side video with:
- **Left**: Original video with AI annotations
- **Right**: Tactical pitch view with player positions

## 🛠️ Troubleshooting

### CUDA Issues
```bash
# Check CUDA availability
python -c "import torch; print(torch.cuda.is_available())"
```

### Model Download Issues
- Ensure internet connection
- Check API keys are valid
- Models are downloaded automatically on first run

### Memory Issues
- Reduce video resolution
- Use CPU mode: `device: "cpu"` in config
- Process shorter video segments

## 🔄 Processing Pipeline

1. **Object Detection** → Detect players, ball, etc.
2. **Field Detection** → Get pitch keypoints
3. **Coordinate Transform** → Frame to pitch coordinates
4. **Tracking** → Assign consistent IDs
5. **Team Classification** → Stable team assignment using sports library
6. **Pose Estimation** → Human pose keypoints
7. **Segmentation** → Player masks
8. **Possession Detection** → Who has the ball
9. **Visualization** → Annotate and combine views

## 🏗️ Architecture

- **supervision**: Main computer vision framework
- **Roboflow**: Object and field detection models
- **Ultralytics**: YOLO pose estimation and SAM segmentation
- **sports**: Pitch coordinate transformations
- **transformers**: Team classification features

## 📄 License

MIT License - see LICENSE file for details.