import ee
import yaml
from pathlib import Path

def load_config():
    config_path = Path(__file__).resolve().parent.parent / 'config' / 'project_config.yaml'
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def initialize_ee():
    config = load_config()
    project_id = config['gee']['project_id']
    
    if project_id == "YOUR-GEE-PROJECT-ID":
        raise ValueError("Please update 'project_id' in config/project_config.yaml")
        
    try:
        # High-level API initialization
        ee.Initialize(project=project_id)
        print(f"[GEE] Successfully initialized Earth Engine for project: {project_id}")
    except Exception as e:
        print(f"[ERROR] Earth Engine Initialization Failed: {e}")
        print("Run 'earthengine authenticate' in your terminal.")
        raise e

def get_delhi_geometry():
    config = load_config()
    bbox = config['study_area']['bbox']
    # Format: [min_lon, min_lat, max_lon, max_lat]
    return ee.Geometry.BBox(bbox[0], bbox[1], bbox[2], bbox[3])