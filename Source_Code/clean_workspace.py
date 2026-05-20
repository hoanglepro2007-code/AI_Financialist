import os
import shutil
from pathlib import Path

def clean_workspace():
    """
    Organize workspace by moving files to appropriate directories.
    - Moves .pkl and .keras files to models/
    - Moves .json files to metrics/
    """
    # Define target directories
    models_dir = Path("models")
    metrics_dir = Path("metrics")
    
    # Create directories if they don't exist
    models_dir.mkdir(exist_ok=True)
    metrics_dir.mkdir(exist_ok=True)
    
    print(f"✓ Directories created/verified: {models_dir}, {metrics_dir}")
    
    # Scan current directory
    current_dir = Path(".")
    
    # Move .pkl and .keras files to models/
    for ext in ["*.pkl", "*.keras"]:
        for file in current_dir.glob(ext):
            if file.is_file():
                dest = models_dir / file.name
                shutil.move(str(file), str(dest))
                print(f"→ Moved {file.name} to models/")
    
    # Move .json files to metrics/
    for file in current_dir.glob("*.json"):
        if file.is_file():
            dest = metrics_dir / file.name
            shutil.move(str(file), str(dest))
            print(f"→ Moved {file.name} to metrics/")
    
    print("\n✓ Cleanup complete!")

if __name__ == "__main__":
    clean_workspace()
