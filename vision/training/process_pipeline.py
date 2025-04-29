import os
import time
import subprocess
import platform
import sys
import pkg_resources
import importlib.util

def run_step(step_name, command):
    """Run a step in the processing pipeline"""
    print(f"\n{'='*80}")
    print(f"  STEP: {step_name}")
    print(f"{'='*80}\n")
    start_time = time.time()
    result = subprocess.run(command, shell=True)
    end_time = time.time()
    
    if result.returncode == 0:
        print(f"\n✅ {step_name} completed successfully in {end_time - start_time:.2f} seconds")
        return True
    else:
        print(f"\n❌ {step_name} failed with return code {result.returncode}")
        return False

# Make sure we're in the correct directory
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Check for required packages from requirements.txt
def check_requirements():
    """Check if all required packages are installed"""
    # Get the root directory (where requirements.txt is located)
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    req_file = os.path.join(root_dir, 'requirements.txt')
    
    print("Checking required packages...")
    
    if not os.path.exists(req_file):
        print(f"Warning: requirements.txt not found at {req_file}")
        return True
    
    # Read requirements file
    with open(req_file, 'r') as f:
        requirements = f.readlines()
    
    # Parse requirements, removing version specifiers
    required_packages = []
    for req in requirements:
        req = req.strip()
        if not req or req.startswith('#'):
            continue
            
        # Extract package name (remove version info)
        package_name = req.split('==')[0].split('>=')[0].split('>')[0].split('<')[0].strip()
        if package_name:
            required_packages.append(package_name)
    
    # Check if each package is installed
    missing_packages = []
    for package in required_packages:
        if package == "tensorflow" and importlib.util.find_spec("tensorflow") is not None:
            print(f"✅ {package} is installed")
            continue
            
        try:
            pkg_resources.get_distribution(package)
            print(f"✅ {package} is installed")
        except pkg_resources.DistributionNotFound:
            print(f"❌ {package} is not installed")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\nMissing packages: {', '.join(missing_packages)}")
        print("Installing missing packages...")
        
        if run_step("Install Required Packages", f"pip install {' '.join(missing_packages)}"):
            print("All required packages installed successfully.")
            return True
        else:
            print("\nFailed to install some packages. Please install them manually with:")
            print(f"pip install {' '.join(missing_packages)}")
            return False
    
    return True

# Ensure all required packages are installed
if not check_requirements():
    print("Please install all required packages before continuing.")
    exit(1)

# Create required directory structure
system = platform.system()
if system == 'Darwin':  # macOS
    base_video_dir = 'training/videos'
else:  # Windows or other systems
    base_video_dir = 'videos'

# Create category directories if they don't exist
video_categories = ['forward', 'left_turn', 'right_turn']
for category in video_categories:
    os.makedirs(os.path.join(base_video_dir, category), exist_ok=True)

# Check if videos exist
has_videos = False
for category in video_categories:
    video_dir = os.path.join(base_video_dir, category)
    video_files = [f for f in os.listdir(video_dir) if f.endswith(('.mp4', '.avi', '.mov'))]
    if video_files:
        has_videos = True
        break

if not has_videos:
    print(f"\n⚠️ No video files found in {base_video_dir} subdirectories.")
    print(f"Please add videos to the following directories before running the pipeline:")
    for category in video_categories:
        print(f"  - {os.path.join(base_video_dir, category)}")
    exit(1)

# Step 1: Extract frames from videos
if not run_step("Frame Extraction", "python -m training.frame_extraction"):
    exit(1)

# Step 2: Extract pose landmarks from frames
if not run_step("Pose Landmark Extraction", "python -m training.pose_landmark_extraction"):
    exit(1)

# Step 3: Train the classification model
if not run_step("Model Training", "python -m training.train_model"):
    exit(1)

print("\n🎉 Processing pipeline completed successfully!")
print("The trained LSTM model is saved in training/models/lstm_model directory")
print("You can now use the model for real-time classification using main.py") 