#!/usr/bin/env python3
"""
IDE Integration Setup Guide

This script provides instructions and verification for setting up
different IDEs with the TensorFlow GPU Docker container:
- PyCharm Professional
- Visual Studio Code
- Other IDEs (general instructions)

It also tests GPU availability and remote debugging.
"""

import os
import sys
import tensorflow as tf
import socket
import platform
import time

def print_separator(message):
    """Print a message with separators for better readability."""
    print("\n" + "=" * 70)
    print(f" {message}")
    print("=" * 70)

def check_environment():
    """Check TensorFlow and system environment."""
    print_separator("Environment Check")
    
    print(f"TensorFlow version: {tf.__version__}")
    print(f"Python version: {sys.version.split()[0]}")
    print(f"Hostname: {socket.gethostname()}")
    print(f"Platform: {platform.platform()}")
    
    # Check GPU devices
    gpus = tf.config.list_physical_devices('GPU')
    print(f"GPUs available: {len(gpus)}")
    for i, gpu in enumerate(gpus):
        print(f"  GPU {i}: {gpu}")
    
    # Check if running in Docker
    in_docker = os.path.exists('/.dockerenv')
    print(f"Running in Docker: {in_docker}")
    
    # Check if SSH is running (for PyCharm)
    ssh_running = False
    try:
        with open('/var/run/sshd.pid', 'r') as f:
            ssh_running = True
    except:
        pass
    
    if not ssh_running:
        try:
            import subprocess
            result = subprocess.run(['service', 'ssh', 'status'], 
                                   capture_output=True, text=True)
            ssh_running = 'Active: active' in result.stdout
        except:
            pass
    
    print(f"SSH server running: {ssh_running}")
    if not ssh_running:
        print("  To start SSH server: sudo service ssh start")
    
    return len(gpus) > 0, in_docker

def test_debugpy():
    """Test if debugpy (for VS Code and PyCharm) is available."""
    try:
        import debugpy
        print(f"debugpy version: {debugpy.__version__}")
        print("✅ Remote debugging with VS Code and PyCharm is available")
        return True
    except ImportError:
        print("❌ debugpy not found. Remote debugging may not work.")
        print("   Install it with: pip install debugpy")
        return False
    except Exception as e:
        print(f"❌ Error importing debugpy: {e}")
        return False

def print_vscode_setup_instructions():
    """Print instructions for setting up VS Code with this container."""
    print_separator("Visual Studio Code Setup Instructions")
    
    print("""
To set up VS Code with this TensorFlow GPU Docker container:

OPTION 1: VS CODE REMOTE - CONTAINERS (RECOMMENDED)
====================================================
1. Install the 'Remote - Containers' extension in VS Code
2. Open the command palette (Ctrl+Shift+P) and select 
   'Remote-Containers: Open Folder in Container...'
3. Select the directory containing your project and this Docker setup
4. VS Code will build and start the container automatically, then connect to it
5. The .devcontainer/devcontainer.json file provides configuration

OPTION 2: VS CODE REMOTE - SSH
====================================================
1. Start the container with SSH enabled:
   docker run --gpus all -e START_SSH=true -p 2222:22 -v $(pwd):/app/projects -it tensorflow-gpu-custom

2. In VS Code, install the 'Remote - SSH' extension
3. Connect to the container:
   - Open the command palette (Ctrl+Shift+P)
   - Select 'Remote-SSH: Connect to Host...'
   - Add a new SSH host: ssh root@localhost -p 2222 (password: tfgpu)
   - Select 'Linux' as the platform
4. Once connected, open the /app/projects folder

DEBUGGING SETUP
====================================================
- VS Code should automatically detect the Python interpreter
- Use the .vscode/launch.json configurations for debugging
- Set breakpoints as usual in your code
""")

def print_pycharm_setup_instructions():
    """Print instructions for setting up PyCharm with this container."""
    print_separator("PyCharm Setup Instructions")
    
    print("""
To set up PyCharm Professional with this TensorFlow GPU Docker container:

1. PREPARE THE CONTAINER:
   - Start the container with SSH enabled:
     docker run --gpus all -e START_SSH=true -p 2222:22 -v $(pwd):/app/projects -it tensorflow-gpu-custom
   
   - This maps your current directory to /app/projects in the container
   - SSH server runs on port 22 in the container, mapped to 2222 on your host

2. CONFIGURE PYCHARM:
   a) Open PyCharm Professional
   b) Go to Settings/Preferences → Project → Python Interpreter
   c) Click the gear icon → Add → SSH Interpreter
   d) Configure the SSH connection:
      - Host: localhost (or your Docker host)
      - Port: 2222
      - Username: root
      - Password: tfgpu
   e) Set the Python interpreter path: /usr/bin/python3
   f) Set path mappings:
      - Local path: [Your project directory]
      - Remote path: /app/projects

3. CONFIGURE RUN CONFIGURATIONS:
   - Create a new Run Configuration
   - Select the Docker Python interpreter
   - Make sure to set the working directory to /app/projects

4. FOR DEBUGGING:
   - Set breakpoints as usual
   - Run in debug mode
   - PyCharm will automatically use debugpy for remote debugging
""")

def print_general_ide_instructions():
    """Print general instructions for other IDEs."""
    print_separator("General IDE Setup Instructions")
    
    print("""
For other IDEs or text editors, you can use this container as follows:

1. CONTAINER SETUP:
   - Start the container with your code directory mounted:
     docker run --gpus all -v $(pwd):/app/projects -it tensorflow-gpu-custom

2. JUPYTER NOTEBOOK INTEGRATION:
   - Start Jupyter from inside the container:
     jupyter notebook --ip 0.0.0.0 --port 8888 --no-browser --allow-root
   - Access in your browser using the link provided in the terminal
   - Your notebooks will be saved to the mounted directory

3. X11 FORWARDING (FOR GUI APPLICATIONS):
   - On Linux/macOS, you can enable X11 forwarding:
     docker run --gpus all -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix -v $(pwd):/app/projects -it tensorflow-gpu-custom

4. DIRECT TERMINAL ACCESS:
   - Connect to a running container:
     docker exec -it tensorflow-gpu-custom bash
     
5. REMOTE DEVELOPMENT:
   - Many IDEs (like Sublime Text, Atom, etc.) support SFTP/SCP
   - Configure your IDE to upload files to the container over SSH
   - Host: localhost (or your Docker host)
   - Port: 2222
   - Username: root
   - Password: tfgpu
   - Remote path: /app/projects
""")

def run_demo():
    """Run a small TensorFlow demo to verify integration."""
    print_separator("TensorFlow Demo")
    
    print("Creating a simple model to test debugging...")
    
    # Create and train a very simple model
    import numpy as np
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense
    
    # Point for setting a breakpoint in PyCharm/VS Code
    print("This is a good place to set a breakpoint in your IDE")
    time.sleep(1)
    
    # Generate some data
    X = np.random.random((100, 10))
    y = np.random.random((100, 1))
    
    # Create model
    model = Sequential([
        Dense(5, activation='relu', input_shape=(10,)),
        Dense(1, activation='linear')
    ])
    
    # Compile model
    model.compile(optimizer='adam', loss='mse')
    
    # Train for just 1 epoch
    model.fit(X, y, epochs=1, verbose=1)
    
    print("Demo complete. If debugging, you should have been")
    print("able to set breakpoints, inspect variables, etc.")

def main():
    """Main function to run the script."""
    print_separator("IDE Integration for TensorFlow GPU Docker")
    print("This script helps you set up your preferred IDE with this container.")
    
    has_gpu, in_docker = check_environment()
    
    if not in_docker:
        print("WARNING: This script appears to be running outside of Docker.")
        print("For IDE integration, this script must run inside the Docker container.")
    
    debugpy_available = test_debugpy()
    
    # Print all setup instructions
    print("\nPlease select your IDE/editor for specific instructions:")
    print("1. Visual Studio Code")
    print("2. PyCharm Professional")
    print("3. Other IDEs/editors")
    print("4. All instructions")
    print("5. Run TensorFlow demo")
    
    choice = input("\nEnter your choice (1-5): ")
    
    if choice == '1':
        print_vscode_setup_instructions()
    elif choice == '2':
        print_pycharm_setup_instructions()
    elif choice == '3':
        print_general_ide_instructions()
    elif choice == '4':
        print_vscode_setup_instructions()
        print_pycharm_setup_instructions()
        print_general_ide_instructions()
    elif choice == '5':
        print("Running TensorFlow demo...")
        run_demo()
    else:
        print("Invalid choice. Please run the script again.")

if __name__ == "__main__":
    main()