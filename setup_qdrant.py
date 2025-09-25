import subprocess
import time
import requests
from config import QDRANT_HOST, QDRANT_PORT

def check_docker():
    """Kiểm tra Docker có được cài đặt không"""
    try:
        result = subprocess.run(['docker', '--version'], capture_output=True, text=True)
        if result.returncode == 0:
            print("✓ Docker is installed")
            return True
        else:
            print("✗ Docker is not installed")
            return False
    except FileNotFoundError:
        print("✗ Docker is not found")
        return False

def start_qdrant_docker():
    """Khởi động Qdrant bằng Docker"""
    if not check_docker():
        print("Please install Docker first:")
        print("- Windows/Mac: Download from https://www.docker.com/products/docker-desktop")
        print("- Linux: sudo apt-get install docker.io")
        return False
    
    print("Starting Qdrant with Docker...")
    try:
        # Dừng container cũ nếu có
        subprocess.run(['docker', 'stop', 'qdrant'], capture_output=True)
        subprocess.run(['docker', 'rm', 'qdrant'], capture_output=True)
        
        # Khởi động container mới
        cmd = [
            'docker', 'run', '-d',
            '--name', 'qdrant',
            '-p', f'{QDRANT_PORT}:6333',
            '-v', 'qdrant_data:/qdrant/storage',
            'qdrant/qdrant'
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✓ Qdrant container started successfully")
            print(f"Container ID: {result.stdout.strip()}")
            
            # Đợi Qdrant khởi động
            print("Waiting for Qdrant to be ready...")
            for i in range(30):
                try:
                    response = requests.get(f"http://{QDRANT_HOST}:{QDRANT_PORT}/health")
                    if response.status_code == 200:
                        print("✓ Qdrant is ready!")
                        return True
                except:
                    time.sleep(1)
                    print(f"Waiting... ({i+1}/30)")
            
            print("✗ Qdrant did not start in time")
            return False
        else:
            print(f"✗ Failed to start Qdrant: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"Error starting Qdrant: {e}")
        return False

def check_qdrant_status():
    """Kiểm tra trạng thái Qdrant"""
    try:
        response = requests.get(f"http://{QDRANT_HOST}:{QDRANT_PORT}/health")
        if response.status_code == 200:
            print("✓ Qdrant is running")
            
            # Lấy thông tin collections
            collections_response = requests.get(f"http://{QDRANT_HOST}:{QDRANT_PORT}/collections")
            if collections_response.status_code == 200:
                collections = collections_response.json()
                print(f"Collections: {collections}")
            
            return True
        else:
            print("✗ Qdrant is not responding")
            return False
    except Exception as e:
        print(f"✗ Cannot connect to Qdrant: {e}")
        return False

def stop_qdrant():
    """Dừng Qdrant"""
    try:
        result = subprocess.run(['docker', 'stop', 'qdrant'], capture_output=True, text=True)
        if result.returncode == 0:
            print("✓ Qdrant stopped")
        else:
            print("✗ Failed to stop Qdrant")
    except Exception as e:
        print(f"Error stopping Qdrant: {e}")

def main():
    print("=== QDRANT SETUP ===")
    print("1. Check status")
    print("2. Start Qdrant")
    print("3. Stop Qdrant")
    print("4. Restart Qdrant")
    
    choice = input("Choose option (1-4): ").strip()
    
    if choice == "1":
        check_qdrant_status()
    elif choice == "2":
        start_qdrant_docker()
    elif choice == "3":
        stop_qdrant()
    elif choice == "4":
        stop_qdrant()
        time.sleep(2)
        start_qdrant_docker()
    else:
        print("Invalid choice")

if __name__ == "__main__":
    main()