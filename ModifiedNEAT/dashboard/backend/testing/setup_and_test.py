#!/usr/bin/env python3
"""Setup and test script for NeatBoard."""

import subprocess
import sys
import os
from pathlib import Path
import time
import requests
from typing import Tuple


def run_command(cmd: list, description: str, cwd: str = None) -> Tuple[bool, str]:
    """Run a shell command and return success status and output."""
    print(f"\n{'='*60}")
    print(f"STEP: {description}")
    print(f"{'='*60}")
    print(f"Running: {' '.join(cmd)}\n")
    
    try:
        result = subprocess.run(
            cmd,
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=120
        )
        
        if result.stdout:
            print(result.stdout)
        
        if result.returncode != 0:
            print(f"ERROR: {result.stderr}")
            return False, result.stderr
        
        return True, result.stdout
    except subprocess.TimeoutExpired:
        print(f"TIMEOUT: Command took too long to execute")
        return False, "Timeout"
    except Exception as e:
        print(f"ERROR: {e}")
        return False, str(e)


def check_backend_ready(port: int = 8000, timeout: int = 10) -> bool:
    """Check if backend is ready to serve requests."""
    print(f"\nChecking if backend is ready on port {port}...")
    
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            response = requests.get(f"http://localhost:{port}/health", timeout=2)
            if response.status_code == 200:
                print(f"✓ Backend is ready!")
                return True
        except requests.exceptions.RequestException:
            pass
        
        time.sleep(0.5)
    
    print(f"✗ Backend did not respond within {timeout} seconds")
    return False


def test_api_endpoints(port: int = 8000) -> bool:
    """Test basic API endpoints."""
    print(f"\n{'='*60}")
    print(f"TESTING: API Endpoints")
    print(f"{'='*60}")
    
    endpoints = [
        ("GET", "/health", None),
        ("GET", "/api/info", None),
        ("GET", "/api/files", None),
    ]
    
    all_passed = True
    
    for method, path, data in endpoints:
        url = f"http://localhost:{port}{path}"
        print(f"\n{method} {url}")
        
        try:
            if method == "GET":
                response = requests.get(url, timeout=5)
            else:
                response = requests.post(url, json=data, timeout=5)
            
            print(f"  Status: {response.status_code}")
            
            if response.status_code == 200:
                print(f"  Response: {response.json()}")
                print(f"  ✓ PASS")
            else:
                print(f"  ✗ FAIL")
                all_passed = False
        
        except Exception as e:
            print(f"  ✗ ERROR: {e}")
            all_passed = False
    
    return all_passed


def main():
    """Run setup and tests."""
    dashboard_dir = Path(__file__).parent.resolve()
    project_dir = dashboard_dir.parent.parent
    
    print(f"\n{'='*60}")
    print(f"NeatBoard Setup and Test")
    print(f"{'='*60}")
    print(f"Dashboard directory: {dashboard_dir}")
    print(f"Project directory: {project_dir}")
    
    # Step 1: Build frontend
    print(f"\n{'='*60}")
    print(f"STEP 1: Build Frontend")
    print(f"{'='*60}")
    
    success, _ = run_command(
        ["bun", "run", "build"],
        "Building frontend with Vite",
        cwd=str(dashboard_dir)
    )
    
    if not success:
        print(f"\n✗ Frontend build failed. Please check the error above.")
        return False
    
    # Step 2: Create sample model
    print(f"\n{'='*60}")
    print(f"STEP 2: Create Sample Model")
    print(f"{'='*60}")
    
    sample_output_dir = dashboard_dir / "sample_models"
    sample_output_dir.mkdir(exist_ok=True)
    
    success, output = run_command(
        ["python3", "backend/create_sample.py", str(sample_output_dir)],
        "Creating sample PyTorch model",
        cwd=str(dashboard_dir)
    )
    
    if not success:
        print(f"\n✗ Failed to create sample model")
        return False
    
    sample_model = sample_output_dir / "sample_model.pkl"
    if not sample_model.exists():
        print(f"\n✗ Sample model file was not created at {sample_model}")
        return False
    
    print(f"✓ Sample model created: {sample_model}")
    
    # Step 3: Reinstall package with CLI entry point
    print(f"\n{'='*60}")
    print(f"STEP 3: Install Package with CLI Entry Point")
    print(f"{'='*60}")
    
    success, _ = run_command(
        ["pip", "install", "-e", "."],
        "Installing package in editable mode",
        cwd=str(project_dir)
    )
    
    if not success:
        print(f"\n✗ Package installation failed")
        return False
    
    # Step 4: Verify CLI command
    print(f"\n{'='*60}")
    print(f"STEP 4: Verify CLI Command")
    print(f"{'='*60}")
    
    success, output = run_command(
        ["neatboard", "--help"],
        "Checking neatboard CLI command",
    )
    
    if not success:
        print(f"\n✗ CLI command not available")
        return False
    
    # Final summary
    print(f"\n{'='*60}")
    print(f"SETUP COMPLETE")
    print(f"{'='*60}")
    print(f"""
✓ All setup steps completed successfully!

To start the dashboard:

  Terminal 1 (Backend):
    neatboard --logdir {sample_output_dir}
  
  Terminal 2 (Frontend):
    cd {dashboard_dir}
    bun run dev

Then open your browser to:
  http://localhost:5173

The sample model should be available in the file list.
API documentation available at:
  http://localhost:8000/docs
""")
    
    return True


if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print(f"\n\nSetup cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nUnexpected error: {e}")
        sys.exit(1)
