"""
NeatBoard Backend - FastAPI server for PyTorch Module Visualization.

This module provides the backend API for visualizing NEAT PyTorch modules.
It can be run directly or through the neatboard CLI command.
"""

import argparse
import socket
import sys
import logging
from pathlib import Path
from backend.app import create_app
import uvicorn


def find_available_port(start_port: int = 8000, max_attempts: int = 10) -> int:
    """Find the next available port starting from start_port."""
    for port in range(start_port, start_port + max_attempts):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(('127.0.0.1', port))
                return port
        except OSError:
            continue
    raise RuntimeError(f"Could not find an available port in range {start_port}-{start_port + max_attempts - 1}")


def main():
    """Run the backend server directly."""
    parser = argparse.ArgumentParser(description='NeatBoard Backend Server')
    parser.add_argument('--logdir', type=str, default=None, help='Directory containing .pkl files')
    parser.add_argument('--port', type=int, default=None, help='Server port (auto-finds if in use)')
    parser.add_argument('--host', type=str, default=None, help='Server host (default: localhost)')
    parser.add_argument('--dev', action='store_true', help='Development mode with auto-reload')
    parser.add_argument(
        '--verbose',
        type=int,
        nargs='?',
        const=1,
        default=1,
        help='Enable verbose logging (0=none, 1=basic, 2=debug). Use without argument for level 1.'
    )
    args, _ = parser.parse_known_args()

    # Always use current working directory if --logdir not specified
    base_dir = Path(args.logdir).resolve() if args.logdir else Path.cwd().resolve()
    host = args.host if args.host else "127.0.0.1"
    
    # Find available port
    requested_port = args.port or 8000
    try:
        port = find_available_port(requested_port)
        if port != requested_port:
            print(f"Port {requested_port} is in use, using {port} instead")
    except RuntimeError as e:
        print(f"Error: {e}", file=sys.stderr)
        exit(1)
    
    # Configure logging based on verbose level
    log_level = "critical"  # default # TODO: Verbose levels subject to change
    if args.verbose == 0:
        log_level = "warning"
    elif args.verbose == 1:
        log_level = "info"
    elif args.verbose >= 2:
        log_level = "debug"
    
    # Set up Python logging
    logging.basicConfig(
        level=getattr(logging, log_level.upper(), logging.WARNING),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    print(f"Starting NeatBoard Backend")
    print(f"Dashboard: http://localhost:{port}")
    print(f"Source directory: {base_dir}")
    if args.dev:
        print(f"Development mode: enabled (auto-reload)")
    if args.verbose > 0:
        print(f"Verbose logging: level {args.verbose}")

    app = create_app(base_dir)

    uvicorn.run(app, host=host, port=port, reload=args.dev, log_level=log_level)


if __name__ == "__main__":
    main()
