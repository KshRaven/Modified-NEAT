"""CLI module for neatboard command."""

import sys
import argparse
import uvicorn
import socket
import os
import logging
from pathlib import Path

# Global variable to store base_dir for use in get_app() during reload
_NEATBOARD_BASE_DIR = None
_NEATBOARD_HOST = None
_NEATBOARD_PORT = None
_NEATBOARD_VERBOSE = 0


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
    """Main entry point for the neatboard CLI."""
    global _NEATBOARD_BASE_DIR, _NEATBOARD_HOST, _NEATBOARD_PORT
    
    parser = argparse.ArgumentParser(
        prog="neatboard",
        description="NeatBoard - PyTorch NEAT Module Visualizer and Inspector",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  neatboard  # Uses current directory, opens on http://localhost:8000
  neatboard --logdir /path/to/models
  neatboard --logdir . --port 3000  # Custom port
  neatboard --dev  # Development mode with auto-reload
  neatboard --host 0.0.0.0  # Listen on all interfaces
        """
    )

    parser.add_argument(
        "--logdir",
        type=str,
        default=None,
        help="Directory containing .pkl files (default: current directory)"
    )

    parser.add_argument(
        "--port",
        type=int,
        default=None,
        help="Server port (default: 8000, auto-finds next available if in use)"
    )

    parser.add_argument(
        "--host",
        type=str,
        default=None,
        help="Server host address (default: localhost). Use 0.0.0.0 to listen on all interfaces."
    )

    parser.add_argument(
        "--dev",
        action="store_true",
        help="Enable development mode with auto-reload"
    )

    parser.add_argument(
        "--verbose",
        type=int,
        nargs="?",
        const=1,
        default=1,
        help="Enable verbose logging (0=none, 1=basic, 2=debug). Use without argument for level 1."
    )

    args = parser.parse_args()

    # Set base directory - use current working directory if not specified
    if args.logdir:
        base_dir = Path(args.logdir).resolve()
        if not base_dir.exists():
            print(f"Error: Directory '{args.logdir}' does not exist", file=sys.stderr)
            sys.exit(1)
    else:
        # Always use current working directory when --logdir not specified
        base_dir = Path.cwd().resolve()

    # Set host - default to localhost if not specified
    host = args.host if args.host else "127.0.0.1"
    
    # Find available port (auto-detect if default is in use)
    requested_port = args.port or 8000
    try:
        port = find_available_port(requested_port)
        if port != requested_port:
            print(f"Port {requested_port} is in use, using {port} instead")
    except RuntimeError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    
    # Store globally for use in get_app() during reload
    _NEATBOARD_BASE_DIR = str(base_dir)
    _NEATBOARD_HOST = host
    _NEATBOARD_PORT = port
    _NEATBOARD_VERBOSE = args.verbose
    
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
    
    print(f"Starting NeatBoard...")
    print(f"Source directory: {base_dir}")
    print(f"Dashboard: http://localhost:{port}")
    if args.host and args.host != "127.0.0.1":
        print(f"Also available at: http://{args.host}:{port}")
    if args.dev:
        print(f"Development mode: enabled (auto-reload)")
    if args.verbose > 0:
        print(f"Verbose logging: level {args.verbose}")
    print(f"Press Ctrl+C to stop\n")

    # Import here to avoid issues when module is imported but not run
    from .app import create_app

    if args.dev:
        # Set environment variables for get_app() to use during reload
        os.environ['NEATBOARD_BASE_DIR'] = str(base_dir)
        os.environ['NEATBOARD_HOST'] = host
        os.environ['NEATBOARD_PORT'] = str(port)
        os.environ['NEATBOARD_VERBOSE'] = str(args.verbose)
        
        # Use app factory approach for reload to work
        uvicorn.run(
            "ModifiedNEAT.dashboard.backend.cli:get_app",
            host=host,
            port=port,
            reload=True,
            log_level=log_level,
        )
    else:
        app = create_app(base_dir)
        uvicorn.run(
            app,
            host=host,
            port=port,
            log_level=log_level
        )


def get_app():
    """Factory function for uvicorn reload mode."""
    from .app import create_app
    
    # Try to get base_dir from environment variable (set during reload)
    base_dir_str = os.environ.get('NEATBOARD_BASE_DIR')
    if base_dir_str:
        base_dir = Path(base_dir_str).resolve()
    else:
        # Fallback to current working directory
        base_dir = Path.cwd().resolve()
    
    return create_app(base_dir)


if __name__ == "__main__":
    main()
