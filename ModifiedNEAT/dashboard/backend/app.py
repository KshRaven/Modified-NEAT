"""FastAPI application factory."""

from pathlib import Path
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import mimetypes

from .handlers import ModuleHandler, TensorHandler
from .models import SliceRequest


def create_app(base_dir: Path = None) -> FastAPI:
    """Create and configure the FastAPI application.
    
    Args:
        base_dir: Directory containing .pkl files. Defaults to current working directory.
    
    Returns:
        Configured FastAPI application instance.
    """
    if base_dir is None:
        base_dir = Path.cwd().resolve()
    else:
        base_dir = Path(base_dir).resolve()

    app = FastAPI(
        title="NeatBoard API",
        description="PyTorch Module Visualizer API for NEAT modules",
        version="1.0.0"
    )

    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Initialize handlers
    module_handler = ModuleHandler(base_dir)
    tensor_handler = TensorHandler(base_dir)

    # Routes
    @app.get("/api/info")
    async def get_info():
        """Get application info including source directory."""
        from starlette.requests import Request
        # Note: Request will be injected by FastAPI
        return {
            "source_dir": str(base_dir),
            "has_files": any(base_dir.glob("*.pkl")), # TODO: Fix: Returns false even when .pkl files are available
        }
    
    @app.get("/api/config")
    async def get_config(request: Request):
        """Get frontend configuration including API endpoint."""
        # Get the base URL from the request
        base_url = f"{request.url.scheme}://{request.url.netloc}"
        return {
            "api_base": base_url, # TODO: Ensure when run on server it does not expose server ip unless...
            "source_dir": str(base_dir), # TODO: Or ensure only authenticated users can use certain API like this
        }

    @app.get("/api/files")
    async def list_pkl_files():
        """List all pickle files in the source directory."""
        return module_handler.list_pkl_files()
    
    @app.post("/api/refresh")
    async def refresh_files():
        """Refresh file list (cache busting endpoint)."""
        return module_handler.list_pkl_files()

    @app.post("/api/load")
    async def load_module(filename: str):
        """Load a pickle file and return its module graph."""
        return module_handler.load_module(filename)

    @app.post("/api/tensor")
    async def get_tensor(request: SliceRequest):
        """Get a specific tensor from a module."""
        return tensor_handler.get_tensor(request)

    @app.get("/api/tensor/list")
    async def list_tensors(filename: str, module_path: str, model_id: str = None):
        """List all tensors in a module.
        
        Args:
            filename: Pickle file name
            module_path: Dot-separated path to module
            model_id: Optional model/genus ID for multi-model files
        """
        return tensor_handler.list_tensors(filename, module_path, model_id)

    @app.get("/health")
    async def health():
        """Health check endpoint."""
        return {"status": "healthy", "source_dir": str(base_dir)}

    # Mount static files (assets) if dist directory exists
    dist_dir = Path(__file__).parent.parent / "dist"
    if dist_dir.exists():
        assets_dir = dist_dir / "assets"
        if assets_dir.exists():
            app.mount("/assets", StaticFiles(directory=str(assets_dir)), name="assets")

    # Helper function to serve SPA files
    def serve_spa_file(full_path: str):
        """Serve frontend static files with fallback to index.html for SPA routing."""
        dist_dir = Path(__file__).parent.parent / "dist"
        
        if not dist_dir.exists():
            # Ensure build with package manager like bun or npm
            return {
                "error": "Frontend not built",
                "message": "Run 'cd <path_to>/ModifiedNEAT/dashboard && npm run build' to build the frontend",
                "source_dir": str(base_dir)
            }
        
        # Try to serve the requested file
        file_path = dist_dir / full_path if full_path else dist_dir / "index.html"
        
        # Security: prevent directory traversal
        try:
            file_path.resolve().relative_to(dist_dir.resolve())
        except ValueError:
            file_path = dist_dir / "index.html"
        
        # If file exists and is a file, serve it
        if file_path.is_file():
            mime_type, _ = mimetypes.guess_type(str(file_path))
            return FileResponse(file_path, media_type=mime_type)
        
        # Otherwise serve index.html for client-side routing
        index_html = dist_dir / "index.html"
        if index_html.exists():
            return FileResponse(index_html, media_type="text/html")
        
        return {"error": "Frontend files not found"}
    
    # Root path - serve index.html
    @app.get("/")
    async def root():
        """Serve frontend root."""
        return serve_spa_file("")
    
    # Catch-all route for SPA (serve index.html for all non-API routes)
    @app.get("/{full_path:path}")
    async def serve_spa(full_path: str):
        """Serve frontend static files with fallback to index.html for SPA routing."""
        # Don't serve non-existent API routes
        if full_path.startswith("api/"):
            return {"error": "API endpoint not found"}
        
        return serve_spa_file(full_path)

    return app
