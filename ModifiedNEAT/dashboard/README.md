# Dashboard guide

NeatBoard is the built-in dashboard for inspecting serialized NEAT modules and related artifacts. It is implemented as a Python backend plus a React frontend and is exposed through the neatboard CLI entry point.

## What the dashboard does

The dashboard is designed to help you:

- load pickled module or population artifacts
- inspect the module graph structure
- examine tensor values and module metadata
- navigate multiple saved artifacts from a chosen source directory

## Running the dashboard

```bash
neatboard --logdir /path/to/artifacts
```

Useful options include:

- --port
- --host
- --dev
- --verbose

## Typical workflow

1. Save a module or population artifact to disk.
2. Start the dashboard with the desired source directory.
3. Open the reported local URL in a browser.
4. Select a file and inspect the content.

## Notes

The dashboard expects the selected directory to contain readable artifact files. For large experiments, using a narrower source directory helps keep the file list focused and easier to browse.
