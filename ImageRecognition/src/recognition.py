"""Compatibility entry point for single-image recognition."""

if __package__:
    from .cli import main
else:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from ImageRecognition.src.cli import main

if __name__ == "__main__":
    import sys
    main(["predict", *sys.argv[1:]])

