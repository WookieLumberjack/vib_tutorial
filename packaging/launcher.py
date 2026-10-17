"""PyInstaller entry point: a plain script, since vib_tutorial/__main__.py uses a relative import."""

from vib_tutorial import main

if __name__ == "__main__":  # main() first hands a worker process (the rotor page's analysis) its job
    main()
