## Dataset Tools

This folder contains scripts for preparing datasets for labeling and model training.
Use these scripts to prepare datasets before using them in the labeling app or machine learning workflows.

### `feature_extraction.py`
- **Purpose**: Generates `info.json` files with metadata for STEP models.
- **Usage**:
  - Update the desired **INPUT** and **OUTPUT** folder paths inside the script.
  - Run the script to process all STEP files in the specified folder.

  ```bash
  python feature_extraction.py

### `part_screenshots.py`
- **Purpose**: Captures screenshots of STEP models from various perspectives for visualization and training.
- **Usage**:
  - Update the desired **INPUT** and **OUTPUT** folder paths inside the script.
  - Run the script to save screenshots as PNGs.

  ```bash
  python part_screenshots.py
