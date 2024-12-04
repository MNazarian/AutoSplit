# Name: AutoSplit: Two-Stage AI Architecture for Enhanced Classification of Manufacturing Processes

<div align="center">
  <img src="Images/AutoSplit.png" alt="1111" width="400" />
</div>


## Overview
AutoSplit is an innovative machine learning framework for manufacturing process classification. Using a two-stage architecture, it combines computer vision and ML techniques for accurate process classification.
<table style="border-collapse: collapse; border: none;">
<tr style="border: none;">
<td style="border: none;"><img src="Images/Architecture1.jpg" alt="Architecture diagram" width="600" /></td>
<td style="border: none;"><img src="Images/Architecture.jpg" alt="1111" width="400" /></td>
</tr>
</table>

### Features:

**Stage 1**: Hybrid Neural Network (CNN + MLP)
88.84% accuracy for standard/non-standard parts
ResNet50V2 backbone
Multi-perspective CAD analysis
Visual and geometric feature processing

**Stage 2**: Random Forest Classifier
82% accuracy in process classification
Categories: AM, milling, sheet metal
50 quantitative features
Probabilistic output

### Technical Highlights:
- Integration with OpenCascade Technology (OCCT) for feature extraction  
https://dev.opencascade.org/doc/refman/html/class_g_prop___g_props.html
- Processing of STEP file formats
- Automated feature extraction pipeline
- Support for both geometric and visual analysis
- Batch processing capabilities for large assemblies
- Real-time classification support

### Applications:
- Automated process planning in Industry 4.0
- Design for Manufacturing (DfM)
- Cost estimation and optimization
- Manufacturing process selection
- Standard part identification
- CAD model classification

### Performance Improvements:
- 15.6% improvement in standard part classification
- 18.0% enhancement in AM component recall
- 4.8% improvement in machining identification
- 24.6% increase in sheet metal component recall
- 7.4% overall accuracy improvement.


## Visuals
<div style="display: flex; justify-content: space-between;">
    <img src="Images/AM.png" alt="AM Process" width="45%">
    <img src="Images/Milling.png" alt="Milling Process" width="45%">
</div>
<div style="display: flex; justify-content: space-between;">
    <img src="Images/Sheet_Metal.png" alt="Sheet Metal Process" width="45%">
    <img src="Images/standard_component.png" alt="Standard Component" width="45%">
</div>

## Dataset
The model was trained on:
- 20,000 CAD STEP models
- Sources: GrabCAD, Fusion360 Gallery, and TraceParts
- Categorized into four classes: AM, milling, sheet metal, and standard parts

## Installation requirements
- Python 3.8+
- PyTorch 1.9+
- OpenCascade Technology (OCCT)
- scikit-learn
- numpy
- pandas
  
## Quick Start

### Prerequisites
- Python 3.8 or higher
- PyTorch 1.9+
- CUDA-enabled GPU (recommended)

### 1. Clone the repository
```bash
git clone https://github.com/MNazarian/AutoSplit.git
```

### 2. Install Dependencies
First, ensure you have all the necessary dependencies installed. You can install them using `pip`:

```bash
pip install -r requirements.txt

```
### 3. Load the Model
```bash
hier muss the modelle hochgeladen werden

```

### Example Output

- Additive Manufacturing (AM)
- Milling
- Sheet Metal
- Standard Parts

**Example:** The predicted manufacturing process is: **AM**



## Support
Tell people where they can go to for help. It can be any combination of an issue tracker, a chat room, an email address, etc.

## Roadmap

| folder     | Contents                 |
|------------|------------------------|
| `data/`    | preprocessing, feature_extraction |
| `models/`  | cnn_mlp, random_forest |
| `utils/`   | Useful scripts     |
| `tests/`   | test cases              |


## Authors and acknowledgment
- Mehdi Nazarian (Fraunhofer IAPT),
- Rafael Neves (Fraunhofer IAPT)
- Léon Klick (Autoflug GmbH)
- Robert Lau (Fraunhofer IAPT)
- Felix Weigand (Fraunhofer IAPT)

**Acknowledgement**
The authors express their sincere thanks to the Federal Ministry of Economics and Climate Protection and the Project Management Jülich for providing funding and support for the associated project.


## License
For open source projects, say how it is licensed.


## Citation

```bibtex
@article{nazarian2024autosplit,
    title={AutoSplit: A Novel Two-Stage AI Architecture for Enhanced Classification of Manufacturing Processes},
    author={Nazarian, Mehdi and Neves, Rafael and Klick, Le'on and Lau, Robert and Weigand, Felix},
    journal={arXiv preprint},
    year={2024},
    url={https://gitlab.cc-asp.fraunhofer.de/iapt/prozesskettenautomatisierung/bauteildesign/autosplit},
    note={Under Review}
}
