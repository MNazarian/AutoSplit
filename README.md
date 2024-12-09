# AutoSplit: Two-Stage AI Architecture for Enhanced Classification of Manufacturing Processes

<div align="center">
  <img src="/docs/img/AutoSplit.png" alt="AutoSplit Logo" width="350" />
</div>


## Overview
AutoSplit is an innovative machine learning framework for manufacturing process classification. Using a two-stage architecture, it combines computer vision and ML techniques for accurate process classification.

<div align="center">
  <img src="/docs/img/Architecture2.jpg" alt="AutoSplit Logo" width="500" />
</div>

### Key Features:

**Stage 1**: Hybrid Neural Network (CNN + MLP)
- 88.84% accuracy for standard/non-standard parts
- ResNet50V2 backbone
- Multi-perspective CAD analysis
- Visual and geometric feature processing

**Stage 2**: Random Forest Classifier
- 82% accuracy in process classification
- Categories: AM, milling, sheet metal
- 50 quantitative features
- Probabilistic output

### Technical Highlights:
- Integration with OpenCascade Technology (OCCT) for feature extraction  
 [OCCT Documentation](https://dev.opencascade.org/doc/refman/html/class_g_prop___g_props.html)
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

## Dataset
The model was trained on:
- 20,000 CAD STEP models
- Sources: GrabCAD, Fusion360 Gallery, and TraceParts
- Categorized into four classes: AM, milling, sheet metal, and standard parts

### Visuals
<div style="display: flex; justify-content: space-between;">
    <img src="/docs/img/AM.png" alt="AM Process" width="45%">
    <img src="/docs/img/Milling.png" alt="Milling Process" width="45%">
</div>
<div style="display: flex; justify-content: space-between;">
    <img src="/docs/img/Sheet_Metal.png" alt="Sheet Metal Process" width="45%">
    <img src="/docs/img/standard_component.png" alt="Standard Component" width="45%">
</div>

## Installation

### Prerequisites
- Python 3.8+
- PyTorch 1.9+
- OpenCascade Technology (OCCT)
- scikit-learn
- numpy
- pandas
- CUDA-enabled GPU (recommended)

### Getting Started
```bash
# Clone repository
git clone https://github.com/MNazarian/AutoSplit.git
```
## Install dependencies
```bash
pip install -r requirements.txt
```

### Example Output

- Additive Manufacturing (AM)
- Milling
- Sheet Metal
- Standard Parts

**Example:** The predicted manufacturing process is: **AM**


## Project Structure

| folder     | Contents                 |
|------------|------------------------|
| `data/`    | preprocessing, feature_extraction |
| `models/`  | cnn_mlp, random_forest |
| `utils/`   | Useful scripts     |
| `tests/`   | test cases              |


## Support
For technical questions or support:

- Create an issue in the GitHub repository
- Contact: mehdi.nazarian@iapt.fraunhofer.de

## Authors and acknowledgment
- Mehdi Nazarian (Fraunhofer IAPT)
- Rafael Neves (Fraunhofer IAPT)
- Léon Klick (Autoflug GmbH)
- Robert Lau (Fraunhofer IAPT)
- Felix Weigand (Fraunhofer IAPT)

**Acknowledgement:**  

The authors express their sincere thanks to the Federal Ministry of Economics and Climate Protection and the Project Management Jülich for providing funding and support for the associated project.


## Citation

```bibtex
@software{autosplit_2024,
  author = {Mehdi Nazarian and Rafael Neves and Léon Klick and Robert Lau and Felix Weigand},
  title = {AutoSplit: Two-Stage AI Architecture for Enhanced Classification of Manufacturing Processes},
  year = {2024},
  publisher = {GitHub},
  url = {https://github.com/MNazarian/AutoSplit},
  version = {1.0.0}
}
