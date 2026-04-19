# Dataset Documentation

## Overview
This directory contains bioimpedance datasets for tomato plant stress detection. The data represents measurements under different stress conditions (water stress and iron stress).

## Dataset Structure

### water_stress/
- **File**: `Water_Stress.npz`
- **Description**: Bioimpedance measurements recorded from tomato plants under water stress conditions
- **Format**: NumPy compressed archive (.npz)
- **Content**: Time-series bioimpedance measurements at multiple frequencies

### iron_stress/
- **File**: `Iron_Stress.npz`
- **Description**: Bioimpedance measurements recorded from tomato plants under iron deficiency stress conditions
- **Format**: NumPy compressed archive (.npz)
- **Content**: Time-series bioimpedance measurements at multiple frequencies

## Data Format
Each .npz file contains:
- **Impedance measurements**: Multi-dimensional array of bioimpedance values
- **Frequency information**: Operating frequencies for impedance measurements
- **Metadata**: Sampling information, measurement conditions, plant identifiers

## Usage
Data loading and preprocessing can be done using:
- `src/data_processing_management.py` - Main data manager class
- `src/data_processing_preprocessing.py` - Feature selection and normalization
- `src/model/temporal_encoding.py` - Temporal encoding for SNN input

## Loading Data
```python
from src.data_processing_management import PlantDataManager

# Load water stress data
manager = PlantDataManager(stress_type="water")
train_loader, test_loader = manager.load_and_split()

# Or load iron stress data
manager = PlantDataManager(stress_type="iron")
train_loader, test_loader = manager.load_and_split()
```

## Notes
- All measurements are in impedance units (Ω)
- Temporal encoding converts continuous signals to spike trains for SNN processing
- Feature selection uses optimal frequency indices based on Elastic Net analysis
