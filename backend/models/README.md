# Models Directory

Place your trained model files here after running the Colab notebook.

## Required Files

After running `notebooks/train.ipynb`, download and place these files here:

- `classifier.pkl` - The trained classifier (~1MB)
- `label_encoder.pkl` - Label mapping (~1KB)  
- `config.json` - Model metadata

## Without These Files

The application will still work using the rule-based fallback system (labeling functions).
However, the neural model provides better generalization and accuracy.

## How to Get These Files

1. Open `notebooks/train.ipynb` in Google Colab
2. Run all cells
3. Download the 3 files when prompted
4. Place them in this directory
