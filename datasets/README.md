# Downloaded Datasets

This directory contains datasets for the research project. Data files are NOT committed to git due to size (see .gitignore).

## Dataset 1: Geometry of Truth (Cities)
- **Source**: [saprmarks/geometry-of-truth](https://github.com/saprmarks/geometry-of-truth)
- **Format**: CSV
- **Task**: Truthfulness detection (geography)
- **Location**: datasets/cities.csv

## Dataset 2: Geometry of Truth (Common Claim)
- **Source**: [saprmarks/geometry-of-truth](https://github.com/saprmarks/geometry-of-truth)
- **Format**: CSV
- **Task**: Truthfulness detection (general knowledge)
- **Location**: datasets/common_claim.csv

## Dataset 3: WiC (Word in Context) Sample
- **Source**: [HuggingFace (super_glue/wic)](https://huggingface.co/datasets/super_glue)
- **Format**: HuggingFace Dataset (saved to disk)
- **Task**: Word sense disambiguation / Polysemy
- **Location**: datasets/wic_sample/
- **Download Instructions**:
  ```python
  from datasets import load_dataset
  ds = load_dataset('super_glue', 'wic', split='train[:100]')
  ds.save_to_disk('datasets/wic_sample')
  ```

## Dataset 4: SP-EN Trans
- **Source**: [saprmarks/geometry-of-truth](https://github.com/saprmarks/geometry-of-truth)
- **Format**: CSV
- **Task**: Translation alignment
- **Location**: datasets/sp_en_trans.csv

## Sample Data (WiC)
```json
{
  "word": "beat",
  "sentence1": "The beat of the drums.",
  "sentence2": "The heart beat.",
  "label": 1
}
```
