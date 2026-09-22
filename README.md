# BGRSOD

**Focusing on the Boundary: Bidirectional Granular-Ball Regeneration Subspaces with Fuzzy Deviation Permeation for Outlier Detection**

## Running the Code

Run `main_BGRSOD_parallel_20seeds.py` from the project root directory:

```bash
python main_BGRSOD_parallel_20seeds.py
```

The script runs BGRSOD on the `.mat` datasets in the `mydata6` directory. Each dataset should contain a variable named `trandata`, with features in the first columns and the ground-truth label in the last column.

## Parameter Settings

The following parameter grid is used:

| Parameter | Search range          | Step                 |
| --------- | --------------------- | -------------------- |
| `omega`   | {2, 4, 8, 16, 32, 64} | Doubles at each step |
| `alpha`   | 0.0–1.0               | 0.1                  |

Each parameter combination is evaluated **20 times**, using random seeds **1–20**. The script reports the mean AUC and standard deviation across the 20 runs.
