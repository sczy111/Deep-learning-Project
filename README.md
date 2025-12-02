# Deep Learning Project

This project is implemented primarily in Jupyter Notebooks.  
Each task has its own notebook file (e.g., `csu_Task1.ipynb` contains all code for Task 1), and each notebook includes preliminary results.  
For Tasks 2 and 3, the notebooks also contain larger training loops where different hyperparameters were tested.  
These sections are commented out, but all preliminary results are preserved.

---

## Environment Setup

The project was developed in **Python 3.12** using the **PyTorch nightly CUDA 12.8 build**:

```bash
pip3 install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cu128
```

You should be able to run the notebooks with standard machine learning libraries such as:

- torch
- torchvision
- pandas
- numpy
- scikit-learn
- tqdm
- Pillow
- matplotlib

If you encounter issues, you may create a virtual environment and install the packages from `requirements.txt`.

```bash
pip install -r requirements.txt
```

Then install the correct PyTorch version depending on your hardware.

However, this is **not highly recommended**, as the requirements file was generated from my global environment and some package versions may not be compatible with your setup.  
It may also include libraries that are not strictly required for this project, so always install versions that match your own environment.

---

## Project Structure

- **`csu_Task1.ipynb`, `csu_Task2.ipynb`, `csu_Task3.ipynb`, `csu_Task4.ipynb`**
  Each file contains the full implementation and results for its corresponding task.

- **Task 4 (Ensemble Learning)**
  No new model is trained. Task 4 uses an ensemble of the trained models from Task 3.
  To verify the results, run the Task 4 notebook. Preliminary results are also included inside it.

- **`csu_onsiteSubmission.ipynb`**
  This notebook is used to reproduce all Kaggle submissions and evaluation metrics.
  It loads trained models from the `/checkpoints` folder and recomputes the predictions.

---

## Checkpoints and Submissions

Trained models from each task are saved under:

```
/checkpoints/
```

For example, Task 2.1 models:

```
/checkpoints/task2/csu_task2_1_resnet18.pt
/checkpoints/task2/csu_task2_1_efficientnet.pt
```

Final Kaggle submissions (the ones achieving the best score in each task) are stored in:

```
/submission/
```

For example, Task 3.1 (ResNet18) submission:

```
/submission/Final_submission_task3/submission_resnet18_task3_1.csv
```

Any newly generated models or Kaggle submissions will also be saved under `/checkpoints` and `/submission`.

---

## Notes on Reproducibility

Due to randomness in training:

- Your results may differ from the reported ones
- Even with identical hyperparameters, long training times may be needed to match performance
- If you run without CUDA or with insufficient VRAM, some training processes may crash

All evaluation and submission files are included for verification.

---

## Usage

Each notebook contains section explanations and can be run sequentially.
To quickly verify results or regenerate submissions, use:

- `csu_onsiteSubmission.ipynb`

which reproduces both Kaggle submissions and evaluation metrics using the saved checkpoints.

---
