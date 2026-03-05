# CRF for OCR (CS 512 – Lab 1: Graphical Models)

End-to-end implementation of Conditional Random Fields (CRF) for OCR word recognition using graphical models.  
Covers MAP decoding (DP), forward–backward partition function, gradient computation, CRF training (LBFGS), benchmarking vs SVM baselines, stochastic optimization (SGD/Momentum), MCMC-based approximate inference, and robustness to distortions.

---

## 1) Clone the Repository
git clone https://github.com/YuktaSalvi2000/AdvancedML_LAB1_ocr-graphical-models
cd AdancedML_LAB1_ocr-graphical-models


## 2) Create an Environment
python -m venv env

Activate environment using:
env\Scripts\activate

To deactivate command is:
deactivate env

On conda:
conda create -n env
conda activate env
conda deactivate


## 3) Install dependencies:
pip install -r requirements.txt


## 4) Run the files:
This folder contains Jupyter notebooks implementing the solutions for each question.
Using VSCode:
1. Open the project folder in VSCode.
2. Select the Python interpreter from the created environment.
3. Open the desired .ipynb file.
4. Click Run All Cells.

Using Jupyter:
1. go to jupyter notebook
2. Run → Run All Cells.

code/
│
└── Q2/
    ├── Q2a.ipynb
    └── Q2b.ipynb

result/
├── gradient.txt
├── solution.txt
└── prediction.txt


## 5) results:
Q2a. The average conditional log-likelihood of the training dataset under the provided CRF parameters was computed using the forward–backward dynamic programming algorithm.
Average log likelihood: -4.140274439213303
Generated files:
result/gradient.txt

Q2b. The CRF parameters were optimized using the LBFGS optimization algorithm starting from zero initialization with regularization parameter C=1000.
Optimal objective value: 28874289.074148625
Generated files:
result/solution.txt
result/prediction.txt


## 6) Git commands:
# Sync with main
git checkout main
git pull origin main

# Create branch for your work
git checkout -b Q2   

# Add and commit changes
git add .
git commit -m "Implemented Q2"

# Push branch to GitHub
git push origin Q2 
