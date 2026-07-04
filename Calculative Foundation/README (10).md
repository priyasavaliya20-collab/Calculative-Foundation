# 🧮 Calculative Foundation

This project presents a **Linear Algebra + Probability Theory & Practical Analysis** on a real-world **student performance dataset** containing 20 records. The objective is to apply **vector/matrix operations, decompositions, dimensionality reduction, and probability distribution fitting** to analyze student performance patterns and derive meaningful academic insights.

The project combines mathematical theory with practical implementation in Python (Jupyter Notebook), covering the complete analytical workflow — from data loading and vector representation to eigen-decomposition, PCA/LDA, and probability distribution analysis.

## 🎯 Objective

To convert raw student marks into a mathematical foundation — vectors, matrices, and distributions — that reveals *how* students differ, *which* subjects drive that difference, and *what* statistical pattern their overall performance follows.

## 🗂️ Project Files

| File | Description |
|---|---|
| 📓 `Calculative_Foundation.ipynb` | Complete linear algebra & statistics analysis notebook (Part B) |
| 📊 `student_performance_dataset.xlsx` | Student performance dataset — 20 records, 5 subjects |
| 📄 `Theory.pdf` | Theoretical foundations, formulas & visual explanations (Part A) |
| 📘 `README.md` | Project documentation (this file) |

## 🛠️ Tools & Libraries

**Python** · **Jupyter** · **NumPy** · **Pandas** · **SciPy** · **Matplotlib** · **Scikit-learn**

## 📕 Part A — Theoretical Foundation (10 Topics)

Q1. What are Vectors and Matrices? · Basic Matrix Operations
Q2. What are Norms (L1, L2, L∞)?
Q3. Dot Product and Cross Product
Q4. Vector Projections
Q5. Eigenvalues and Eigenvectors
Q6. LU Decomposition
Q7. Singular Value Decomposition (SVD)
Q8. Principal Component Analysis (PCA)
Q9. Linear Discriminant Analysis (LDA)
Q10. Quick Summary — all formulas & data-science use cases in one table

📄 *Full theory, formulas, and visual diagrams available in* `Theory.pdf`.

## 📗 Part B — Data Analysis & Testing Tasks (12 Questions, Parts A–F)

**Part A — Vector & Matrix Fundamentals**
Q1. Represent student scores as vectors · Q2. Norms, dot product, angle, cross product · Q3. Vector projection

**Part B — Matrix Operations**
Q4. Matrix form, addition, multiplication, transpose, inverse, determinant

**Part C — Linear Transformations & Geometry**
Q5. Line, plane & hyperplane · Q6. Dimensionality growth (2D → 3D → 5D)

**Part D — Eigenvalues & Decomposition**
Q7. Eigenvalues/eigenvectors of covariance matrix · Q8. LU Decomposition · Q9. Singular Value Decomposition (SVD)

**Part E — Dimensionality Reduction**
Q10. Principal Component Analysis (PCA) · Q11. Linear Discriminant Analysis (LDA)

**Part F — Probability Distribution Analysis**
Q12. Fit Normal, Uniform, Log-Normal, Exponential & Gamma distributions to Total Score

### 📂 Dataset Overview

```python
import pandas as pd, numpy as np
df = pd.read_excel("student_performance_dataset.xlsx")
subjects = ["Math", "Physics", "Chemistry", "English", "Computer"]
matrix = df[subjects].to_numpy()      # 20 × 5 students-by-subjects matrix
```

### 1️⃣ Vectors, Norms, Dot & Cross Product

**Objective:** Represent each student as a vector and measure magnitude & similarity between students.

```python
v1, v2 = matrix[0], matrix[1]
norm2 = np.linalg.norm(matrix, axis=1)          # Norm-2 per student
dot = np.dot(v1, v2)                            # Dot product
angle = np.degrees(np.arccos(dot / (np.linalg.norm(v1) * np.linalg.norm(v2))))
cross = np.cross(v1[:3], v2[:3])                # Cross product (3D subjects)
```

✅ **Conclusion:** Norm-2 ranks students by overall performance magnitude; angle shows how similar their subject-wise *pattern* is, independent of level.

📊 **Insight:** Norm-2 ranges from **139.74 (weakest)** to **207.59 (strongest)**. Students 1 & 2 have a tiny **4.97° angle** — near-identical strength patterns despite different score levels.

### 2️⃣ Vector Projection

**Objective:** Find how much one student's performance vector projects onto another's direction.

```python
proj = (np.dot(v1, v2) / np.dot(v2, v2)) * v2
```

✅ **Conclusion:** Projection is the geometric base of regression — it shows how much of one student's profile is "explained" by another's.

### 3️⃣ Matrix Operations (Addition, Multiplication, Transpose, Determinant)

**Objective:** Perform core matrix algebra on the students × subjects matrix.

```python
addition = matrix + matrix
similarity = matrix @ matrix.T                  # Gram matrix (pairwise similarity)
transpose = matrix.T
det = np.linalg.det(matrix[:5, :5])
```

✅ **Conclusion:** `matrix @ matrix.T` packs every pairwise student similarity into one multiplication; a **non-zero determinant (−56537)** confirms the sampled rows are linearly independent (full rank).

### 4️⃣ Line, Plane, Hyperplane & Dimensionality

**Objective:** Show how the "space" a student lives in grows with each added subject.

```python
line = df[["Math"]]                              # 1D
plane = df[["Math", "Physics"]]                  # 2D
hyperplane = df[subjects]                        # 5D
```

✅ **Conclusion:** Each subject added is one more dimension — from a line (1D) to a 5D hyperplane — setting up the need for dimensionality reduction (PCA).

### 5️⃣ Eigenvalues, LU & SVD

**Objective:** Decompose the covariance/score matrix to reveal dominant patterns.

```python
cov = np.cov(matrix, rowvar=False)
eigvals, eigvecs = np.linalg.eig(cov)
from scipy.linalg import lu
P, L, U = lu(matrix[:5, :5])
U_svd, S, VT = np.linalg.svd(matrix)
```

✅ **Conclusion:** A single dominant eigenvalue/singular value drives almost all student-to-student variation — i.e., one "general ability" factor.

📊 **Insight:** Largest eigenvalue explains **≈96.9%** of covariance variance; the top singular value (787.4) captures **≈99.9%** of the matrix's total energy.

### 6️⃣ PCA & LDA

**Objective:** Reduce 5 subjects to 2 dimensions and relate them to performance categories.

```python
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
pca_data = pca.fit_transform(df[subjects])
```

✅ **Conclusion:** PC1 alone re-derives "total marks" from pure geometry — strong students score high positive PC1, weak students score large negative PC1, confirming the single dominant-ability finding from the eigen-analysis.

### 7️⃣ Probability Distribution Fitting

**Objective:** Fit Normal, Uniform, Log-Normal, Exponential & Gamma distributions to student **Total Score** and identify the best fit.

```python
from scipy import stats
df["Total_Score"] = df[subjects].sum(axis=1)
data = df["Total_Score"].values.astype(float)

for name, dist in {"Normal": stats.norm, "Uniform": stats.uniform,
                    "Log-Normal": stats.lognorm, "Exponential": stats.expon,
                    "Gamma": stats.gamma}.items():
    params = dist.fit(data)
    ks_stat, ks_p = stats.kstest(data, dist.cdf, args=params)
    aic = 2 * len(params) - 2 * np.sum(dist.logpdf(data, *params))
```

✅ **Conclusion:** **Uniform is the best-fit distribution** for Total Score — confirmed statistically (not just visually).

📊 **Distribution (AIC & KS test, n=20):**

| Distribution | AIC (lower=better) | KS p-value |
|---|---|---|
| **Uniform** | **204.96** | **0.898 ✅** |
| Normal | 215.73 | 0.917 |
| Log-Normal | 217.73 | 0.917 |
| Gamma | 217.74 | 0.903 |
| Exponential | 218.60 | 0.392 ❌ |

## 📊 Key Findings Summary

- Successfully analyzed 20 students' performance using vector, matrix, and probability techniques.
- Represented every student as a 5D vector and measured similarity via norms, dot product, and angle.
- Verified full-rank, invertible structure in the score matrix via a non-zero determinant.
- Decomposed the covariance matrix and matrix itself via eigen-decomposition, LU, and SVD.
- Found that **one dominant "general ability" factor** explains ~97–99.9% of all variation between students.
- Reduced 5 subjects to 2 principal components with almost no information loss.
- Fit five probability distributions to Total Score and identified **Uniform** as the statistically best fit.

## 🎯 Final Conclusion

Linear algebra and probability techniques (vectors, norms, dot/cross product, projection, matrix operations, eigen-decomposition, LU, SVD, PCA, LDA, distribution fitting) were applied to a 20-student performance dataset to convert raw marks into evidence-based academic insights using Python.

Results showed student performance is dominated by a **single "overall ability" axis** — the largest eigenvalue explains ~96.9% of variance and the top singular value ~99.9% of total energy — meaning 5 subjects can be compressed to 1–2 dimensions with barely any information loss (confirmed by PCA).

For the score distribution, **Uniform fit best** (AIC = 204.96, KS p = 0.898), beating Normal, Log-Normal, Gamma, and Exponential. Total Scores (312–464) are spread evenly rather than clustered around a mean (excess kurtosis ≈ −1.2, matching Uniform's theoretical value) — likely a small-sample effect that would move toward Normal on a larger cohort (e.g. the 50-student version of this dataset).

## 🚀 How to Run

```bash
# 1. Install dependencies
pip install pandas numpy scipy matplotlib scikit-learn openpyxl

# 2. Launch Jupyter
jupyter notebook

# 3. Open and run
Calculative_Foundation.ipynb
```

## ✅ Project Checklist

- [x] Vector Representation & Norms (L1, L2)
- [x] Dot Product, Angle & Cross Product
- [x] Vector Projection
- [x] Matrix Operations (Addition, Multiplication, Transpose, Determinant)
- [x] Line, Plane & Hyperplane Geometry
- [x] Eigenvalues & Eigenvectors
- [x] LU Decomposition
- [x] Singular Value Decomposition (SVD)
- [x] Principal Component Analysis (PCA)
- [x] Linear Discriminant Analysis (LDA)
- [x] Probability Distribution Fitting (Normal, Uniform, Log-Normal, Exponential, Gamma)
- [x] Dataset Included (20 records)
- [x] Jupyter Notebook Included
- [x] Theory PDF Included

## 👩‍💻 Author

**Priya Savaliya**
📍 Ahmedabad, Gujarat, India

*"Data-Driven Decisions · Statistical Thinking · Evidence-Based Conclusions"*

⭐ If you found this project helpful, give it a star and feel free to fork!
