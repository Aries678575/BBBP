# Prediction blood-brain barrier permeability with RDKit + Random forest

- python: 3.13.15
- RDKit: 2023.09+
- scikit-learn: 1.3+

# 1. Introduction
A minimal pipeline that predicts whether a small chemical molecule can penetrate the blood-brain barrier (BBB) using Morgan (ECFP4) fingerprints[^1] and a Random Forest Classifier[^2].

BBB excludes most circulating molecules, leaving challenges on drug delivery to the neuronal system[^3]. Predicting the permeability from molecular structures alone by machine learning narrows the design space and saves time and sources during experimental screening.

---

# 2.Features
## Dataset
BBBP(Blood-Brain Barrier Penetration) from [DeepChem](https://deepchem.io/). [^4]

| Field  | Description                          |
| ------ | ------------------------------------ |
| num    | molecule index                       |
| name   | compound name                        |
| p_np   | 1=penetrate BBB, 0=does not          |
| smiles | molecular structure as SMILES string |
There are 2050 molecules in total, of which 11 invalid SMILES were removed. Among the 2039 molecules left, 1567 (76.4%) are positive, 483 (23.6%) are negative, which present an imbalanced class distribution.

---
## Method
**Feature engineering**
Encode each SMILES string into a 2048-demensional binary vector where each bit represents the presence/absence of a certain substructure using RDKit:
```python
from rdkit import Chem
from rdkit.Chem import AllChem

def smiles_to_fingerprint(smiles, radius=2, nbits=2048):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    fp = AllChem.GetMorganFingerprintAsBitVector(mol, radius, nBits=nbits)
    return np.array(fp)
```
**Model**
`RandomForestClassifier(n_estimators=200, random_state=0)` with a stratified 20% test size.

---
## Results
| Metric   | Value |
| -------- | ----- |
| Accuracy | 0.868 |
| ROC AUC  | 0.911 |
**Classification Report**

| Class               | Precision | Recall | F1   | Support |
| ------------------- | --------- | ------ | ---- | ------- |
| non-penetrating (0) | 0.88      | 0.51   | 0.64 | 96      |
| penetrating (1)     | 0.87      | 0.98   | 0.92 | 312     |

**ROC curve**
![ROC curve](<file:///C:/Users/13913/Desktop/.ipynb_checkpoints/BBBP_figures/roc_curve.png>)

**Confusion matrix**
![Confusion matrix](<file:///C:/Users/13913/Desktop/.ipynb_checkpoints/BBBP_figures/confusion_matrix.png>)

**Sanity Check**
Caffeine, a known BBB-penetrant, was predicted using this model and achieved a 97.5% probability of penetration.

---
## Discussion
Although the overall accuracy (0.868) and ROC AUC (0.911) seem to be highly satisfactory, the class imbalance actually hides a critical weakness. In classification report, the recall of the minority group (0) is merely 0.51, which means about half of the non-penetrating molecules are misclassified by the model. However, almost all the molecules in the majority group (1) are correctly identified. This maybe attribute to the imbalanced class distribution. Since 76.4% of the data is positive, by just predicting "penetrating" all the time, the model can achieve a 0.764 accuracy, which is the trivial baseline modestly lower than 0.868 accuracy. This key point indicates the actual weakness of the model in learning the imbalanced classification task.

---

# 3. Quick Start
```bash
#1. Create an envirnment (recommended)
conda create -n bbbp python=3.13.15
conda activate bbbp
#2. Install dependencies
conda install pandas numpy scikit-learn matplotlib seaborn
conda install -c conda forge rdkit
#3. Run the notebook
jupyter notebook BBBP.ipynb
```
The full analysis is in [BBBP](<file:///C:/Users/13913/Desktop/.ipynb_checkpoints/BBBP.ipynb>)

---

# 4. Repository Structure
|--`BBBP.ipynb  #full pipline (data -> fingerprint -> model -> evaluation)`
|--`README.md`
|--`requirements.txt`

---

# 5. Future Work
- Handle class imbalance: `class_weight='balanced'`, SMOTE, or decision-threshold tuning to lift minority-class recall.
- Use PR-ACU (precision-recall) as the primary metric for this imbalance task.
- Upgrade features: RDKit 2D descriptors (LogP, TPSA, H-bond donors/acceptors) in addition to fingerprints.
- Try stronger models: gradient boosting, or graph neural networks (GNNs) that learn directly from molecular graphs.
- Connect to application: screen BBB permeability of ionizable lipids / LNP components relevant to CNS delivery.

---
[^1]: - Rogers & Hahn, _Extended-Connectivity Fingerprints_, _J. Chem. Inf. Model._ 2010. 

[^2]: - Breiman, _Random Forests_, _Mach. Learn._ 2001.

[^3]: Li, Y. et al (2024) Endothelial leakiness elicited by amyloid protein aggregation. _Nature communication._

[^4]: - Wu et al., _MoleculeNet: a benchmark for molecular machine learning_, _Chem. Sci._ 2018. 

---

*This is my first cheminformatics + ML project. Feedback welcome!*


