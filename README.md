# CardioCare

### 1. **CHEST\_PAIN** (Type of Chest Pain)

* **Definition**: Characterizes the type of chest pain experienced.
* **Typical classification** (0–3 or 1–4):

  * 0/1: **Typical Angina**
  * 1/2: **Atypical Angina**
  * 2/3: **Non-Anginal Pain**
  * 3/4: **Asymptomatic**

---

### 2. **RESTING\_BP** (Resting Blood Pressure)

* **Definition**: Blood pressure (systolic) when at rest, measured in mm Hg.
* **Normal Range**:

  * **90–120 mm Hg** (systolic)
  * **Less than 80 mm Hg** (diastolic)

---

### 3. **SERUM\_CHOLESTROL**

* **Definition**: Total cholesterol level in blood (mg/dL).
* **Desirable Range**:

  * **Less than 200 mg/dL**
  * **200–239 mg/dL** is borderline high
  * **240+ mg/dL** is high

---

### 4. **TRI\_GLYCERIDE** (Triglycerides)

* **Definition**: Fat (lipid) found in the blood, measured in mg/dL.
* **Normal Range**:

  * **Less than 150 mg/dL** is normal
  * **150–199 mg/dL**: borderline
  * **200–499 mg/dL**: high
  * **500+ mg/dL**: very high

---

### 5. **LDL** (Low-Density Lipoprotein - "Bad" Cholesterol)

* **Definition**: Carries cholesterol to tissues, contributes to plaque.
* **Optimal Range**:

  * **< 100 mg/dL** is optimal
  * **100–129 mg/dL**: near optimal
  * **130–159 mg/dL**: borderline high
  * **160–189 mg/dL**: high
  * **≥190 mg/dL**: very high

---

### 6. **HDL** (High-Density Lipoprotein - "Good" Cholesterol)

* **Definition**: Removes excess cholesterol from blood.
* **Desirable Range**:

  * **Men**: ≥ 40 mg/dL
  * **Women**: ≥ 50 mg/dL
  * **60+ mg/dL**: considered protective against heart disease

---

### 7. **FBS** (Fasting Blood Sugar)

* **Definition**: Blood glucose level after fasting (usually 8+ hours).
* **Normal Range**:

  * **< 100 mg/dL**: normal
  * **100–125 mg/dL**: prediabetes
  * **≥126 mg/dL**: diabetes

---

### 8. **RESTING\_ECG** (Electrocardiogram at rest)

* **Definition**: Measures electrical activity of the heart.
* **Typical Encoded Values**:

  * **0**: Normal
  * **1**: ST-T wave abnormality (suggesting ischemia or past MI)
  * **2**: Left ventricular hypertrophy

---

### 9. **MAX\_HEART\_RATE**

* **Definition**: Maximum heart rate achieved during stress test (beats per minute).
* **Normal Estimate**:

  * **220 − Age** is the general formula for max heart rate
  * During stress test: 85% of max is considered adequate

---

### 10. **ECHO** (Echocardiogram Result)

* **Definition**: Imaging test to visualize heart structure/function.
* **Common Outputs** (numeric or text):

  * Normal
  * LV Dysfunction (left ventricular)
  * Wall motion abnormalities
  * Ejection fraction < 50% may suggest cardiac issue

---

### 11. **TMT** (Treadmill Test or Stress Test)

* **Definition**: Measures how the heart performs under stress (exercise).
* **Outcomes**:

  * **Positive TMT**: Suggests ischemia (blocked arteries)
  * **Negative TMT**: Normal
  * Often encoded as: 0 = normal, 1 = abnormal
