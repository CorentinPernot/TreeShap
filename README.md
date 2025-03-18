# TreeShap

# 🎯 **Goals of the Repo**  

This notebook aims to **reproduce the results** of the paper  
📖 *Explainable AI for Trees: From Local Explanations to Global Understanding* by **S.L. Lundberg et al. (2019)**.  

It leverages some code from [this repository](https://github.com/suinleelab/treeexplainer-study) and focuses **exclusively** on the **NHANES I dataset**, which is one of the three datasets used in the original study.  

---

## **Outline**  

🔹 **1. Data Loading** – Load the dataset following the original approach.  
🔹 **2. Problem Exploration** – Understand key features and distributions.  
🔹 **3. Model Training** – Train:  
   - 🌳 an **XGBoost model**  
   - 📉 a **linear model**  
   *(Hyperparameter tuning via **RandomSearch**)*
   
🔹 **4. Algorithm Implementation** – Code Algorithm 1 (Explainer) **from scratch**.  
🔹 **5. SHAP Comparison** – Compare our implementation with `TreeExplainer` from the **SHAP** library.  
🔹 **6. Complexity Analysis** – Evaluate **computational efficiency**.  
🔹 **7. Dependence Plots** – Analyze **feature dependencies & interactions**.  
🔹 **8. Global vs. Local SHAP** – Compare **SHAP values** with the **Gain method**.  

---
