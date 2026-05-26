# develop
Collaborating to pass the programming with python course
# Least-Squares Ideal Function Selection & Test Data Mapping

This project implements a complete analytical pipeline for selecting optimal ideal functions using least-squares minimisation and mapping unseen test data based on deviation thresholds. It was developed as part of the DLMDSPWP01 – Programming with Python module at IU International University of Applied Sciences.

---

## 📌 Project Overview

The goal of this project is to:

- Load and validate training, ideal, and test datasets  
- Identify the best-fitting ideal function for each training function using least-squares error  
- Compute deviation thresholds for robust test-data mapping  
- Map each test point to the closest ideal function  
- Store results in a SQLite database  
- Provide visualisations and a reproducible analysis workflow  

The full analysis was executed in a Jupyter Notebook and exported as images for documentation.

---

## 🧠 Methodology Summary

1. **Data Loading & Validation**  
   - CSV files for training, ideal, and test datasets  
   - Missing-value checks and structural validation  

2. **Least-Squares Ideal Function Selection**  
   - For each training function (y1–y4), compute SSE against all 50 ideal functions  
   - Select the ideal function with the minimum SSE  

3. **Deviation Threshold Computation**  
   - Compute maximum absolute residuals between training and ideal functions  
   - Use thresholds to validate test-data mapping  

4. **Test Data Mapping**  
   - For each test point, compute deviation from selected ideal functions  
   - Assign the test point to the ideal function with the smallest deviation  

5. **Database Storage**  
   - Results stored in `analysis.db` using SQLite  

6. **Unit Testing**  
   - A full unittest suite validates:
     - Data integrity  
     - Ideal function selection  
     - Threshold correctness  
     - Mapping logic  
     - Deviation behaviour  
   - **All 9 tests passed successfully.**

---

## 📂 Project Structure

