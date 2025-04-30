# 🚗 Telepass Insurance Purchase Prediction

Predictive modeling project to help Telepass forecast customer insurance purchases using user, vehicle, and transaction data.

## 📊 Problem

Telepass wants to identify likely insurance buyers and decide whether to expand its current brokerage model or begin selling insurance directly.

## 🔍 Approach

- Merged `Insurance Quotes` with `Transactions` on `client_id`.
- Engineered features from demographics, vehicle data, insurance pricing, and behavioral usage.
- Trained and compared:
  - Logistic Regression
  - Decision Tree
  - Random Forest

## ⚙️ Tools

- Python (pandas, scikit-learn)
- Excel (`Telepass.xlsx`)

## 📈 Results

| Model            | F1 (Buyers) | Recall | Precision | ROC AUC |
|------------------|-------------|--------|-----------|---------|
| Logistic Reg.    | 0.48        | 83%    | 34%       | 64.4%   |
| Decision Tree    | 0.53        | 85%    | 39%       | 71.6%   |
| Random Forest ✅ | **0.55**    | 74%    | 44%       | **76.1%** |

## 📌 Conclusion

Random Forest performed best overall. Key recommendation: Telepass should continue optimizing the brokerage model instead of selling insurance directly, due to regulatory complexity and lack of telematics data.

## ▶️ Run Locally

```bash
git clone https://github.com/jarvma18/SPMB-product-analytics-telepass-assignment.git
cd SPMB-product-analytics-telepass-assignment
python main.py
