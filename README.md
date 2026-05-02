# 🏏 IPL Score Predictor — Streamlit App

## Setup & Run

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Place the dataset
Put `ipl_dataset.csv` in the **same folder** as `ipl_app.py`.
The provided `ipl_dataset.csv` is a realistic synthetic dataset (25,000+ rows).
You can also upload your own CSV from within the app sidebar.

### 3. Launch
```bash
streamlit run ipl_app.py
```
Then open **http://localhost:8501** in your browser.

---

## Dataset columns required
| Column | Description |
|---|---|
| mid | Match ID |
| date | Match date |
| venue | Stadium name |
| bat_team | Batting team |
| bowl_team | Bowling team |
| batsman | Striker name |
| bowler | Current bowler name |
| runs | Runs scored so far |
| wickets | Wickets fallen so far |
| overs | Overs completed |
| striker | Striker end (0 or 1) |
| total | Final innings total (target) |

---

## App Pages

| Page | Description |
|---|---|
| 🏠 Overview | Dataset stats, sample data |
| 📊 EDA & Insights | Venue/batsman/bowler charts + correlation heatmap |
| 🤖 Model Training | GradientBoosting model metrics, actual vs predicted, residuals |
| 🎯 Predict Score | Live score predictor with run rate dashboard |

---

## Model
- **Algorithm**: Gradient Boosting Regressor (sklearn)
- **Features**: bat_team, bowl_team, venue, runs, wickets, overs, striker, batsman, bowler
- **Advantage over original**: No TensorFlow/Keras required, faster training, interpretable
- To use Keras/TF, replace the `train_model()` function in `ipl_app.py` with the original neural network code

---

## To use your own real IPL dataset
Replace `ipl_dataset.csv` with the Kaggle IPL dataset:
- https://www.kaggle.com/datasets/nowke9/ipldata
- Ensure column names match the table above (rename if needed)
