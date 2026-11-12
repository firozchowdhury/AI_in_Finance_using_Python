# AI in Finance using Python – Exercise Notebooks

This repository contains my exercise notebooks for the course **AI in Finance using Python (3 ECTS)**, taken at Metropolia University of Applied Sciences through the Viope platform.

> **Note:** These are my personal course learning notes and practical practice. The notebooks include my own comments, experiments and repeated attempts, so some files are work in progress rather than polished final solutions.
>
> **Course page (Viope, login required):** [AI in Finance using Python](https://vw5.viope.com/organisations/metropolia/courses/16389/chapters/148805/sections/426995)

The notebooks cover the full workflow of applying data science and machine learning to financial data: cleaning and exploring data, forecasting time series, analysing financial text, measuring risk, detecting fraud and building portfolios.

**Author:** Md Firoz Chowdhury, Bachelor of Computer Applications student at Häme University of Applied Sciences (HAMK), with a background in accounts payable, accounts receivable and treasury.

---

## Environment

- Python 3.13, run locally in **VS Code with Jupyter notebooks** on Windows
- Main libraries: `pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`, `statsmodels`, `prophet`, `tensorflow / keras`, `torch`, `transformers`, `yfinance`, `minisom`, `scipy`, `nltk`, `textblob`

Install the main packages with:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn statsmodels prophet tensorflow torch transformers yfinance minisom scipy nltk textblob kagglehub
```

TensorFlow runs on the CPU on native Windows, so the deep learning notebooks take a few minutes to train.

---

## Contents

### 1. Data preprocessing and exploratory analysis

| Notebook | What it does |
|---|---|
| `2.DataPreprocessing.ipynb` | Cleans the `Financials4.csv` sales dataset: strips spaces from column names, converts text with `$`, `,` and `-` to numbers, removes duplicates, and compares dropping missing values with group-mean imputation. Summarises sales and profit by segment and country with box plots, a correlation heatmap and `groupby`. |
| `3.DataSummaries(EDA).ipynb` | Mean, median, quartiles and `describe()` on financial data. |
| `2.4Normality&Scalling.ipynb` | Dice-throw simulation showing the law of large numbers and the normal (bell) curve. |
| `6.9EDAexercise.ipynb` | EDA on `sales_data.csv`: correlation matrix, total sales by region, maximum sale per sales rep and mean unit price by category. |
| `DataProcessing.py` | Short script for loading and inspecting a CSV file. |

### 2. Time series analysis and forecasting

| Notebook | What it does |
|---|---|
| `3.Timeseries.ipynb` | IBM and Apple price plots, centred moving averages, peak and valley detection, and a multivariate LSTM predicting next-day Open and Close prices. |
| `7.8TrendingEx.ipynb` | Tesla prices: trailing 5-day rolling mean of Close, centred rolling mean of Volume, and local peaks. |
| `3.6Prophet.ipynb` | Prophet forecast of monthly car sales with multiplicative seasonality and an 85% uncertainty interval. |
| `7.6ARIMAforecasting.ipynb` | ARIMA(2,1,1) on daily minimum temperatures, with an ADF stationarity test, differencing, ACF/PACF plots and evaluation metrics. |
| `7.7LSTMforecasting.ipynb` | LSTM forecast of Bitcoin prices (2020–2026) with 60-day input windows and a time-ordered train/validation/test split. Test results: RMSE ≈ 3,959 USD, MAE ≈ 3,020 USD, MAPE ≈ 3.6%. Includes a 60-day recursive forecast. |
| `7.7.ipynb`, `LSTM_Practice.ipynb` | The same LSTM workflow repeated on other assets (Bitcoin variant and Meta stock, MAPE ≈ 3.5%). |

### 3. Sentiment analysis

| Notebook | What it does |
|---|---|
| `SentimentAnly.ipynb` | Lexicon-based sentiment with VADER and TextBlob. |
| `LSTMSentimentAnly.ipynb` | Text-cleaning pipeline for tweets: HTML and URL removal, punctuation, chat abbreviations and stop words. |
| `4.5.5 Sentiment analysis.ipynb` | Financial PhraseBank (3,448 sentences, 75% annotator agreement). Compares a bidirectional LSTM with class weights against a fine-tuned DistilBERT model. |

Results on the test set:

| Model | Accuracy | Macro F1 |
|---|---|---|
| Bidirectional LSTM | 0.803 | 0.744 |
| DistilBERT (fine-tuned, 3 epochs) | 0.925 | 0.904 |

### 4. Risk management

| Notebook | What it does |
|---|---|
| `5.6.5 VaR and Volatility.ipynb` | Two years of NVIDIA prices: daily returns, 21-day rolling volatility, and historical vs. parametric Value at Risk at 95% and 99%, with a breach count to compare the methods. |

Key results: annualised volatility ≈ 44%; 99% VaR is 6.9% (historical) vs. 6.3% (parametric). Returns have a kurtosis of 6.8 (fat tails), so the normal-distribution VaR underestimates the most extreme losses.

### 5. Fraud detection

| Notebook | What it does |
|---|---|
| `7.7 Sampling exploration.ipynb` | Credit card fraud (284,807 transactions, 0.17% fraud). Compares Logistic Regression, KNN, SVC, Decision Tree and Random Forest trained on 1:1 and 2:1 undersampled data, evaluated on the original imbalanced test set. The 2:1 ratio improves precision for every model while recall stays similar. |
| `6.7.8 SOM.ipynb` | Self-Organizing Map (MiniSom) on the Kaggle credit card transactions dataset. Compares the undersampled 1:1 case (fraud recall 0.84) with 50,000 realistic records, where majority-vote labelling detects no fraud despite 99.4% accuracy. Includes scatter and 3D plots of the predicted clusters. |

### 6. Portfolio construction

| Notebook | What it does |
|---|---|
| `7.6.6 SP500 clustering.ipynb` | Hierarchical clustering of 100 S&P 500 stocks with four linkage methods (single, complete, average, ward) and two correlation distances. Clusters are built on 2023–2024 data and checked on 2025. Half of the clusters that beat the index in the training period also beat it in the test period. |

---

## Datasets

| File | Source / content |
|---|---|
| `Financials4.csv`, `sales_data.csv` | Course sales datasets |
| `IBM.csv`, `final_data_adj.csv`, `tsla.csv` | Course stock price data |
| `BTC-USD.csv`, `META.csv`, `NVDA.csv`, `GC-F.csv` | Downloaded with `yfinance` |
| `monthly-car-sales.csv`, `daily-min-temperatures.csv` | Classic time series datasets |
| `fpb.zip` | Financial PhraseBank (Hugging Face) |
| `sentiment_tweets3.csv.zip` | Tweet sentiment dataset |

Large datasets are not stored here and are downloaded inside the notebooks: the credit card fraud datasets (Kaggle, via `kagglehub` or a public mirror) and S&P 500 prices (via `yfinance`).

---

## What I learned

- **Real data needs cleaning first.** Hidden spaces, text-formatted numbers, duplicate rows and multi-row CSV headers had to be fixed before any analysis.
- **A good-looking metric is not the same as a good model.** Accuracy is misleading on imbalanced data, a low MAPE should be compared with a naive baseline, and results should be checked on data the model has not seen.
- **The model must match the data.** A plain ARIMA forecast flattened out because it has no seasonal component; seasonal data needs a seasonal model.
- **Pre-trained language models are powerful.** DistilBERT clearly outperformed an LSTM trained from scratch on financial sentences, especially on the rare negative class.
- **Risk models have assumptions.** Parametric VaR assumes normal returns, which underestimates tail risk for volatile stocks.
- **Domain knowledge matters.** My accounting background helped me judge results, for example that every false fraud alert costs an investigator's time, so precision matters as much as recall.

---

## How to run

1. Clone the repository and open the folder in VS Code (or Jupyter).
2. Install the packages listed above.
3. Open a notebook and run the cells from top to bottom. Notebooks that download data need an internet connection.
