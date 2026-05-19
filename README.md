# 📈 Financial Sentiment Analysis & Price Correlation

## 🎯 Problem Statement
Financial news moves the stock market, but there's too much of it to read manually. Furthermore, standard language models struggle with financial jargon (e.g., "cutting losses" is actually a good thing). This project automates sentiment extraction from financial news using NLP to see if it correlates with short-term stock price changes.

## 🗃️ Dataset Source
- **Financial PhraseBank (Kaggle)**: 4,846 sentences from English financial news. 
- **Labels**: Annotated by experts as `positive`, `neutral`, or `negative`.
- **Live Data**: Recent news and stock prices were fetched live using the **yfinance** API.

## 🛠️ Approach
1. **EDA**: Visualized class distributions, sentence lengths, and word clouds.
2. **Baseline**: Trained a Logistic Regression classifier using `TfidfVectorizer`.
3. **FinBERT**: Evaluated `ProsusAI/finbert`, a BERT model specifically fine-tuned for finance.
4. **Live Market Correlation**: Scored live news for 10 major tickers and plotted the average sentiment against their 10-day price changes to find correlations.

## 📊 Key Findings
- **FinBERT Wins**: The baseline model achieved ~73% accuracy but struggled to tell negative and neutral news apart. FinBERT understood the financial context much better, hitting **~88% accuracy**.
- **Sentiment Matches Price**: Our live test on 10 tickers showed a clear positive trend—highly positive news sentiment generally aligned with upward short-term price movement.

---

## 🚀 Getting Started

### 1. Install dependencies
```bash
pip install transformers torch yfinance scikit-learn pandas numpy matplotlib seaborn wordcloud
pip install "numpy<2.0.0" # Required for PyTorch compatibility
```

### 2. Get the Data
Download `all-data.csv` from [Kaggle](https://www.kaggle.com/datasets/ankurzing/sentiment-analysis-for-financial-news) and place it in the project root.

### 3. Run it
Open `code.ipynb` in Jupyter and run all cells. *(Note: A GPU is recommended for the final FinBERT section).*
