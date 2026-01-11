# AAPL stock price predictor using sentiment analysis

In this project we built an ML-pipeline that ingests daily AAPL stock data and articles from the yfinance library and outputs a prediction of tomorrows opening stock price. 
We display this information in a github page that is updated whenever a new inference is made. See screenshots or follow the link below. 
https://idguafi.github.io/nlp-stock-prediction/ 

### A note on why this project differs from our original proposition: 

We originally aimed to build a sentiment analysis predictor that ingested reddit sentiments via PRAW and predicted whether or not the price of said stock would see a 1% uptick or not. 
This, however, we realised was not possible due to the reddit API being unavailable to us. We tried applying for a token, but were denied access. 
Furthermore, it would be hard to find historical sentiment data for ALL stocks, which is another problem with our idea... 
##### TLDR: 
The original idea was too ambitious/unfeasible, and we decided to make the best of the situation by scaling back the scope, and applying a similar idea to the AAPL stock as a proof of concept.

# Technologies, data, model, training and inference

#### Technologies used

For the feature store/model registry we used hopsworks (cheers, jim!). Sentiment analysis model was finbert via huggingface, which can be found here: https://huggingface.co/ProsusAI/finbert. Lastly, we used github pages to display our model's predictions. 

#### Historical (static) data and training

The historical datasets used were: 

1. https://www.kaggle.com/datasets/frankossai/apple-stock-aapl-historical-financial-news-data which consisted of apple articles with corresponding sentiment scores
2. yfinance historical stock prices

Which were joined together as a pandas dataframe. We also added the target variable ('target_open') which we aimed to predict.

This data was then backfilled into a hopsworks feature store and was later used to train our xgboost model. 

#### Dynamic data - yfinance articles

For the dynamic data source, we ingested daily articles about AAPL via the yfinance library, along with todays opening price for plotting and accuracy measurement. 
The ingested articles were then fed through the FinBert NLP model that scored the sentiment and sentiment polarity for today which was then used to predict tomorrows price. 


# Model performance 

The model is slightly off when it comes to the opening price predictions, as shown by the gap between the predicted opening price and the actual opening price. 
Why is this the case? Our best guess is that it's due to the fact that AAPL traded at much lower prices for a majority of the data that we have access to. 
Looking at the apple stock price in january of 2024 compared to december the same year (which is the last year we had access to) you can see an astronimical leap from trading at 180 USD to 253 USD. 

I can hear you thinking that we easily could have accessed stock data for 2025 as well, and that is true, but the issue is that we didn't have historical sentiment for 2025 to go along with it, which means that it wouldn't have worked within the scope of our project.

##### TLDR: 
Major shift in prices in 2024, sustained at that level in 2025/26. 
Limited data at that price with associated sentiment score which affects predictive quality.


# High level architecture
```mermaid
graph TD
    %% Nodes
    subgraph Sources ["External Sources"]
        YF[Yahoo Finance API]
        HF[Hugging Face API]
        CSV[apple_news_data.csv]
    end

    subgraph Ingestion ["Ingestion & Processing"]
        DAILY[daily_data.ipynb]
        BACK[backfill.ipynb]
    end

    subgraph Platform ["Hopsworks Platform"]
        FS[("Feature Store<br/>(Sentiments & Prices)")]
        MR[Model Registry]
    end

    subgraph ML ["Machine Learning"]
        TRAIN[model_training.ipynb]
        INFER[inference.ipynb]
    end

    subgraph Web ["Dashboard"]
        JSON[docs/predictions.json]
        HTML[docs/index.html]
    end

    %% Edges
    YF --> DAILY & BACK
    HF --> DAILY & BACK
    CSV --> BACK
    
    DAILY --> FS
    BACK --> FS

    FS --> TRAIN
    TRAIN -- Save Model --> MR
    
    MR -- Load Model --> INFER
    FS -- Load Features --> INFER
    
    INFER --> JSON
    JSON --> HTML
```
