# Machine-Learning-Stock-Analyzer
Machine Learning Stock Market Analysis Tool
By: Morgan Evans


Overview:
My project is a user-friendly tool designed for individuals interested in analyzing stock market data using machine learning. This project aims to provide users with a simple yet effective way to perform stock market analysis and make informed investment decisions. Using historical stock price data and a Random Forest machine learning model, this tool predicts whether the price of a given stock will increase or decrease in the future. The model's configurations, including the choice of predictors, number of estimators, minimum sample split, and probability threshold, can be easily adjusted via a command-line interface. This allows users to customize the model according to their preferences and investment strategies.


Usage:

There is only one file (main.py) that you need to run. Then follow the prompts to input the data and parameters you want to analyze. The prompts provide exactly the input they want, so they do not have any error handling in place. Once the model is running it takes awhile to get the prediction accuracy. The lower the number of estimators is the faster it will run. 

The order of which I would suggest making changes to the model is first the number of estimators and minimum sample split because those typically have the smallest change. Then changing the probability threshold, which can actually have the largest change. Then removing some predictors. Then adding RSI. And finally end with adding Horizon because it will eliminate all the other predictors. 


Warnings:

While the model aims to provide insights into potential future stock price movements based on historical data and machine learning algorithms, it should not be used as a sole basis for making investment decisions. I am not a financial advisor, and the information provided by this tool should not be construed as financial advice. Investing in the stock market carries inherent risks, and past performance is not indicative of future results. Use this tool at your own risk. I do not guarantee the accuracy or reliability of the predictions made by the model, and I accept no liability for any financial losses incurred as a result of using this tool for investment purposes.



