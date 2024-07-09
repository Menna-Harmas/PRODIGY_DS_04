# 🎭 SentiSocial: Decoding Public Opinion through AI

## 🌟 Project Overview

SentiSocial is an advanced sentiment analysis tool that dives deep into the ocean of social media data to extract valuable insights about public opinion and attitudes towards specific topics or brands. Using cutting-edge machine learning techniques, this project aims to transform raw social media text into actionable intelligence.

## 🚀 Features

- 📊 Data Preprocessing: Cleans and prepares raw social media text for analysis
- 🧠 Sentiment Classification: Utilizes Random Forest to categorize sentiments
- 🎨 Visualization: Creates insightful word clouds and charts to represent sentiment patterns
- 🔍 Brand Analysis: Provides sentiment breakdown across different companies/brands
- 📈 Performance Metrics: Evaluates model accuracy, precision, recall, and F1-score

## 🛠️ Technologies Used

- Python 3.x
- Pandas for data manipulation
- Scikit-learn for machine learning tasks
- NLTK for natural language processing
- Matplotlib and Seaborn for data visualization
- WordCloud for generating word clouds

## 📊 Dataset

This project uses the [Twitter Entity Sentiment Analysis dataset](https://www.kaggle.com/datasets/jp797498e/twitter-entity-sentiment-analysis) from Kaggle, which includes:
- Training set: 74,682 tweets
- Validation set: 1,000 tweets

## 🏗️ Project Structure

1. **Data Preparation**: 
   - Loading and initial cleaning of data
   - Handling missing values
   - Text preprocessing (lowercasing, removing URLs, @mentions, special characters)

2. **Exploratory Data Analysis**:
   - Generating word clouds for each sentiment category
   - Visualizing sentiment distribution
   - Creating heatmaps for sentiment across different companies

3. **Feature Extraction**:
   - Using TF-IDF vectorization to convert text to numerical features

4. **Model Building**:
   - Implementing Random Forest Classifier
   - Training the model on preprocessed data

5. **Model Evaluation**:
   - Calculating accuracy, precision, recall, and F1-score
   - Analyzing model performance on both training and validation sets

## 🎯 Results

Our Random Forest model achieved impressive results:
- Validation Accuracy: 96.2%
- Precision: 0.962
- Recall: 0.962
- F1-score: 0.962

These metrics demonstrate the model's strong performance in classifying sentiments across various topics and brands.

## 🙏 Acknowledgments

- Kaggle for providing the dataset
- The open-source community for the amazing libraries and tools

---

