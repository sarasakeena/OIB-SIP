# OIB-SIP

OASIS INFOBYTE INTERNSHIP

1.EMAIL SPAM DETECTION

This is a simple spam detection machine learning model using the Multinomial Naive Bayes algorithm. It classifies SMS messages as Spam or Not Spam. Built as part of the Oasis Infobyte Internship Program.

## Technologies Used
- Python
- scikit-learn
- pandas
- CountVectorizer
- VS Code
- Streamlit - UI

## Concept

This project focuses on detecting **spam emails or SMS messages** using **Machine Learning**. The key idea is to train a model that can differentiate between *spam* and *not spam* (ham) messages based on the text content.

The steps involved in the project are:

1. **Data Loading**: The dataset (`spam.csv`) contains labeled SMS messages as 'spam' or 'ham'.
2. **Text Preprocessing**: Cleaning and vectorizing text using `CountVectorizer` to convert it into a format the machine can understand.
3. **Train-Test Split**: The data is split into training and testing sets to evaluate performance.
4. **Model Training**: A **Multinomial Naive Bayes** classifier is trained on the training data.
5. **Prediction**: The model predicts whether new/unseen messages are spam or not.
6. **Evaluation**: We evaluate the model using metrics like accuracy, precision, recall, and confusion matrix.

This is an example of a **Supervised Learning** classification task, where:
- **Input (X)**: SMS message text
- **Output (Y)**: Spam or Not Spam label

The `Naive Bayes` algorithm is particularly well-suited for text classification tasks because of its simplicity, speed, and surprisingly good performance on spam filtering problems.

---

**Why Naive Bayes?**  
- It works well with **word frequency data**
- It’s **fast and efficient**
- It handles **large text data** with ease

  
## How to Run
1. Clone this repo
2. Install required libraries: scikit-learn, pandas, streamlit
3. Run the script

## Deployment using Streamlit

This project includes a **Streamlit interface** for deploying the spam detection model as a simple and interactive web app.

### What is Streamlit?

[Streamlit](https://streamlit.io) is an open-source Python library that lets you create **beautiful web apps** for data science and machine learning projects — with **just a few lines of Python code**.

It’s perfect for deploying ML models without needing complex front-end development.



### How to Run the App Locally

1. Make sure you have Python installed.
2. Install Streamlit (if not already installed): pip install streamlit
3. In the terminal, navigate to your project folder and run: streamlit run spam.py
4. This will open the app in your browser at: http://localhost:8501

## Sample Output (without streamlit)
Accuracy : 0.98
['Spam']



2. SALES PREDICTION

This is a simple sales prediction machine learning model using **Linear Regression**. It predicts product sales based on advertising budgets for **TV**, **Radio**, and **Newspaper** campaigns. Built as part of the Oasis Infobyte Internship Program.

## Technologies Used
- Python
- pandas
- scikit-learn
- matplotlib
- seaborn
- Streamlit - UI

## Concept

This project focuses on predicting **product sales** using **Supervised Machine Learning**. The model learns the relationship between advertising spend and resulting sales figures.

The steps involved in the project are:

1. **Data Loading**: The dataset (`Advertising.csv`) contains advertising budgets for TV, Radio, Newspaper, and actual sales.
2. **Exploratory Data Analysis (EDA)**: Visualizing data and correlations using `matplotlib` and `seaborn`.
3. **Train-Test Split**: The dataset is split into training and testing sets.
4. **Model Training**: A **Linear Regression** model is trained on the data.
5. **Prediction**: The model predicts sales for new budget inputs.
6. **Evaluation**: The model is evaluated using **R² score** and **Mean Squared Error (MSE)**.

This is a **Regression** task, where:
- **Input (X)**: TV, Radio, Newspaper ad budgets
- **Output (Y)**: Predicted Sales

The Linear Regression algorithm helps us find the **best-fit line** that represents the relationship between features and target.

---

**Why Linear Regression?**  
- It’s simple and interpretable  
- Great for predicting **numerical outcomes**  
- Helps understand feature impact on target variable

## How to Run
1. Clone this repo
2. Install required libraries: scikit-learn, pandas
3. Run the training script 


3. UNEMPLOYMENT ANALYSIS

This is a **data analysis and visualization project** focused on understanding **unemployment trends in India**, especially during the COVID-19 period. Built as part of the Oasis Infobyte Internship Program.

## Technologies Used
- Python
- pandas
- matplotlib
- seaborn
- Jupyter Notebook / VS Code

## Concept

This project focuses on analyzing the **state-wise and time-wise unemployment rate in India** using a real dataset. The goal is to generate **insights** using visualizations.

The steps involved in the project are:

1. **Data Loading**: The dataset is loaded from a CSV file, containing unemployment rates by area, region, and date.
2. **Data Cleaning**: Checking for missing values, renaming columns, and converting dates.
3. **Exploratory Data Analysis (EDA)**: Using graphs to understand patterns and trends.
4. **Visualization**: Graphs such as line plots, bar charts, and box plots are used to show unemployment by region and time.
5. **Insights**: The analysis reveals which regions and areas were more affected during the COVID-19 phase.

This is a **Data Analysis** project — no machine learning is involved. It helps interpret real-world economic impact using data.

---

**Why Data Analysis?**  
- Helps visualize real-world trends  
- Supports **data-driven decision making**  
- Simplifies complex datasets through visuals

## How to Run
1. Clone this repo
2. Open the project in Jupyter Notebook or VS Code
3. Run the notebook file step by step
---

### Sample Visual Outputs
- 📈 **Line Plot** – Unemployment trend over time
- 📊 **Bar Chart** – Unemployment by region
- 📦 **Box Plot** – Urban vs Rural unemployment

---



Author
Sara Sakeena
Intern @ Oasis Infobyte

## Acknowledgments
Thanks to Oasis Infobyte for the opportunity to build this project!



