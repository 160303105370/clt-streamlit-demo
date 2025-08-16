# clt-streamlit-demo
Streamlit app + report demonstrating the Central Limit Theorem and a regression analysis of time on page vs revenue.

This project has 2 components:- 

1. **Interactive Streamlit App**  
   - A visual, interactive way to explore the **Central Limit Theorem (CLT)**.  

2. **Statistical Analysis Report (PDF)**
   - Includes an upload feature so you can test CLT and bootstrap confidence intervals on your **own CSV data**. 
   - Explores the relationship between **time on page (`top`)** and **revenue**.  
   - Shows results from simple regression and from models that control for platform/browser/site.

  
## Features

### Tab 1: CLT Playground
- Choose a base distribution (**Exponential, Uniform, Poisson**).  
- Set sample size and number of draws.  
- See how the distribution of **sample means** becomes bell-shaped (normal) as the sample size increases.  
- Overlay a Normal curve + KS test for goodness-of-fit.

### Tab 2: Your Data (Upload)
- Upload a CSV with at least one numeric column (e.g., `revenue`).  
- See:
  - **Raw distribution** of your chosen variable.  
  - **Bootstrap distribution of sample means** with a 95% confidence interval. 
- (Optional) Peek at regression summaries (`revenue ~ top`) with and without controls.


##  Repository Structure

├── streamlit_app.py # Main Streamlit app
├── requirements.txt # Python dependencies
├── report/ # PDF report & appendix
│ ├── time_on_page_revenue_report.pdf
│ └── code_appendix.pdf
└── README.md # This file


## How can you run the project

-**Install dependencies:**
  pip install -r requirements.txt
  
- **Run locally:**
  streamlit run streamlit_app.py  


**Author**
- [LinkedIn](https://www.linkedin.com/in/prashanttrivedi370/)
- [GitHub](https://github.com/160303105370)
