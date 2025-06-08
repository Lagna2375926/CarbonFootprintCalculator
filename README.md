# Carbon Footprint Predictor

**Carbon Footprint Predictor** is a web application that estimates the embodied carbon emissions (CO₂e) of building construction projects based on key design parameters. This tool helps architects, engineers, and sustainability professionals assess and minimize the environmental impact of their projects using machine learning.

---

## Features

- Predicts embodied carbon emissions based on inputs like floor area, number of floors, concrete volume, steel mass, climate zone, and transport radius.
- Uses a trained XGBoost regression model for accurate predictions.
- Interactive and user-friendly interface built with Streamlit.
- Easily accessible via a public URL using ngrok tunneling (for local development).

---

## Technologies Used

- Python  
- Streamlit  
- XGBoost  
- Pandas  
- NumPy  
- Joblib  
- Pyngrok

---

## Setup Instructions

1. Clone the Repository

```bash
git clone <repository-url>
```

2. Install Dependencies

```bash
pip install -r requirements.txt
```

3. Run the Application

```bash
streamlit run app.py
```
