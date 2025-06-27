# classifAI: Smart Review Classifier

Smart review classifier that reads user feedback and routes it where it belongs — praise, bugs, requests, or just spam. Built with FastAPI, machine learning, and a modern web interface.

---

## 🚀 Features
- **5-Class Review Classification:** Positive, Negative, Bug Report, Feature Request, Spam
- **Semi-Automatic Labeling:** Heuristic rules + ML
- **Data Cleaning & Exploration:** Jupyter notebooks for transparency
- **Production-Ready API:** FastAPI backend
- **Modern Web Interface:** Classify reviews instantly in your browser
- **Prediction Logging:** All predictions stored in SQLite
- **Easy Deployment:** Render-ready, cloud-friendly

---

## 📁 Folder Structure
```
classifAI/
├── api/                  # FastAPI app and backend code
│   └── main.py
├── data/                 # Data and models
│   ├── data_cleaned.csv
│   ├── labeled_reviews.csv
│   └── models/
│       ├── classifier_model.pkl
│       └── tfidf_vectorizer.pkl
├── templates/            # HTML templates for web UI
│   └── index.html
├── requirements.txt      # All dependencies
├── render.yaml           # Render deployment config
├── README.md             # This file
├── data_exploration.ipynb         # Data cleaning notebook
├── labeling_rules.ipynb           # Labeling rules notebook
└── model_training.ipynb           # Model training notebook
```

---

## 🛠️ Setup (Local)

1. **Clone the repo:**
   ```bash
   git clone https://github.com/yourusername/classifAI.git
   cd classifAI
   ```

2. **Create and activate a virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **(Optional) Run Jupyter notebooks:**
   ```bash
   jupyter notebook
   # Open and run the notebooks in the project root:
   # - data_exploration.ipynb
   # - labeling_rules.ipynb
   # - model_training.ipynb
   ```

5. **Train the model (if not already trained):**
   - Run the notebooks in order:
     1. `data_exploration.ipynb`
     2. `labeling_rules.ipynb`
     3. `model_training.ipynb`
   - This will produce `data/models/classifier_model.pkl` and `tfidf_vectorizer.pkl`

---

## 🌐 Run the Web App Locally

```bash
python api/main.py
```
- Visit [http://localhost:8000](http://localhost:8000) for the web interface
- API docs at [http://localhost:8000/docs](http://localhost:8000/docs)

---

## 🖥️ Web Interface
- Enter a review in the textbox
- Click "Classify Review"
- See the predicted label, confidence, and stats

---

## 🧑‍💻 API Usage
### **POST /predict**
- **Request:**
  ```json
  { "review": "This app keeps crashing when I open messages" }
  ```
- **Response:**
  ```json
  {
    "review": "This app keeps crashing when I open messages",
    "predicted_label": "Bug Report",
    "confidence": 0.98,
    "timestamp": "2024-06-27T12:34:56.789Z"
  }
  ```

### **GET /stats**
- Returns prediction stats and label breakdown

---

## ☁️ Deploy on Render
1. **Push your code to GitHub**
2. **Create a free account at [render.com](https://render.com)**
3. **Click "New Web Service" → Connect your repo**
4. **Render auto-detects `render.yaml` and sets up everything**
5. **First deploy takes 2-5 minutes**
6. **Visit your live app at the provided URL!**

---

## 📝 Customization & Extending
- **Add more classes:** Update labeling rules and retrain
- **Improve labeling:** Refine heuristics or add manual review
- **Swap models:** Try XGBoost, SVM, or deep learning
- **Batch prediction:** Add CSV upload endpoint
- **Analytics:** Build dashboards from the SQLite logs

---

## 🤝 Credits
- Built with [FastAPI](https://fastapi.tiangolo.com/), [scikit-learn](https://scikit-learn.org/), [Jupyter](https://jupyter.org/), and [Render](https://render.com/)
- Inspired by real-world app review workflows

---

## 📬 Questions?
Open an issue or reach out! This project is beginner-friendly and open to contributions.
