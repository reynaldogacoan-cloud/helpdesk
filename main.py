
from flask import Flask, request, jsonify
import pandas as pd
import gspread
from google.auth import default
from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk

app = Flask(__name__)

SPREADSHEET_ID = "1DmA5tgK2d-zusPEvNda2sNkprNjabIUs-oyt4cn6QEg"

def preprocess_text(text, lemmatizer, stop_words):
    if isinstance(text, str):
        words = text.lower().split()
        words = [lemmatizer.lemmatize(w) for w in words if w not in stop_words]
        return ' '.join(words)
    return ''

@app.route("/", methods=["GET"])
def home():
    return "✅ Helpdesk Classifier API aktif."

@app.route("/run", methods=["POST"])
def run_classification():
    try:
        # auth from environment (Cloud Run service identity)
        creds, _ = default()
        client = gspread.authorize(creds)

        # load spreadsheets
        spreadsheet = client.open_by_key(SPREADSHEET_ID)
        sheet_model = spreadsheet.worksheet("Model")
        sheet_data = spreadsheet.worksheet("Raw")
        sheet_catlist = spreadsheet.worksheet("Category List")

        df_model = pd.DataFrame(sheet_model.get_all_records())
        df_data = pd.DataFrame(sheet_data.get_all_records())
        df_catlist = pd.DataFrame(sheet_catlist.get_all_records())

        # nltk
        nltk.download('stopwords')
        nltk.download('wordnet')
        lemmatizer = WordNetLemmatizer()
        stop_words = set(stopwords.words('indonesian'))

        df_model.dropna(subset=["Detail Problem (SDG)"], inplace=True)
        df_model["Detail Problem (SDG)"] = df_model["Detail Problem (SDG)"].apply(lambda t: preprocess_text(t, lemmatizer, stop_words))

        # train quick models
        def train_nb_model(column):
            df_temp = df_model.dropna(subset=[column])
            X = df_temp["Detail Problem (SDG)"]
            y = df_temp[column]
            model = make_pipeline(TfidfVectorizer(), MultinomialNB())
            grid = GridSearchCV(model, {'multinomialnb__alpha': [0.1, 1.0, 10.0]}, cv=3)
            grid.fit(X, y)
            return grid.best_estimator_

        model_category = train_nb_model("Category")
        model_area = train_nb_model("Area")
        model_main = train_nb_model("Main Category")

        # select unclassified rows
        df_unclassified = df_data[df_data["Main Category"] == ""].copy()
        if df_unclassified.empty:
            return jsonify({"status":"done","message":"Semua data sudah diklasifikasi."})

        df_unclassified["Cleaned"] = df_unclassified["Detail Problem (SDG)"].apply(lambda t: preprocess_text(t, lemmatizer, stop_words))

        df_unclassified["Pred_Category"] = model_category.predict(df_unclassified["Cleaned"])

        pred_area = []
        for text, cat in zip(df_unclassified["Cleaned"], df_unclassified["Pred_Category"]):
            valid_areas = df_catlist[df_catlist["Category"] == cat]["Area"].dropna().unique()
            proba = model_area.predict_proba([text])[0]
            classes = model_area.classes_
            filtered = {cls: p for cls, p in zip(classes, proba) if cls in valid_areas}
            pred_area.append(max(filtered, key=filtered.get) if filtered else classes[proba.argmax()])
        df_unclassified["Pred_Area"] = pred_area

        pred_main = []
        for text, cat in zip(df_unclassified["Cleaned"], df_unclassified["Pred_Category"]):
            valid_main = df_catlist[df_catlist["Category"] == cat]["Main Category"].dropna().unique()
            proba = model_main.predict_proba([text])[0]
            classes = model_main.classes_
            filtered = {cls: p for cls, p in zip(classes, proba) if cls in valid_main}
            pred_main.append(max(filtered, key=filtered.get) if filtered else classes[proba.argmax()])
        df_unclassified["Pred_Main_Category"] = pred_main

        # prepare batch updates (columns X=24, Y=25, Z=26 -> A=1 so X=24)
        updates = []
        for idx, (_, row) in enumerate(df_unclassified.iterrows(), start=2):
            updates.append({'range': f'X{idx}', 'values': [[row["Pred_Main_Category"]] ] })
            updates.append({'range': f'Y{idx}', 'values': [[row["Pred_Area"]] ] })
            updates.append({'range': f'Z{idx}', 'values': [[row["Pred_Category"]] ] })

        sheet_data.batch_update(updates)

        return jsonify({"status":"ok","updated":len(df_unclassified)})

    except Exception as e:
        return jsonify({"status":"error","message":str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
