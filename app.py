import streamlit as st
import pandas as pd
import json
import os

from dotenv import load_dotenv
from openai import OpenAI
from pycaret.regression import load_model, predict_model
from langfuse import Langfuse

# =====================
# Konfiguracja
# =====================

load_dotenv()

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY")
)

langfuse = Langfuse(
    public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
    secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
    host=os.getenv("LANGFUSE_BASE_URL")
)

model = load_model("models/halfmarathon_model")

# =====================
# UI
# =====================

st.title("🏃 Half Marathon Predictor")

user_text = st.text_area(
    "Opisz siebie",
    placeholder="Jestem mężczyzną, mam 35 lat i 5 km biegam w 25 minut"
)

# =====================
# Predykcja
# =====================

if st.button("Przewiduj"):

    try:

        response = client.responses.create(
            model="gpt-4.1-mini",
            input=f"""
Wyciągnij dane z tekstu.

Zwróć wyłącznie poprawny JSON.
Nie używaj markdown.
Nie używaj ```json.
Nie dodawaj żadnych komentarzy.

Format:

{{
    "gender": 0,
    "wiek": 0,
    "5km_seconds": 0
}}

Zasady:
- kobieta = 0
- mężczyzna = 1
- czas podaj w sekundach

Tekst:
{user_text}
"""
        )

        data = json.loads(response.output_text)
        langfuse.create_event(
            name="runner-analysis",
            input=user_text,
            output=data
        )

        langfuse.flush()

        if data["5km_seconds"] <= 0:

            st.error(
                "Nie znalazłem czasu na 5 km. "
                "Podaj np. '5 km biegam w 25 minut'."
            )

            st.stop()

        if data["wiek"] <= 0:

            st.error(
                "Nie znalazłem wieku."
            )

            st.stop()

        if data["gender"] not in [0, 1]:

            st.error(
                "Nie znalazłem informacji o płci."
            )

            st.stop()

        test_runner = pd.DataFrame({
            "gender": [data["gender"]],
            "wiek": [data["wiek"]],
            "5km_seconds": [data["5km_seconds"]]
        })

        prediction = predict_model(
            model,
            data=test_runner
        )

        wynik = prediction["prediction_label"].iloc[0]

        hours = int(wynik // 3600)
        minutes = int((wynik % 3600) // 60)
        seconds = int(wynik % 60)

        st.success("Predykcja wykonana")

        st.write(
            f"Przewidywany czas półmaratonu: "
            f"{hours:02d}:{minutes:02d}:{seconds:02d}"
        )

        with st.expander("Dane odczytane przez AI"):
            st.json(data)

    except Exception as e:
        st.error(f"Błąd: {str(e)}")
