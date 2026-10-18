
import streamlit as st
import pandas as pd
from collections import Counter
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score

st.set_page_config(
    page_title="BioAI DNA Analyzer",
    page_icon="🧬",
    layout="wide"
)

st.title("🧬 BioAI — DNA Sequence Analyzer")
st.write(
    "Analyze DNA composition and explore machine-learning "
    "promoter classification."
)

st.divider()
st.header("DNA Sequence Analysis")

dna = st.text_area(
    "Enter DNA sequence",
    placeholder="Example: ATGCGTACGTAG"
)

if st.button("🔬 Analyze DNA"):
    seq = "".join(dna.upper().split())

    if not seq:
        st.warning("Please enter a DNA sequence.")
    elif any(base not in "ATGC" for base in seq):
        st.error("Use only A, T, G and C.")
    else:
        counts = Counter(seq)
        length = len(seq)
        gc = (counts["G"] + counts["C"]) / length * 100

        c1, c2, c3 = st.columns(3)
        c1.metric("Sequence Length", f"{length} bp")
        c2.metric("GC Content", f"{gc:.2f}%")
        c3.metric("AT Content", f"{100 - gc:.2f}%")

        st.subheader("Nucleotide Composition (%)")
        st.bar_chart({
            base: counts[base] / length * 100
            for base in "ATGC"
        })

        st.subheader("Nucleotide Counts")
        st.write({base: counts[base] for base in "ATGC"})

st.divider()
st.header("🤖 Machine Learning Promoter Classifier")

st.caption(
    "The model uses a public bacterial promoter dataset. "
    "Predictions are educational and require independent validation."
)

@st.cache_resource
def train_model():
    dataset = load_dataset(
        "neuralbioinfo/bacterial_promoters",
        split="train"
    )

    df = pd.DataFrame(dataset)

    required = {"segment", "y"}
    if not required.issubset(df.columns):
        raise ValueError(
            f"Expected columns {required}; found {list(df.columns)}"
        )

    df = df[["segment", "y"]].dropna().copy()
    df["segment"] = df["segment"].astype(str).str.upper()
    df["y"] = pd.to_numeric(df["y"], errors="coerce")
    df = df.dropna()
    df["y"] = df["y"].astype(int)

    df = df[
        df["y"].isin([0, 1])
        & df["segment"].str.fullmatch("[ATGC]+")
    ]

    groups = []
    for label in [0, 1]:
        group = df[df["y"] == label]
        if len(group) < 2:
            raise ValueError(
                f"Not enough training examples for label {label}."
            )
        groups.append(
            group.sample(
                n=min(5000, len(group)),
                random_state=42
            )
        )

    data = pd.concat(groups).sample(
        frac=1,
        random_state=42
    )

    X_train, X_test, y_train, y_test = train_test_split(
        data["segment"],
        data["y"],
        test_size=0.2,
        random_state=42,
        stratify=data["y"]
    )

    model = Pipeline([
        (
            "features",
            TfidfVectorizer(
                analyzer="char",
                ngram_range=(2, 5),
                min_df=2
            )
        ),
        (
            "classifier",
            LogisticRegression(max_iter=500)
        )
    ])

    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)

    return model, accuracy, len(X_train), len(X_test)

if st.button("Train Real Dataset Model"):
    with st.spinner(
        "Loading dataset and training model. Please wait..."
    ):
        try:
            model, accuracy, train_size, test_size = train_model()

            st.session_state["bioai_model"] = model
            st.session_state["bioai_accuracy"] = accuracy
            st.session_state["bioai_train_size"] = train_size
            st.session_state["bioai_test_size"] = test_size

            st.success("Model training completed!")

        except Exception as error:
            st.error("Model training failed.")
            st.code(str(error))

if "bioai_model" in st.session_state:
    st.subheader("Model Evaluation")

    st.metric(
        "Held-out Test Accuracy",
        f"{st.session_state['bioai_accuracy']:.2%}"
    )

    st.write(
        f"Training examples: {st.session_state['bioai_train_size']}"
    )
    st.write(
        f"Test examples: {st.session_state['bioai_test_size']}"
    )

    st.caption(
        "This accuracy is measured on a held-out split of the sampled "
        "dataset. It is not independent scientific validation."
    )

    st.subheader("Predict a DNA Sequence")

    test_dna = st.text_input(
        "DNA sequence",
        placeholder="Example: ATGCGTACGTAG"
    )

    if st.button("Predict Promoter Class"):
        seq = "".join(test_dna.upper().split())

        if not seq:
            st.warning("Please enter a DNA sequence.")
        elif any(base not in "ATGC" for base in seq):
            st.error("Use only A, T, G and C.")
        else:
            model = st.session_state["bioai_model"]
            prediction = int(model.predict([seq])[0])
            probabilities = model.predict_proba([seq])[0]
            classes = list(
                model.named_steps["classifier"].classes_
            )
            promoter_index = classes.index(1)
            probability = probabilities[promoter_index]

            if prediction == 1:
                st.success("Predicted class: Promoter (label 1)")
            else:
                st.info("Predicted class: Non-promoter (label 0)")

            st.write(
                f"Model-estimated promoter probability: "
                f"{probability:.1%}"
            )

            st.caption(
                "This is a model estimate, not a guarantee of biological "
                "function. The model is intended for educational use."
            )
