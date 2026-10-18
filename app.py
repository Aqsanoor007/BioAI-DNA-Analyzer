
import streamlit as st
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline

st.set_page_config(
    page_title="BioAI DNA Analyzer",
    page_icon="🧬",
    layout="wide"
)

st.title("🧬 BioAI — DNA Sequence Analyzer")
st.write("Analyze DNA sequences and explore a simple ML demo.")

dna = st.text_area(
    "Enter DNA sequence",
    placeholder="Example: ATGCGTACGTAG"
)

if st.button("🔬 Analyze Sequence", type="primary"):
    sequence = "".join(dna.upper().split())

    if not sequence:
        st.warning("Please enter a DNA sequence.")
    elif any(base not in "ATGC" for base in sequence):
        st.error("Use only A, T, G and C.")
    else:
        counts = Counter(sequence)
        length = len(sequence)
        gc = (counts["G"] + counts["C"]) / length * 100

        st.subheader("Analysis Results")
        c1, c2, c3 = st.columns(3)
        c1.metric("Sequence Length", f"{length} bp")
        c2.metric("GC Content", f"{gc:.2f}%")
        c3.metric("AT Content", f"{100-gc:.2f}%")

        st.subheader("Nucleotide Composition (%)")
        st.bar_chart({
            base: counts[base] / length * 100
            for base in "ATGC"
        })

        st.subheader("Nucleotide Counts")
        st.write({base: counts[base] for base in "ATGC"})

        st.success("Sequence analyzed successfully!")

st.divider()
st.header("🤖 Machine Learning Demo")
st.caption(
    "Educational demonstration only. The toy examples below are not "
    "a validated biological dataset and cannot make real biological claims."
)

if st.button("Train Demo Model"):
    sequences = [
        "AAAAACCCCC", "AAAACCCCCC", "CCCCCAAAAA",
        "GGGGGTTTTT", "TTTTTGGGGG", "GGGGTTTTTT",
        "ATATATATAT", "TATATATATA", "ATATTTATAT",
        "TATTTATATA", "GCGCGCGCGC", "CGCGCGCGCG"
    ]
    labels = [
        "Group A", "Group A", "Group A",
        "Group B", "Group B", "Group B",
        "Group C", "Group C", "Group C",
        "Group C", "Group D", "Group D"
    ]

    model = make_pipeline(
        CountVectorizer(analyzer="char", ngram_range=(1, 2)),
        MultinomialNB()
    )
    model.fit(sequences, labels)
    st.session_state["demo_model"] = model
    st.success("Demo model trained on toy examples.")

test_seq = st.text_input(
    "Enter a DNA sequence for the demo model",
    key="ml_sequence"
)

if st.button("Classify Demo Sequence"):
    seq = "".join(test_seq.upper().split())
    if not seq or any(base not in "ATGC" for base in seq):
        st.warning("Enter a valid DNA sequence using A, T, G and C.")
    elif "demo_model" not in st.session_state:
        st.warning("Click 'Train Demo Model' first.")
    else:
        prediction = st.session_state["demo_model"].predict([seq])[0]
        st.write(f"Demo model output: **{prediction}**")
        st.info(
            "This output is only a toy-model category, not a promoter "
            "prediction or a validated biological result."
        )
