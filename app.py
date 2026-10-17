
import streamlit as st
import matplotlib.pyplot as plt
from collections import Counter

st.set_page_config(
    page_title="BioAI DNA Analyzer",
    page_icon="🧬",
    layout="wide"
)

st.title("🧬 BioAI - DNA Sequence Analyzer")
st.write(
    "Analyze DNA sequences and explore "
    "their biological properties."
)

st.markdown("---")

dna = st.text_area(
    "Enter your DNA sequence",
    placeholder="Example: ATGCGTACGTAG",
    height=120
)

if st.button("🔬 Analyze Sequence", type="primary"):
    sequence = "".join(dna.upper().split())

    if not sequence:
        st.warning("Please enter a DNA sequence.")

    elif any(base not in "ATGC" for base in sequence):
        st.error("Invalid sequence! Use only A, T, G and C.")

    else:
        counts = Counter(sequence)
        length = len(sequence)
        gc = (counts["G"] + counts["C"]) / length * 100

        st.subheader("📊 Analysis Results")

        col1, col2, col3 = st.columns(3)

        col1.metric("Sequence Length", f"{length} bp")
        col2.metric("GC Content", f"{gc:.2f}%")
        col3.metric("AT Content", f"{100-gc:.2f}%")

        st.subheader("Nucleotide Composition")

        chart_data = {
            base: counts[base] / length * 100
            for base in "ATGC"
        }

        st.bar_chart(chart_data)

        st.subheader("Nucleotide Counts")

        st.write({
            base: counts[base]
            for base in "ATGC"
        })

        st.success("DNA sequence analyzed successfully!")

        st.info(
            "This is an educational bioinformatics tool. "
            "AI/ML features can be added in a later stage."
        )
