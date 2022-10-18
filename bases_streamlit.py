import streamlit as st

def streamlit_base():
    st.title("Streamlit crash course")

    st.header("Titre secondaire")

    result = st.button("press me")
    if result:
        st.text("vrai")