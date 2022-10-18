import streamlit as st

def demo_ml_app():

    st.header("Prédire les survivants du Titanic")
    st.subheader("by datascientest")

    df = pd.read_csv('df_model.csv')

    if st.checkbox("Afficher les données"):
        st.dataframe(df)