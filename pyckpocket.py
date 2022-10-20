import streamlit as st

from contexte_streamlit import contexte
from donnees_streamlit import donnees
from dataviz_streamlit import dataviz
from modelisation_streamlit import modelisation
from best_model_streamlit import best_model
from feature_streamlit import feature
from gains_streamlit import gains
from interpretation_streamlit import interpretation
from conclusion_streamlit import conclusion

st.sidebar.title("Paris sportif")

def main():
    
    pages = ["Contexte","Données","Dataviz","Feature Engineering","Recherche du meilleur modèle","Meilleur modèle",
             "Gains selon méthodes de pari", "Conclusion"]
    page = st.sidebar.radio("Aller vers", pages)

    if page == pages[0]:
        contexte()
    elif page == pages[1]:
        donnees()
    elif page == pages[2]:        
        dataviz()
    elif page == pages[3]:
         feature()   
    elif page == pages[4]:
         modelisation()     
    elif page == pages[5]:
         best_model()
    elif page == pages[6]:        
        gains()         
    '''elif page == pages[7]:
                       interpretation() '''   
    else:
        conclusion()

if __name__ == '__main__':
    main()
    
   