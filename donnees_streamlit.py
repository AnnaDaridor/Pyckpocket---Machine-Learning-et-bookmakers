# -*- coding: utf-8 -*-


import streamlit as st
import pandas as pd


def donnees():
    text = '''
    ---
    '''        
               
    original_title = '<p style="font-family:Aral; color:steelblue; font-size: 30px;">Dictionnaire de données brutes</p>' 
    st.markdown(original_title, unsafe_allow_html=True)
    st.markdown(text)
    st.info("""Les données proviennent du site football-data.co.uk https://www.football-data.co.uk/downloadm.php.""")       
    st.markdown("\n")      
    st.write("""La liste des données varie selon les saisons.
             Le tableau suivant récapitule les données présentes dans les fichiers de 2015 à 2021.""")
    st.markdown("\n")     
        
    df = pd.read_excel (r'pyckpocket_Dictionnaire de données.xlsx', sheet_name='DD fichiers importés', keep_default_na=False)
    st.write(df)
    
    st.markdown("\n")     
    st.info("""Notre jeu de données de départ contient 7 fichiers au format CSV (un par saison, de 2015 à 2021) ayant chacun 380 lignes, sauf en 2021 
             car la saison est encore en cours (220 matchs joués) et la saison 2019-2020 qui a été stoppée à cause du COVID (279 matchs joués).
             De plus, le match Bastia/Lyon du 16/04/2017 a été supprimé car interrompu suite à un incident, nous n'avions donc pas toutes les données de ce match.""")
    st.markdown("\n")  
  