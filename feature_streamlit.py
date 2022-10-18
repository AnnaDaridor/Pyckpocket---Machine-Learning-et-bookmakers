# -*- coding: utf-8 -*-
"""
Created on Tue Mar 22 13:48:10 2022

@author: Christophe
"""

import streamlit as st
import pandas as pd
import webbrowser

def feature():
    
    
    text = '''
    ---
    '''
    original_title = '<p style="font-family:Aral; color:steelblue; font-size: 30px;">Feature Engineering</p>'
    st.markdown(original_title, unsafe_allow_html=True)    
    st.markdown(text)  
    
    df = pd.read_csv('df_model.csv')
    
    st.write('Etape clé du projet, avant d’essayer de prédire l’issue des matchs, nous avons commencé par définir nos problématiques, il s’agissait de sélectionner et de collecter les données, les préparer, nettoyer, formater et agréger certaines métriques pour pouvoir être utilisées. Avant cette étape, nos données étaient dans un format ne permettant pas une modélisation efficace.')
    st.markdown(text)
    
    st.subheader('Première étape')

    
    st.write('Dans l’ordre, nous avons procédé aux manipulations et transformations suivantes :' )
    st.write(' •	Gestion des dates au format « date »,') 
    st.write(' •	Passage de la saison en string, ')
    st.write(' •	Ajout du résultat du même match lors de la saison précédente ainsi que les buts lors des rencontres.')

    
    
    
    
    st.markdown(text)
    
    st.subheader('Deuxième étape')
    st.info('Il nous semblait nécessaire de calculer et mettre en évidence certaines variables afin de calculer l’« état de forme » des équipes pour nos prédictions futures :')
    
    st.write(' •	Ajout du nombre de points moyens par match depuis le début de l’année, correspondant à un classement des équipes sur la saison,' )
    st.write(' •	Calcul de différentes moyennes sur les 5 derniers matchs,') 
    st.write(' •	Buts, tirs, tirs cadrés et rapport en les buts / tirs cadrés, ')
    st.write(' •	Nombre de fautes ou encore la moyenne de corners, cartons jaunes ou rouges,')  
    st.write(' •	Nombre de jours entre 2 matchs.')  
   
    st.warning('Nous avons également calculé « l’état de forme » sur les 3 derniers matchs, mais cela donnait de moins bons résultats dans les modèles.')


    st.markdown(text)
    
    
    st.write('Afin d’être le plus pertinent dans nos calculs, nous avons rajouté les notes tirées du jeu Fifa par équipe et par saisons. Ces notes sont des notes sur 100 qui jaugent le niveau de la défense, de l’attaque, du milieu et une note générale.')
  
    
    url = 'https://www.fifaindex.com/fr/teams/?league=16&order=desc'
    if st.button('📎 Fifa'):
        webbrowser.open_new_tab(url)    
    
  
    st.write('Une fois ces manipulations effectuées, nous avons fusionné les différents dataframes générés au cours du parcours afin d’obtenir un dataframe final duquel nous avons supprimé les données non pertinentes pour l’étape suivante, la modélisation.')
  
    st.markdown(text)  
    
    original_title = '<p style="font-family:Aral; color:steelblue; font-size: 30px;">Dictionnaire de données pour la modélisation</p>'
    st.markdown(original_title, unsafe_allow_html=True)
    st.markdown("\n")
    st.write("""Ces données sont celles obtenues après le feature engineering :""")
    # Inject CSS with Markdown
    # CSS to inject contained in a string
    hide_dataframe_row_index = """
                <style>
                .row_heading.level0 {display:none}
                .blank {display:none}
                </style>
                """    
    st.markdown(hide_dataframe_row_index, unsafe_allow_html=True)
    dd_df = pd.read_excel (r'pyckpocket_Dictionnaire de données.xlsx', sheet_name='DD du dataframe', keep_default_na=False)
    # Display an interactive table
    st.dataframe(dd_df)
    st.markdown("\n")

    st.markdown(text)    
    original_title2 = '<p style="font-family:Aral; color:steelblue; font-size: 30px;">Dataframe finale pour la modélisation</p>'
    st.markdown(original_title2, unsafe_allow_html=True)
    
    st.dataframe(df)

