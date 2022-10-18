# -*- coding: utf-8 -*-

import streamlit as st
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier

from sklearn import metrics

def best_model():
    
    text = '''
    ---
    '''    
    original_title = '<p style="font-family:Aral; color:steelblue; font-size: 30px;">Meilleur modèle</p>'
    st.markdown(original_title, unsafe_allow_html=True)
    st.markdown(text)      
    st.subheader('Avec notre sélection de variables manuelle')
   
    # Import DataFrame pour modélisation
    df = pd.read_csv(r'df_model.csv', index_col=0)
    # Variables sélectionnées pour la modélisation
    xvars = ['Saison','Min_cote_domicile','Min_cote_exterieur','note_def_ext','note_gen_dom','Points_moy_match_saison_dom','Moy_buts_5_matchs_dom',
       'Points_moy_match_saison_ext']
    
    # La saison lue en tant que numérique est remise en chaine de caractère
    df['Saison'] = df['Saison'].astype(str)

    df = df.loc[:, xvars + ["Resultat_match"]]
    df['Resultat_match'] = df['Resultat_match'].replace(['H','D','A'], [2, 1,0])
    # Création des dataframes de cible et de données
    target = df[['Resultat_match','Saison']]
    data = df.loc[:, xvars]

    # Remplacement des modalités de target par des nombres pour la modélisation
    target['Resultat_match'] = target['Resultat_match'].replace(['H','D','A'], [2, 1,0])

    # Création des dataframe d'entrainenemnt et de test : entrainement sur les saison 2015 à 2020 et test sur 2021 (saison non encore terminée)
    X_train_=data.loc[data['Saison'] != '2021', :].drop('Saison',axis=1)
    X_test_=data.loc[data['Saison'] == '2021', :].drop('Saison',axis=1)
    y_train=target.loc[target['Saison'] != '2021', :].drop('Saison',axis=1)
    y_test=target.loc[target['Saison'] == '2021', :].drop('Saison',axis=1)

    # Standardisation des données
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_)
    X_test = scaler.transform(X_test_)

    y_test=y_test.to_numpy().reshape(220,)
    y_train=y_train.to_numpy().reshape(2178,)
    
    # Meilleur modèle
    knn = KNeighborsClassifier(algorithm='auto', metric='minkowski', n_neighbors= 55, weights= 'uniform')
    knn.fit(X_train, y_train)
    y_pred_knn = knn.predict(X_test)
    confusion_matrix=pd.crosstab(y_test, y_pred_knn, rownames = ['Classe réelle'], colnames = ['Classe prédite par random forest'])
    score = metrics.accuracy_score(y_test, y_pred_knn)
    
    st.write('Les variables sélectionnées sont :')
    st.info('Min_cote_domicile, Min_cote_exterieur, note_def_ext, note_gen_dom, \n Points_moy_match_saison_dom, Moy_buts_5_matchs_dom, Points_moy_match_saison_ext')
    st.write('Le meilleur modèle est KNN avec les paramètres :')
    st.info('algorithm = auto, metric = minkowski , n_neighbors = 55, weights = uniform')
    st.write('Accuracy : ',round(score,3)), st.write('Matrice de confusion', confusion_matrix)
