# -*- coding: utf-8 -*-
"""
Created on Thu Mar 17 11:09:47 2022

@author: Anna
"""

import streamlit as st
import pandas as pd
import base64
#import matplotlib.pyplot as plt
#import seaborn as sns

from sklearn.preprocessing import StandardScaler

#from sklearn.model_selection import train_test_split, KFold, cross_validate, GridSearchCV
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier#, VotingClassifier, StackingClassifier, BaggingClassifier
from xgboost import XGBClassifier

from sklearn.decomposition import PCA

from sklearn import metrics

def modelisation():
    text = '''
    ---
    '''

   
    original_title = '<p style="font-family:Aral; color:steelblue; font-size: 30px;">Modélisation</p>'
    st.markdown(original_title, unsafe_allow_html=True)
       

    st.markdown(text)  
    
    # Import DataFrame pour modélisation
    df = pd.read_csv(r'df_model.csv', index_col=0)
    
    # La saison lue en tant que numérique est remise en chaine de caractère
    df['Saison'] = df['Saison'].astype(str)
    
    df_complet = df.copy()
    df_complet['Resultat_match'] = df_complet['Resultat_match'].replace(['H','D','A'], [2, 1,0])
    
    # Suppression des variables corrélées
    to_drop = [ 'Moy_cote_domicile', 'Max_cote_domicile','Moy_cote_exterieur','Max_cote_exterieur','Moy_cote_nul','Min_cote_nul','note_att_dom','note_mil_dom','note_def_dom','note_att_ext', 'note_mil_ext', 'note_gen_ext',
  'Moy_carton_R_5_matchs_ext','Moy_nb_fautes_5_matchs_dom', 'Moy_carton_J_5_matchs_dom','Moy_carton_R_5_matchs_dom','Moy_nb_fautes_5_matchs_ext',
   'Moy_corners_5_matchs_ext', 'Moy_carton_J_5_matchs_ext','Res_an_dernier_A','Res_an_dernier_H','Res_an_dernier_D','Pct_buts_par_tirs_cadres_5m_dom',
   'Pct_buts_par_tirs_cadres_5m_ext','Num_match_ext','Num_match_dom','Nb_jours_dernier_match_dom','Nb_jours_dernier_match_ext']
    df.drop(to_drop, axis = 1, inplace = True)
    
    # Création des dataframes de cible et de données
    target = df[['Resultat_match','Saison']]
    data=df.drop(['Resultat_match'],axis=1)

    # Remplacement des modalités de target par des nombres pour la modélisation
    target['Resultat_match'] = target['Resultat_match'].replace(['H','D','A'], [2, 1,0])

    # Création des dataframes d'entraînement et de test : entraînement sur les saisons 2015 à 2020 et test sur 2021 (saison non encore terminée)
    X_train_ = data.loc[data['Saison'] != '2021', :].drop('Saison',axis=1)
    X_test_ = data.loc[data['Saison'] == '2021', :].drop('Saison',axis=1)
    y_train = target.loc[target['Saison'] != '2021', :].drop('Saison',axis=1)
    y_test = target.loc[target['Saison'] == '2021', :].drop('Saison',axis=1)

    # Standardisation des données
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_)
    X_test = scaler.transform(X_test_)

    y_test = y_test.to_numpy().reshape(220,)
    y_train = y_train.to_numpy().reshape(2178,)
    
    
    st.subheader('Pourcentage de matchs par résultat')
    if st.checkbox("Sur l'échantillon d'entraînement"):
        df_y_train=pd.DataFrame(y_train, columns=['Resultat_match'])
        st.dataframe(df_y_train.Resultat_match.value_counts(normalize=True))
        st.markdown('Classe 0 = Défaite à domicile, Classe 1 = Match nul, Classe 2 = Victoire à domicile')
    if st.checkbox("Sur l'échantillon de test"):
        df_y_test=pd.DataFrame(y_test, columns=['Resultat_match'])
        st.dataframe(df_y_test.Resultat_match.value_counts(normalize=True))
        st.markdown('Classe 0 = Défaite à domicile, Classe 1 = Match nul, Classe 2 = Victoire à domicile')
    
    st.markdown("\n")
    st.markdown("\n")
    st.markdown(text)      
    st.subheader('Choix du modèle')
    
    def train_model(model_choisi, X_train, y_train, X_test, y_test) :
        
        if model_choisi == 'SVM' : 
            col1, col2, col3 = st.columns(3)
            C = col1.selectbox('C', [0.1, 1, 10])
            kernel = col2.selectbox('kernel', ['rbf','linear', 'poly'])
            gamma = col3.selectbox('gamma', [0.001, 0.1, 0.5])
            model = svm.SVC(C = C, kernel = kernel, gamma = gamma)
            
        elif model_choisi == 'Random Forest' : 
            col1, col2 = st.columns(2)
            max_features = col1.selectbox('max_features', ["sqrt", "log2" ])
            min_samples_split = col2.slider('min_samples_split', 2, 60)
            model = RandomForestClassifier(max_features = max_features, min_samples_split = min_samples_split)
            
        elif model_choisi == 'KNN' : 
            col1, col2, col3, col4 = st.columns(4)
            n_neighbors = col1.slider('n_neighbors' , 5, 60)
            weights = col2.selectbox('weights' , ['uniform', 'distance'],)
            algorithm = col3.selectbox('algorithm' , ['auto', 'ball_tree', 'kd_tree', 'brute'])
            metric = col4.selectbox('metric' , ['minkowski', 'manhattan', 'chebyshev', 'euclidean'])
            model = KNeighborsClassifier(n_neighbors = n_neighbors, weights = weights, algorithm = algorithm, metric = metric)
            
        elif model_choisi == 'XGBoost':
            col1, col2 = st.columns(2)
            booster = col1.selectbox('booster' , ['gbtree', 'gblinear', 'dart'])
            verbosity = col2.selectbox('verbosity' , [0,1,2,3],)
            model = XGBClassifier(booster = booster, verbosity = verbosity)
            
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        score = metrics.accuracy_score(y_test, y_pred)
        confusion_matrix = pd.crosstab(y_test, y_pred, rownames = ['Classe réelle'], colnames = ['Classe prédite par SVM'])
        return st.write('Accuracy : ',round(score,3)), st.write('Matrice de confusion', confusion_matrix)
    
    def choix_pca(select_pca):
        if select_pca == 'Avec PCA':
            pca = PCA(n_components = 0.9)
            X_train_pca = pca.fit_transform(X_train)
            X_test_pca = pca.transform(X_test)
            if model_choisi == 'SVM' : 
                model = svm.SVC()
            elif model_choisi == 'Random Forest' : 
                model = RandomForestClassifier()
            elif model_choisi == 'KNN' : 
                model = KNeighborsClassifier()
            elif model_choisi == 'XGBoost':
                model = XGBClassifier()
            model.fit(X_train_pca, y_train) 
            return st.write('Accuracy avec PCA :', round(model.score(X_test_pca, y_test),3)), st.write('Nombre de composantes gardées :', pca.n_components_)
       
            
    
    model_choisi = st.selectbox(label = "Choisissez votre modèle" , options = ['SVM', 'Random Forest', 'KNN', 'XGBoost'])
    
    st.title(model_choisi)
    st.write('Choississez les paramètres :')
    train_model(model_choisi, X_train, y_train, X_test, y_test)
    st.markdown(text)  
    st.subheader('Avec ou sans PCA')
    select_pca = st.selectbox('Choisissez', ['Sans PCA', 'Avec PCA'])
    choix_pca(select_pca)    
    
    
   
