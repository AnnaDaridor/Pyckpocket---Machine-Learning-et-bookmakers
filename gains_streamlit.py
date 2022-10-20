# -*- coding: utf-8 -*-
"""
Created on Tue Mar 22 13:48:10 2022

@author: Christophe
"""

import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
from sklearn.model_selection import  KFold, cross_validate
from sklearn.neighbors import KNeighborsClassifier
from PIL import Image 
import plotly.express as px
import base64
    
# Import DataFrame pour modélisation
df = pd.read_csv(r'df_model.csv', index_col=0)

# La saison lue en tant que numérique est remise en chaine de caractère
df['Saison'] = df['Saison'].astype(str)
df_tot=df.copy()

# Variables sélectionnées pour la modélisation
xvars = ['Saison','Min_cote_domicile','Min_cote_exterieur','note_def_ext','note_gen_dom','Points_moy_match_saison_dom','Moy_buts_5_matchs_dom',
           'Points_moy_match_saison_ext']

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
print(X_train_.shape)
print(X_test_.shape)
print(y_train.shape)
print(y_test.shape)

# Standardisation des données
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train_)
X_test = scaler.transform(X_test_)

y_test=y_test.to_numpy().reshape(220,)
y_train=y_train.to_numpy().reshape(2178,)

# On reprend notre modèle KNN
knn = KNeighborsClassifier(algorithm='auto', metric='minkowski', n_neighbors= 55, weights= 'uniform')
cv = KFold(n_splits=3, random_state=22 , shuffle = True)

scores = cross_validate(knn, X_train, y_train, cv=cv, scoring=['accuracy','f1_micro', 'f1_macro', 'f1_weighted'])
knn.fit(X_train, y_train)

y_pred_knn = knn.predict(X_test)
confusion_matrix=pd.crosstab(y_test, y_pred_knn, rownames = ['Classe réelle'], colnames = ['Classe prédite par random forest'])
sns.heatmap(confusion_matrix, annot=True)
print('Accuracy: ',metrics.accuracy_score(y_test, y_pred_knn));
#print(classification_report_imbalanced(y_test, y_pred_knn)) 

# On prend aussi les prédictions sur le jeu d'entraînement
y_pred_knn_train = knn.predict(X_train)
confusion_matrix=pd.crosstab(y_train, y_pred_knn_train, rownames = ['Classe réelle'], colnames = ['Classe prédite par random forest'])
sns.heatmap(confusion_matrix, annot=True)
print('Accuracy: ',metrics.accuracy_score(y_train, y_pred_knn_train));

y_pred_knn_proba = knn.predict_proba(X_test)
y_pred_knn_proba_train = knn.predict_proba(X_train)
# Proba pour chaque résultat
y_pred_knn_proba = pd.DataFrame(knn.predict_proba(X_test))
y_pred_knn_proba_train = pd.DataFrame(knn.predict_proba(X_train))

y_proba = pd.concat([y_pred_knn_proba_train, y_pred_knn_proba])
y_proba.columns = ['Proba_0','Proba_1','Proba_2']
y_proba.reset_index(inplace=True,drop=True)

# On récupère nos prédictions dans un dataframe
y_pred_knn_df= pd.DataFrame(y_pred_knn)
y_pred_knn_train_df = pd.DataFrame(y_pred_knn_train)

y = pd.concat([y_pred_knn_train_df, y_pred_knn_df])
y.columns = ['Prediction']
y.reset_index(inplace=True,drop=True)







def gains():
    text = '''
    ---
    '''

    col1, mid, col2 = st.columns([10,0.1,7])
    with col1:
        original_title = '<p style="font-family:Aral; color:steelblue; font-size: 30px;">Simulation des gains selon notre meilleur modèle</p>' 
        st.markdown(original_title, unsafe_allow_html=True)
    with col2:
 
        
        st.markdown(
            f'<img src="data:image/gif;base64,{data_url}" alt="cat gif" width="400" height="200">',
            unsafe_allow_html=True,
        )  
    

    st.markdown(text)    
  
   
    st.write('Dans la partie visualisation, nous avons pu constater que se contenter de suivre les cotes des bookmakers pour parier donne des résultats très aléatoires (cf. « Gain en suivant les cotes des bookmakers »). Nous allons donc utiliser les résultats de notre modèle pour essayer d’améliorer ces gains sur la saison de test (2021-2022 première partie de saison) en suivant 2 méthodes.')
    st.markdown(text)
    
    st.subheader('Pari sur tous les matchs')
    st.info('En suivant les résultats de la modélisation et en misant 10 € sur tous les matchs, nous obtenons un gain d’environ 200 € sur les 220 premiers matchs de la saison (à comparer à la perte de 220 € si on suivait les cotes des bookmakers)')
    
    img = Image.open("Image1.png") 
      
    st.image(img, width=700) 
    
    st.markdown(text)
    
    
    
    
    
    
    
    st.subheader('Pari sur une sélection ciblée de matchs')
    st.info('Nous avons ensuite souhaité aborder une nouvelle approche en essayant d’élaborer une stratégie de pari comme suit :')
    st.write('10 euros si la prédiction de notre modèle est supérieure à la probabilité bookmakers ainsi que du seuil de pari de 40%')
    st.write('15 euros si la prédiction de notre modèle est supérieure de 10% à la probabilité bookmakers ainsi que du seuil de pari')
    st.write('20 euros si la prédiction de notre modèle est supérieure de 20% à la probabilité bookmakers ainsi que du seuil de pari')
    
    
    
    # On récupère nos données de départ
    df_complet = df_tot.copy()
    df_complet['Resultat_match'] = df_complet['Resultat_match'].replace(['H','D','A'], [2, 1,0])
    df_complet.reset_index(inplace=True,drop=True)
    # Ajout des prédictions faites à partir du meilleur KNN
    df_avec_pred = pd.concat([df_complet, y,y_proba],axis=1)
    
    df_avec_pred['Prob_bk_0'] = (1/df_avec_pred['Max_cote_exterieur']/(1/df_avec_pred['Max_cote_domicile']+1/df_avec_pred['Max_cote_exterieur']+1/df_avec_pred['Max_cote_nul'])).round(3)
    df_avec_pred['Prob_bk_1'] = (1/df_avec_pred['Max_cote_nul']/(1/df_avec_pred['Max_cote_domicile']+1/df_avec_pred['Max_cote_exterieur']+1/df_avec_pred['Max_cote_nul'])).round(3)
    df_avec_pred['Prob_bk_2'] = (1/df_avec_pred['Max_cote_domicile']/(1/df_avec_pred['Max_cote_domicile']+1/df_avec_pred['Max_cote_exterieur']+1/df_avec_pred['Max_cote_nul'])).round(3)
    
    #pari si taux de probabilité > 40%,
    
    taux_proba_ = 0.4
    pari1_ = 10
    pari2_ = 15
    pari3_ = 20  
    
    # Montant à parier en fonction du resultat 
    def calcul_bet(taux_proba,pari1,pari2,pari3,affiche):
        df_avec_pred['bet_0'] = np.where((df_avec_pred['Proba_0']>df_avec_pred['Prob_bk_0']) & (df_avec_pred['Proba_0']>taux_proba), pari1 , 0)
        df_avec_pred['bet_0'] = np.where((df_avec_pred['Proba_0']-df_avec_pred['Prob_bk_0']>0.10) & (df_avec_pred['Proba_0']>taux_proba), pari2 , df_avec_pred['bet_0'])
        df_avec_pred['bet_0'] = np.where((df_avec_pred['Proba_0']-df_avec_pred['Prob_bk_0']>0.20) & (df_avec_pred['Proba_0']>taux_proba), pari3 , df_avec_pred['bet_0'])
        
        df_avec_pred['bet_1'] = np.where((df_avec_pred['Proba_1']>df_avec_pred['Prob_bk_1']) & (df_avec_pred['Proba_1']>taux_proba), pari1 , 0)
        df_avec_pred['bet_1'] = np.where((df_avec_pred['Proba_1']-df_avec_pred['Prob_bk_1']>0.10) & (df_avec_pred['Proba_1']>taux_proba), pari2 , df_avec_pred['bet_1'])
        df_avec_pred['bet_1'] = np.where((df_avec_pred['Proba_1']-df_avec_pred['Prob_bk_1']>0.20) & (df_avec_pred['Proba_1']>taux_proba), pari3 , df_avec_pred['bet_1'])
        
        df_avec_pred['bet_2'] = np.where((df_avec_pred['Proba_2']>df_avec_pred['Prob_bk_2']) & (df_avec_pred['Proba_2']>taux_proba), pari1 , 0)
        df_avec_pred['bet_2'] = np.where((df_avec_pred['Proba_2']-df_avec_pred['Prob_bk_2']>0.10) & (df_avec_pred['Proba_2']>taux_proba), pari2 , df_avec_pred['bet_2'])
        df_avec_pred['bet_2'] = np.where((df_avec_pred['Proba_2']-df_avec_pred['Prob_bk_2']>0.20) & (df_avec_pred['Proba_2']>taux_proba), pari3 , df_avec_pred['bet_2'])
    
   
        st.info('Donne les prévisions de gains suivantes :')
        
        # je reimporte les matchs
        
        df3 = pd.read_csv('df_model.csv')
        
        
        df_gains = df_avec_pred
        df_gains['BET'] = df_gains['bet_0'] + df_gains['bet_1'] + df_gains['bet_2']
        df_gains['gain_potentiel'] = df_gains['bet_0'] * df_gains['Max_cote_exterieur'] + df_gains['bet_1'] * df_gains['Max_cote_nul'] + df_gains['bet_2'] * df_gains['Max_cote_domicile'] 
        
        
        df_gains['match'] = df3['Unnamed: 0']
        debut_saison_test=2178
        df_gains=df_gains.iloc[debut_saison_test:]
        
        
        a = df_gains['Resultat_match']
        b = df_gains['Prediction']
        df_gains['resultknn'] = a == b
        df_gains['gains'] = np.where( a== b,df_gains['gain_potentiel'],-df_gains['BET'] )
        df_gains['pertes'] = np.where( a != b, -df_gains['BET'], np.nan)
        df_gains['gain'] = np.where( a== b,df_gains['gain_potentiel'],np.nan)
    
        if affiche=='O':         
            df_gains2 = df_gains[['Resultat_match', 'bet_0',	'bet_1',	'bet_2',	'BET',	'gain_potentiel',	'match',	'resultknn','pertes',	'gain']]
            
            st.write(df_gains2)
            
            st.info('Sur la saison 2021-2022')
            
            debut_saison_test=2178
            # nombre de paris pour chaque résultats
            bet0=df_avec_pred.loc[:,['bet_0']].iloc[debut_saison_test:].value_counts()
            bet1=df_avec_pred.loc[:,['bet_1']].iloc[debut_saison_test:].value_counts()
            bet2=df_avec_pred.loc[:,['bet_2']].iloc[debut_saison_test:].value_counts()
            print(bet0,bet1,bet2)
            # nombre de paris  = pour chaque montant de paris
            nb_pari1= len(df_avec_pred[(df_avec_pred.index>=debut_saison_test) & (df_avec_pred['bet_0'] == pari1)]) + \
            len(df_avec_pred[(df_avec_pred.index>=debut_saison_test) & (df_avec_pred['bet_1'] == pari1)])+ \
            len(df_avec_pred[(df_avec_pred.index>=debut_saison_test) & (df_avec_pred['bet_2'] == pari1)])
            print("Le nombre de paris à", pari1,"euros est de",nb_pari1)
            st.write("Le nombre de paris à", pari1,"euros est de",nb_pari1)
            nb_pari2= len(df_avec_pred[(df_avec_pred.index>=debut_saison_test) & (df_avec_pred['bet_0'] == pari2)]) + \
            len(df_avec_pred[(df_avec_pred.index>=debut_saison_test) & (df_avec_pred['bet_1'] == pari2)])+ \
            len(df_avec_pred[(df_avec_pred.index>=debut_saison_test) & (df_avec_pred['bet_2'] == pari2)])
            print("Le nombre de paris à", pari2,"euros est de",nb_pari2)
            st.write("Le nombre de paris à", pari2,"euros est de",nb_pari2)
            nb_pari3= len(df_avec_pred[(df_avec_pred.index>=debut_saison_test) & (df_avec_pred['bet_0'] == pari3)]) + \
            len(df_avec_pred[(df_avec_pred.index>=debut_saison_test) & (df_avec_pred['bet_1'] == pari3)])+ \
            len(df_avec_pred[(df_avec_pred.index>=debut_saison_test) & (df_avec_pred['bet_2'] == pari3)])
            print("Le nombre de paris à", pari3,"euros est de",nb_pari3)
            st.write("Le nombre de paris à", pari3,"euros est de",nb_pari3)
    
           
    calcul_bet(taux_proba=taux_proba_,pari1=pari1_,pari2=pari2_,pari3=pari3_,affiche='O')    

    #Montant parié par match et résultat

    dic_resultat = {2 : 'Max_cote_domicile', 1: 'Max_cote_nul', 0 : 'Max_cote_exterieur'}    
    def gain_paris(sortie):

        st.subheader('Resultat')    

        matchs = []
        #2178
        for i in range(2178,2398):
            matchs.extend([i,i,i])
        
        #Liste des résultats
        resultats = []
        for i in range(2178,2398):
            resultats.extend([2,1,0])  
            
        montants = []
        for i in range(2178,2398):
            montants.extend(df_avec_pred.loc[i][['bet_2', 'bet_1', 'bet_0']])
        
        # Liste des matchs
        
        benefices = 0
        for match, resultat, montant in zip(matchs, resultats, montants):
    
            #itération simultanées des éléments des trois listes 
            if df_avec_pred.loc[match]['Resultat_match'] == resultat:
    
                #bénéfice du pari si le résultat prédit est bon 
                benefice = (df_avec_pred.loc[match][dic_resultat[resultat]] - 1)*montant
    
            else : benefice = -1*montant
            #perte du montant parié si le résultat prédit est faux
    
            benefices += benefice
            #on agrège le bénéfice du pari aux bénéfices totaux
            #print(match,benefices)
    
        pourcentage = benefices*100/sum(montants)
        if sortie=='O':
        #calcul du pourcentage gagné/perdu totaux
            st.write('Montant parié :', sum(montants),'€')
            st.write('Gains/pertes :', benefices.round(2),'€')
            st.write('Rendement du modèle :', pourcentage.round(2), ' %')

        return print('Montant parié :', sum(montants)), print('Gains/pertes :', benefices), print('Rendement du modèle :', pourcentage.round(2), ' %')
    
    gain_knn  = gain_paris(sortie='N')
    
    
    # je reimporte les matchs
    
    df3 = pd.read_csv('df_model.csv')
    df_gains = df_avec_pred
    df_gains['BET'] = df_gains['bet_0'] + df_gains['bet_1'] + df_gains['bet_2']
    df_gains['gain_potentiel'] = df_gains['bet_0'] * df_gains['Max_cote_exterieur'] + df_gains['bet_1'] * df_gains['Max_cote_nul'] + df_gains['bet_2'] * df_gains['Max_cote_domicile'] 
    
    df_gains['match'] = df3['Unnamed: 0']
    debut_saison_test=2178
    df_gains=df_gains.iloc[debut_saison_test:]
    
    a = df_gains['Resultat_match']
    b = df_gains['Prediction']
    df_gains['resultknn'] = a == b
    df_gains['gains'] = np.where( a== b,df_gains['gain_potentiel'],-df_gains['BET'] )
    df_gains['pertes'] = np.where( a != b, -df_gains['BET'], 0)
    df_gains['gain'] = np.where( a== b,df_gains['gain_potentiel'],0)
    
    
    st.write('Gains potentiels : 2 169.25 €')
    st.write('Montant parié : 950 €')
    st.write('Gains cumulés : 1 188.65 €')
    st.write('Bénéfice : 238.65 €')
    st.write('Rendement : 25.12 %')
    
    
    st.markdown(text)
    
    
    text = '''
    ---
    '''
    

    
    st.info('Répartition des gains selon paris')
    
    
    fig2 = px.bar(df_gains, x="Prediction", y="gain", color="BET", color_continuous_scale=px.colors.sequential.Blues)
    fig2.update_layout(
        xaxis = dict(
            tickvals = [0, 1, 2],
            ticktext = [0, 1, 2]
        ))
    
    st.plotly_chart(fig2, use_container_width=True)
    
    
    st.write('On remarque que la majorité de nos gains porte sur les victoires à domicile. Cependant en proportion de gains, les revenus sont meilleurs sur les matchs nuls.') 
    
    st.markdown(text)
    
    
    text = '''
    ---
    '''
    
    
    st.info('Répartition des pertes')
    
    
    
    fig = px.bar(df_gains, x="Prediction", y="pertes", color="BET", color_continuous_scale=px.colors.sequential.Blues)
    fig.update_layout(
        xaxis = dict(
            tickvals = [0, 1, 2],
            ticktext = [0, 1, 2]
        ))
    
    st.plotly_chart(fig, use_container_width=True)
    
    
    
    
    #t.error("On s'aperçoit en analysant nos pertes :"  )
    #t.write("Que nous perdons souvent sur les prédicitions de Girondins de Bordeaux. En effet, de par ses précédents résultats, l'équipe est souvent donnée victorieuse par notre modèle alors qu'elle réalise une saison catastrophique.")
    #t.write("A contrario, nous perdons sur des prédictions comme Rennes ou Lens qui réalisent une bonne saison !")
    
    st.markdown(text)
    
    
    text = '''
    ---
    '''
    
    st.warning('Avec cette dernière méthode, nous observons que les gains sont nettement meilleurs qu’en suivant les côtes bookmakers et meilleurs qu’en pariant sur tous les matchs. Cependant cela reste encore très théorique car nous n’examinons pas, à ce niveau, un certain nombre de variables qui peuvent influer de manière importante sur l’issue d’un match.')
    
    st.markdown(text)
    st.markdown("\n") 
    st.subheader('Choisissez vos paramètres de paris :')
    st.markdown("\n") 
    
    col1, col2, col3, col4 = st.columns(4)
    taux_proba_c = float(col1.slider("Taux de proba à partir duquel on parie", 0.2,1.0,value =0.4))
    pari1_c = float(col2.text_input("Montant du pari si proba > proba bookmaker", 10))
    pari2_c = float(col3.text_input("Montant du pari si proba > proba bookmaker * 1.1", 15))
    pari3_c = float(col4.text_input("Montant du pari si proba > proba bookmaker * 1.2", 20))
    
    calcul_bet(taux_proba=taux_proba_c,pari1=pari1_c,pari2=pari2_c,pari3=pari3_c,affiche='N')
    gain_choix  = gain_paris(sortie='O')
    

