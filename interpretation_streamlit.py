# -*- coding: utf-8 -*-

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import shap
from PIL import Image
import base64

# Import DataFrame pour modélisation
#df = pd.read_csv(r"C:\Users\Isabelle\Documents\Streamlit\df_model.csv", index_col=0)
df = pd.read_csv(r"df_model.csv", index_col=0)   

# La saison lue en tant que numérique est remise en chaine de caractère
df['Saison'] = df['Saison'].astype(str)

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
# Création des dataframe d'entrainement et de test : entrainement sur les saisons 2015 à 2020 et test sur 2021 (saison non encore terminée)
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
 
X_test_df=pd.DataFrame(X_test,columns=X_train_.columns)

# Reprise des valeurs de résultat de SHAP pour gain de temps
knn_explainer_value = 1.3772727272727272
from numpy import load
knn_shap_values = load('knn_shap_values.npy')

def interpretation():
    text = '''
    ---
    '''        

    col1, mid, col2 = st.columns([10,1,3])
    with col1:
        original_title = '<p style="font-family:Aral; color:steelblue; font-size: 30px;">Interprétation des modèles avec SHAP</p>'
        st.markdown(original_title, unsafe_allow_html=True)
    with col2:
 
        
        st.markdown(
            f'<img src="data:image/gif;base64,{data_url}" alt="cat gif" width="200" height="200">',
            unsafe_allow_html=True,
        ) 
    
    st.markdown(text)  
    
    st.subheader('Feature importance')
    fig1 = plt.figure(figsize=(12,8))
    shap.summary_plot(knn_shap_values, X_test_df)
    st.write(fig1)
    
    st.info("""Les variables sont triées par ordre décroissant d'importance. Plus les valeurs de la note défensive de l'équipe extérieure sont élevées (rouge), plus le résultat attendu est 0 (victoire de l'équipe extérieure) car c'est du côté négatif de l'axe que nous les trouvons. 
             Au contraire, on retrouve des points rouges sur le côté positif de l'axe pour la note générale de l'équipe à domicile. Les 5 premières variables séparent bien les faibles et hautes valeurs, avec un effet négatif (plus de chance de perdre pour l'équipe à domicile) pour la note de défense de l'équipe extérieur et le minimum de la cote domicile et avec un effet positif (plus de chance de gagner pour l'équipe à domicile) pour les variables note générale à domicile, moyenne de buts sur 5 matchs à domicile et minimum de la cote de l'équipe extérieure. 
             Les 2 variables restantes séparent moins bien les hautes et faibles valeurs sur l'axe même si on devine un effet négatif pour les points moyens par match sur la saison pour l'équipe extérieure.""")
    
 
    st.markdown(text)      
    st.subheader('Dependence plots') 
    
    image = Image.open("shap_dependance_1.png")
    st.image(image,width=500)
    st.info("""Nous observons une tendance linéaire négative marquée entre la note défensive de l'équipe extérieur et le résultat. 
             Le minimum de la cote de l'équipe à l'extérieur baisse quand la note défensive augmente.""")   
    st.markdown("\n")

    image = Image.open("shap_dependance_2.png")

    st.markdown("\n") 
    
    st.image(image,width=500)
    st.info("""Nous remarquons une tendance linéaire positive entre la note générale de l'équipe à domicile et le résultat. 
             Il n'y a pas d'interaction probante entre cette variable et la moyenne des buts sur 5 matchs.""")   
   

    st.markdown(text)      
    st.subheader('Explication des résultats sur un match précis')
    exemple = '<p style="font-family:Aral; color:seagreen; font-size: 15px;">Exemple sur 2 matchs :</p>'
    st.markdown(exemple, unsafe_allow_html=True)
    
    st.info("""Pour le match **Rennes-Lens** de la saison 2021, la prévision est une victoire à domicile (le véritable résultat a été match nul). 
            Les variables qui entrainent cette prédiction haute sont les variables en rouge, en particulier la moyenne de buts sur 5 matchs à domicile, alors que la note défensive de l'équipe adverse à l'effet de faire diminuer cette probabilité.""")   
    
    
    fig3=shap.force_plot(knn_explainer_value,knn_shap_values[3,:], X_test_df.iloc[3,:].round(3), matplotlib=True,show=False)
    st.write(fig3)
    
  
    
    st.info("""Pour le match **Montpellier - Strasbourg**, la prévision de victoire à l'extérieur de Strasbourg est portée par toutes les variables sauf la cote à domicile qui était en faveur d'une victoire de Montpellier.
            Le résultat du match a été un match nul (mauvaise prévision des bookmakers et de notre modèle).""")    
            
    fig4=shap.force_plot(knn_explainer_value,knn_shap_values[80,:], X_test_df.iloc[80,:].round(3), matplotlib=True,show=False)
    st.write(fig4)

    
  

    '''st.markdown(text)      
                interactif = '<p style="font-family:Aral; color:seagreen; font-size: 15px;">Partie interactive où vous pouvez choisir le match :</p>'
                st.markdown(interactif, unsafe_allow_html=True)    
            
                options =  X_test_.index
                num_match_test = st.selectbox("Choisir le match dont vous voulez interpréter le résultat du modèle :", range(len(options)), format_func=lambda x: options[x])
                
                #st.write("index:", num_match_test)
                
                fig5=shap.force_plot(knn_explainer_value,knn_shap_values[num_match_test,:], X_test_df.iloc[num_match_test,:].round(3), matplotlib=True,show=False)
                st.write(fig5)
                
                vrai_result=y_test[num_match_test]
                
                if vrai_result==0:
                    vrai_result_label="Victoire extérieur"
                elif vrai_result==1:
                    vrai_result_label="Match nul"
                else:
                    vrai_result_label="Victoire domicile"
                    
                st.write("Le résultat réél du match est :",vrai_result_label)'''



