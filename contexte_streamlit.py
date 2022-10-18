# -*- coding: utf-8 -*-

import streamlit as st
import webbrowser

def contexte():
    text = '''
    ---
    '''    
    
    original_title = '<p style="font-family:Aral; color:steelblue; font-size: 30px;">Contexte et objectif</p>'
    st.markdown(original_title, unsafe_allow_html=True)
    st.markdown(text)
     
    st.write('''Les paris sportifs ont été légalisés en France en 2010 (loi du 6 avril 2010).
     Il existe aujourd’hui 17 sites de paris sportifs actifs et approuvés par l’ANJ, 
     l’Autorité Nationale des Jeux.''')
    st.write(''' Le football étant le sport le plus médiatisé, 
     les paris sportifs associés soulèvent des sommes d’argent impressionnantes. 
     Par exemple, lors de la coupe du monde de foot en 2018, plus de 690 millions d’euros 
     ont été pariés par les particuliers en France (source Arjel).''')

    url = 'https://www.data.gouv.fr/fr/organizations/autorite-de-regulation-des-jeux-en-ligne-arjel/'
    if st.button('Arjel'):
        webbrowser.open_new_tab(url)    
    
    st.write('''Les bookmakers sont des spécialistes des paris sportifs embauchés par ces sites 
     afin de fixer les cotes des matchs. Les cotes sont inversement proportionnelles à la 
     probabilité que le résultat d’un match se réalise, selon les bookmakers. 
     Donc plus la cote est basse pour la victoire d’une équipe, plus celle-ci a des chances de gagner.''')
    st.write('''Lorsque l’on parie, les gains sont calculés en multipliant la cote par la mise de départ. 
     Donc plus la cote est haute, plus on peut gagner d’argent, mais moins on a de chances de gagner !''')
     
    st.markdown("\n")  
    
    st.write('''L’objectif de ce projet est d’essayer de battre les algorithmes des bookmakers sur 
      l’estimation de la probabilité d’une équipe gagnant un match.''')
     
    st.write('''Nous allons mettre à profit nos compétences acquises en Data Science pour 
     tenter de gagner davantage d’argent qu’un parieur lambda, qui suivrait simplement 
     les cotes des bookmakers. Cela sera d’autant plus intéressant d’un point de vue 
     scientifique que nous ne sommes absolument pas spécialistes des paris sportifs. 
     C’est ainsi que l’on va pouvoir admirer toute la puissance de la data science, 
     même sans expertise métier !''')
   
    st.markdown(text)     
    plan = '<p style="font-family:Aral; color:steelblue; font-size: 30px;">Plan de la présentation</p>'
    st.markdown(plan, unsafe_allow_html=True)    
    st.write('''Les différentes étapes de notre présentation sont :''')    
    st.markdown("""       
               * le **dictionnaire des données** brutes utilisées,
               * quelques **visualisations** de ces données brutes,
               * les étapes de **features engineering** effectuées pour obtenir des données utlisables dans la recherche du meilleur modèle,
               * la partie **modélisation** qui laissera le choix à l'utilisateur de tester différents modèles et hyperparamètres
               * le **meilleur modèle** trouvé,
               * les résultats de ce modèle en terme de **gains** selon des stratégies à tester
               * la **conclusion** qui permettra de présenter les axes d'amélioration que nous pourrions apporter à notre modèle et stratégie de gains.""")    