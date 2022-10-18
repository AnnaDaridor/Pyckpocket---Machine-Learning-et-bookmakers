# -*- coding: utf-8 -*-

import streamlit as st

def conclusion():
    text = '''
    ---
    '''      
     
    original_title = '<p style="font-family:Aral; color:steelblue; font-size: 30px;">Conclusion</p>'
    st.markdown(original_title, unsafe_allow_html=True)
    st.markdown(text)     
    st.write('''Même si nous avons des résultats corrects avec notre modèle final 
         ainsi qu’un gain positif après simulation, cela reste encore trop aléatoire. 
         En effet, en faisant ces tests, nous pouvons évaluer si notre modèle est bon ou pas, 
         alors qu’en situation réelle, nous ne le saurions qu’à la fin de la saison, 
         après avoir gagné ou perdu de l’argent.''')
    st.write('''Nous avons conscience que notre modèle est encore largement améliorable, 
         une approche possible serait d’ajouter plus de variations dans les données d’entrée. 
         En effet, une grande partie des matchs sont des victoires de l’équipe à domicile. 
         L’algorithme a donc moins d'informations pour prédire les victoires des équipes à l'extérieur, 
         ainsi que les matchs nuls.''') 
    st.write("""Il est également possible d’ajouter d’autres features grâce par exemple au WebScraping (ce que nous aurions fait si nous 
         disposions de plus de temps), comme :""")
    st.markdown("""       
            * le(s) joueur(s) blessé(s),
            * la météo, 
            * l’affluence prévue du match,
            * l’arbitre,
            * le changement d’entraineur,
            * le fait d’avoir un match de championnat d’Europe avant ou après le match...""")
    st.write('''Pour aller plus loin, il faudrait également s'intéresser aux caractéristiques des joueurs
         qui constituent chaque équipe. Avec la feuille de match de chaque rencontre, il est déjà bien plus 
         facile de faire des pronostics. C’est ce que font les bookmakers chaque jour, c’est un métier 
         qui ne s’improvise pas.''')
    st.write('''De plus, il faudrait mettre en place l'automatisation de la récupération des données et du calcul des nouveaux indicateurs à 
         chaque fin de journée de championnat pour pouvoir utiliser les résultats de la modélisation de manière automatique afin de prévoir 
         les résultat de la journée suivante.''')            