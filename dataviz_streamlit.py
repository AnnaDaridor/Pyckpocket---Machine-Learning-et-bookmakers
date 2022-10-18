# -*- coding: utf-8 -*-

import streamlit as st
from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt
from bokeh.plotting import figure
from bokeh.models.widgets import Panel, Tabs
from bokeh.models import ColumnDataSource
from bokeh.models.tools import HoverTool
import plotly.express as px


def dataviz():
    text = '''
    ---
    '''   
    
#    df_by_team = pd.read_csv(r"C:\Users\Isabelle\Documents\Streamlit\df_by_team.csv")    
    df_by_team = pd.read_csv(r"df_by_team.csv")     
    df_by_team['Saison'] = df_by_team['Saison'].astype(str)
    
    # Gestion des dates au format date avec format européen
    df= pd.read_csv(r'df.csv', parse_dates=['Date'],dayfirst=True,index_col=0)
    # La saison lue en tant que numérique est remise en chaine de caractère
    df['Saison'] = df['Saison'].astype(str)
    
    original_title = '<p style="font-family:Aral; color:steelblue; font-size: 30px;">Data Visualisation</p>'
    st.markdown(original_title, unsafe_allow_html=True)    
    st.markdown(text)
       
    st.subheader('Caractéristiques des clubs')
    
    st.info("""Les heatmaps suivantes permettent de voir que le PSG est toujours dans les meilleures équipes sur toutes les caractéristiques (beaucoup de vert). 
             Cela met aussi en évidence les équipes qui vont descendre (beaucoup de rouge), même s'il y a quelques incohérences comme Bastia en 2015 dont les statistiques sont mauvaises mais qui ne descend pas. """)   
    st.markdown("\n") 
    
    saison = st.selectbox(label = "Choisissez la saison :" , options = ['2015', '2016', '2017', '2018', '2019', '2020', '2021'])    
    if saison == '2015' : 
        image = Image.open("Caracteristique_clubs2015.png")
    elif saison == '2016' : 
        image = Image.open("Caracteristique_clubs2016.png")    
    elif saison == '2017' : 
        image = Image.open("Caracteristique_clubs2017.png")   
    elif saison == '2018' : 
        image = Image.open("Caracteristique_clubs2018.png")   
    elif saison == '2019' : 
        image = Image.open("Caracteristique_clubs2019.png")
    elif saison == '2020' : 
        image = Image.open("Caracteristique_clubs2020.png")
    elif saison == '2021' : 
        image = Image.open("Caracteristique_clubs2021.png")   
       
    st.image(image,width=500)
 
    if saison == '2016' : 
        st.info("""La bonne saison de Monaco qui termine champion ressort sur toutes les caractéristiques.""")
    elif saison == '2020' : 
        st.info("""Pour Lille qui devient champion en 2020, l'équipe ne ressort que sur le nombre de points moyens (ce qui est évident) et sur le nombre de cartons rouges peu nombreux.""")
             
    st.markdown("\n")        
    
    st.markdown(text)
    st.subheader('Pourcentages de victoires/nuls/défaites')
    # Couleurs utilisées dans le bar chart, le dernier correspond à du bordeaux
    cols = ['navy','white','#922b21']
    
    hover = HoverTool(
            tooltips=[
                ("Victoire", "@Victoire{:.2%}"),
                ("Nul", "@Nul{:.2%}"),
                ("Défaite", "@{Defaite}{:.2%}")
                 ])
    
    dictionnaire = {}
    # Liste des saisons étudiées
    saisons = ['2015','2016','2017','2018','2019','2020','2021']
    
    # Le même graphique est effectué sur toutes les saisons, on boucle sur chacune d'elles
    for saison in saisons:
    
      # Récupérer les données de la saison
      df_statsaison = df_by_team[df_by_team['Saison']==saison].drop(['Saison'],axis=1)
    
      # Calculer les agrégations des résultats des matchs par équipe
      dictionnaire['ctab_'+saison]=pd.crosstab(df_statsaison['Equipe'],
                  df_statsaison['Resultat_match'],rownames=['Equipe'],colnames=['Resultat_match'],
                  normalize = 'index').reset_index()
      dictionnaire['ctab_'+saison].sort_values(['Victoire','Nul'], ascending=True,inplace=True)
      dictionnaire['source'+saison]=ColumnDataSource(dictionnaire['ctab_'+saison])  
      
      # Création du bar chart de la saison
      dictionnaire['p'+saison]=figure(y_range = dictionnaire['ctab_'+saison].Equipe, plot_width = 650, plot_height = 400)
      dictionnaire['p'+saison].hbar_stack(['Victoire','Nul','Defaite'], y='Equipe', color = cols, height=0.5, source=dictionnaire['ctab_'+saison],line_color='black', legend_label=['Victoire','Nul','Défaite'])
      dictionnaire['p'+saison].add_tools(hover)
      dictionnaire['p'+saison].legend.location = "center_right"
      dictionnaire['p'+saison].legend.background_fill_alpha = 0.8
      dictionnaire['p'+saison].title.text = 'Pourcentage de V/D/N par équipe et saison'
      dictionnaire['tab'+saison] = Panel(child=dictionnaire['p'+saison], title="Saison"+ saison)
    
    tabs = Tabs(tabs=[ dictionnaire['tab2015'],dictionnaire['tab2016'],dictionnaire['tab2017'],dictionnaire['tab2018'],dictionnaire['tab2019'],dictionnaire['tab2020'],dictionnaire['tab2021']])
    
    st.bokeh_chart(tabs, use_container_width=False)
    
    
    st.markdown("\n") 
    st.info("""Cette visualisation permet de voir les différences de pourcentage de victoire/nul/défaite entre les équipes de haut de tableau et celles de bas de tableau. 
    L'équipe avec le plus de victoires n'est pas toujours celle qui gagnera le championnat. En 2020, le PSG a gagné plus de matchs que Lille mais a aussi concédé beaucoup plus de défaites. 
    C'est la constance des résultats de Lille qui leur a permis d'être champion de France.
    Nous pouvons aussi voir qu'il y a une différence assez significative entre les pourcentages de victoires des 4 ou 5 premiers du championnat et les autres équipes.""")
    
    st.markdown("\n") 
    st.markdown("\n") 

    st.markdown(text)
    st.subheader('Nombre de buts par match domicile/extérieur')
    st.info("""Comme attendu, nous observons bien une hausse du nombre de buts en fonction du nombre de tirs. 
Les médianes du nombre de tirs sont assez stables d'une saison à l'autre sur les nombres de buts faibles. 
A partir de 4 buts, les dispersions sont plus importantes. 
Ces graphiques mettent aussi en évidence quelques outliers, comme par exemple 11 tirs cadrés pour aucun but en 2016 (Bastia/Dijon).""")  
    
    #df = pd.read_csv(r"C:\Users\Isabelle\Documents\Streamlit\df.csv", index_col=0)
    # Création d'un dataframe spécifique pour modifier des variables
    df_box = df
    df_box['Nb_buts_equipe_ext']=df_box['Nb_buts_equipe_ext'].replace([6,7,8,9], [5,5,5,5]) 
    df_box['Nb_buts_equipe_dom']=df_box['Nb_buts_equipe_dom'].replace([6,7,8,9], [5,5,5,5]) 
    
    
    # Création du box plot des données à domicile
    fig_box=px.box(data_frame=df.sort_values(['Saison','Nb_buts_equipe_dom']),y='Nb_tirs_cadres_equipe_dom',facet_col='Nb_buts_equipe_dom',color='Saison',
     labels={
                         "Nb_tirs_cadres_equipe_dom": "Nombre de tirs cadrés à domicile",
                         'Nb_buts_equipe_dom': ""
                     })
    fig_box.update_layout(title_text='Nombre de buts par match à domicile (5 correspond à 5 buts et plus)', title_x=0.5)    
    st.plotly_chart(fig_box, use_container_width=True)   
    
    # Création du box plot des données à l'extérieur
    fig_box2 = px.box(data_frame=df.sort_values(['Saison','Nb_buts_equipe_ext']),y='Nb_tirs_cadres_equipe_ext',facet_col='Nb_buts_equipe_ext',color='Saison',
                 labels={
                         "Nb_tirs_cadres_equipe_ext": "Nombre de tirs cadrés à l'extérieur",
                         'Nb_buts_equipe_ext': ""
                     })
    fig_box2.update_layout(title_text="Nombre de buts par match à l'extérieur (5 correspond à 5 buts et plus)", title_x=0.5)    
    st.plotly_chart(fig_box2, use_container_width=True)    
    
    st.markdown(text)   
    st.subheader('Bonnes prévisions des bookmakers')
    st.info("""Le pourcentage de bonnes prédictions des bookmakers est de 51,2 % (en considérant qu'une bonne prédiction est le résultat pour la cote la plus basse).
             Les bookmakers prévoient très mal les matchs nuls mais quand il y a une victoire à domicile, ils l'avaient en bien prévu à plus de 85 % de fois.""")
    
    image_book = Image.open("Previsions_bookmakers.png")   
    st.image(image_book,width=500)    
   
    st.markdown("\n") 
    st.markdown("\n")     

    st.markdown(text)
    st.subheader('Gain en suivant les cotes des bookmakers')
    st.info("""En suivant les cotes des bookmakers, nous aurions gagné de l'argent 3 années sur les 6 saisons 2015 à 2020 (très peu en 2017) avec un maximum d'environ 250 € en 2016. 
             Les pertes auraient été importantes en 2018 (225 €). 
             Le début de saison 2021-2022 ne laisse pas présager de bons résultats avec une perte assez importante à mi-saison.""")

    # Nous ne gardons que les données de cote
    df_cote=df[['Saison','Equipe_domicile','Equipe_exterieur','Resultat_match','Moy_cote_domicile','Max_cote_domicile','Min_cote_domicile','Moy_cote_exterieur','Max_cote_exterieur','Min_cote_exterieur','Moy_cote_nul',
               'Max_cote_nul','Min_cote_nul']]
    
    # Recherche du résultat le plus probable
    # Le résultat le plus probable pour les bookmakers est celui avec la cote la plus faible
    df_cote['Cote_result_plus_probable']=df_cote[['Max_cote_domicile','Max_cote_exterieur','Max_cote_nul']].min(axis=1)
    
    # Récupération de la modalité la plus probable
    df_cote.loc[df_cote['Cote_result_plus_probable'] == df_cote['Max_cote_domicile'], 'Result_plus_probable'] = 'H'
    df_cote.loc[df_cote['Cote_result_plus_probable'] == df_cote['Max_cote_exterieur'], 'Result_plus_probable'] = 'A' 
    df_cote.loc[df_cote['Cote_result_plus_probable'] == df_cote['Max_cote_nul'], 'Result_plus_probable'] = 'D'
    
    # Mise de départ
    df_cote[['gain_perte']] = -10
    
    # Si bon pari, calcul du gain : cote * mise - mise
    df_cote.loc[df_cote['Resultat_match'] == df_cote['Result_plus_probable'], 'gain_perte'] = 10 * df_cote['Cote_result_plus_probable'] - 10
    
    # Calcul des cumuls des montants
    # La somme de départ est de 1 000 €
    df_cote['gain_perte_tot'] = df_cote.groupby(['Saison'])['gain_perte'].cumsum()+1000
    
    # Numéro du match de la saison
    df_cote['x'] = 1
    df_cote['x'] = df_cote.groupby(['Saison'])['x'].cumsum()
    
    # Pour affichage par saison
    df_cote.set_index(['x'], inplace=True)
    
    #import scipy.special

    Saison = ['2015','2016','2017','2018','2019','2020','2021']
    color = ['blue','darkorange','green','red','darkviolet','saddlebrown','fuchsia']
    
    p = figure(title="Gain/perte par saison en suivant les cotes des bookmakers",plot_width=800, plot_height=450)
    
    plt.figure(figsize = (15, 8))
    
    for i in Saison :
        histogram = df_cote.loc[(df_cote.Saison == i),['Saison','gain_perte_tot']].reset_index()
        source=ColumnDataSource(data=histogram)
        p.line('x','gain_perte_tot',  line_color = color[Saison.index(i)],
               source=source,legend_label=str(i))
        hover = HoverTool(tooltips = [('saison ', '@Saison'),
                                      ('valeur ', '@gain_perte_tot{int}'),
                                      ('match', '$x{int}')])
        p.add_tools(hover)
    
    p.xaxis.axis_label = "Nombre de matchs"
    p.yaxis.axis_label = "Niveau de gain (départ à 1000 €)"
    p.title.align='center'
    p.legend.click_policy="hide"
    p.legend.location = 'top_left'
    st.bokeh_chart(p, use_container_width=True)
    #show(p)
             
    #image_book = Image.open("Gain_bookmakers.png")   
    #st.image(image_book,width=800)    

    
  