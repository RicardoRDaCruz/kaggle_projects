import pandas as pd
from pandas.plotting import scatter_matrix
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime
from geopy.geocoders import Nominatim
import folium
import json

def read_csv_to_dataframe(file_path):
    try:
        # Using pandas read_csv function to read the file into a DataFrame
        df = pd.read_csv(file_path, sep=",")
        return df
    except FileNotFoundError:
        return "Error: The file at the provided path was not found."
    except pd.errors.ParserError:
        return "Error: The file could not be parsed as a CSV."
    except Exception as e:
        return f"An unexpected error occurred: {str(e)}"
    
def Na_number(df):
    df_nulos=df.isna()
    df_nulos['contador']=1
    info_x=[]
    info_y=[]
    for coluna in df_nulos.columns:
        if coluna!='contador':
            agrupado=df_nulos[['contador',coluna]].groupby(coluna).sum()
            info_x.append(coluna)
            info_y.append(agrupado.loc[False,'contador'])
    plt.figure(figsize=(10, 6))
    p=plt.barh(info_x,info_y)
    plt.bar_label(p,label_type='center')
    plt.title('Número de registros não-nulos', y=1.1)
    plt.show()

def change_data_types(df, columns):
    for col in columns:
        if col in df.columns:
            df[col] = df[col].astype(float)
    return df

def date_transformer(df, column):
    i=0
    while i<df.shape[0]:        
        df.loc[i,column]=datetime.strptime(df.loc[i,column],'%Y-%m-%dT%H:%M:%SZ').date()
        df.loc[i,'Ano']=df.loc[i,column].year
        i+=1
    return df

def adress_transformer(df,column):
    i=0
    j=0
    while i<df.shape[0]:              
        if ',' in df.loc[i,column]:
            primeira_parte=df.loc[i,column].split(",")[1]     
            df.loc[i,'Bairro']=primeira_parte.split("-")[0].strip()  
        else:
            df.loc[i,'Bairro']=df.loc[i,column].split("-")[0].strip()    
        i+=1
    while j<df.shape[0]:              
        if '-' in df.loc[j,column]:
            primeira_parte=df.loc[j,column].split("-")[1]     
            df.loc[j,'Cidade']=primeira_parte.split("/")[0].strip()  
        else:
            df.loc[j,'Cidade']=df.loc[j,column].split("/")[0].strip()    
        j+=1
    return df

def latlon_transformer(df, coluna):    
    i=0
    while i<df.shape[0]:
        print(i)
        if type(df.loc[i,"Latitude"])!='numpy.float64':
            endereco=df.loc[i,coluna]
            loc=Nominatim(user_agent='Geopy Library')
            get_loc=loc.geocode(endereco, timeout=12)
            if get_loc != None:
                print('substituiu')
                df.loc[i,'Latitude']=get_loc.latitude
                df.loc[i,'Longitude']=get_loc.longitude
        i+=1
    return df

def mapa_pontos(df, map):
    i=0
    df_null=df.isnull()
    while i<df.shape[0]:        
        if df_null.loc[i,"Latitude"]==False:
            print(i)
            folium.Marker(
            location=[df.loc[i,"Latitude"], df.loc[i,"Longitude"]],            
            #icon=folium.Icon(icon="cloud"),
            ).add_to(map)
        i+=1


def grafico_hist(df,coluna,titulo):
    nbins=int(np.ceil(np.sqrt(df.shape[0])))
    plt.figure(figsize=(10, 6))
    plt.hist(df[coluna],bins=nbins)
    plt.title(titulo, y=1.1)
    plt.xlabel(coluna)
    plt.locator_params(axis='both', nbins=20) 
    plt.show()

def grafico_box(df, lista_coluna,titulo):
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df, x=lista_coluna[0], y=lista_coluna[1])
    plt.title(titulo, y=1.1)
    plt.xlabel(lista_coluna[0])
    plt.ylabel(lista_coluna[1])
    plt.locator_params(axis='y', nbins=20) 
    plt.show()

def grafico_box_bairros_preco(df, lista):
    df_bairros=df[df["Bairro"].isin(lista)]
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df_bairros,y=df_bairros['Bairro'],x=df_bairros['Price'])
    plt.title('Boxplot de preços para os 10 bairros com mais registros', y=1.1)
    plt.xlabel('Bairro')
    plt.ylabel('Price')
    plt.locator_params(axis='x', nbins=20) 
    plt.show()
    
def grafico_box_bairros_area(df, lista):
    df_bairros=df[df["Bairro"].isin(lista)]
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df_bairros, y=df_bairros['Bairro'],x=df_bairros['Area'],)
    plt.title('Boxplot da área para os 10 bairros com mais registros', y=1.1)
    plt.xlabel('Bairro')
    plt.ylabel('Price')
    plt.locator_params(axis='x', nbins=20)
    plt.show()

def grafico_box_bairros_preco_area(df, lista):
    df_bairros=df[df["Bairro"].isin(lista)]
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df_bairros, y=df_bairros['Bairro'],x=df_bairros['Price/area'],)
    plt.title('Boxplot da relaçao preço/área para os 10 bairros com mais registros', y=1.1)
    plt.xlabel('Bairro')
    plt.ylabel('Price')
    plt.locator_params(axis='x', nbins=20)
    plt.show()

def grafico_pontos(df,lista_colunas,titulo):    
    plt.figure(figsize=(10, 6))
    plt.scatter(df[lista_colunas[0]],df[lista_colunas[1]])
    plt.title(titulo, y=1.1)
    plt.xlabel(lista_colunas[0])
    plt.ylabel(lista_colunas[1])
    plt.locator_params(axis='x', nbins=20)
    plt.show()  

def grafico_barrah(df, lista_colunas,titulo):
    plt.figure(figsize=(10, 6))
    p=plt.barh(df[lista_colunas[0]],df[lista_colunas[1]])
    plt.bar_label(p,label_type='center')
    plt.title(titulo, y=1.1)
    plt.xlabel(lista_colunas[1])
    plt.ylabel(lista_colunas[0])
    plt.show()

def dataframe_anos(df,ano):
    dataframe_ano=df[df.Ano==ano]
    dataframe2_ano=dataframe_ano[['contador','Bairro']].groupby('Bairro').sum().reset_index()
    dataframe_bairros=dataframe2_ano.sort_values('contador',ascending=False).head(50)
    grafico_barrah(dataframe_bairros,['Bairro','contador'],"50 bairros com mais registros em "+str(ano))

    

def main():
    arquivo1="SaoPaulo.txt"
    dataframe=read_csv_to_dataframe(arquivo1)    
    print(dataframe.columns)
    #dataframe=latlon_transformer(dataframe,'Adress')
    dataframe=date_transformer(dataframe, 'created_date') 
    #dataframe=dataframe.drop(['ID','extract_date'], axis=1)  
    dataframe=adress_transformer(dataframe, 'Adress')   
    change_data_types(dataframe,['Price','Area','Bedrooms','Bathrooms','Parking_Spaces'])      
    #dataframe.to_csv('SaoPaulo_OnlyAppartments_2024-11-25_TRATADO.csv', index=False)
    print(dataframe.dtypes)
    print(dataframe.shape)   
    
    geojson_data1=''
    geojson_data2=''
    with open('SIRGAS_SHP_distrito.geojson') as f:
        geojson_data1 = json.load(f)
    with open('SIRGAS_SHP_subprefeitura_polygon.geojson') as f:
        geojson_data2 = json.load(f)
    mapa=folium.Map((-23.536332497520338, -46.63421064628902), tiles="cartodb positron")
    folium.GeoJson(geojson_data1, 
                   name="distritos",
                   style_function=lambda feature: {
                    "color": "blue",
                    'weight':1.0,
                    }).add_to(mapa)
    folium.GeoJson(geojson_data2, 
                   name="subprefeituras",
                   style_function=lambda feature: {
                    'fillColor': '#00FFFFFF',
                    'lineColor': '#00FFFFFF',
                    }).add_to(mapa)
    mapa_pontos(dataframe, mapa)
    mapa.save("mapa.html")
    dataframe=dataframe.drop_duplicates()  
    Na_number(dataframe)
    print(dataframe[['Price','Area','Bedrooms','Bathrooms','Parking_Spaces','Ano']].describe())
    
    grafico_hist(dataframe,['Ano'],'Histograma para os Anos')
    grafico_hist(dataframe,['Price'],'Histograma para os Preços')   
    grafico_hist(dataframe,['Area'],'Histograma para a Área')
    dataframe['Price/area']= dataframe['Price']/dataframe['Area']
    grafico_hist(dataframe,['Price/area'],'Histograma para o Preço/Area')
    grafico_hist(dataframe,['Bedrooms'],'Histograma para o número de Quartos') 
    grafico_hist(dataframe,['Bathrooms'],'Histograma para o número de Banheiros')
    grafico_hist(dataframe,['Parking_Spaces'],'Histograma para o número de vagas de estacionamento')    
    dataframe['contador']=1
    dataframe2=dataframe[['contador','Cidade']].groupby('Cidade').sum().reset_index()
    dataframe_cidades=dataframe2.sort_values('contador',ascending=False).head(50).reset_index()     
    grafico_barrah(dataframe_cidades,['Cidade','contador'],"50 cidades com mais registros")
    print(pd.crosstab(dataframe['Cidade'],dataframe['Ano']))   
    dataframe2=dataframe[['contador','Bairro']].groupby('Bairro').sum().reset_index()
    dataframe_bairros=dataframe2.sort_values('contador',ascending=False).head(50).reset_index()     
    grafico_barrah(dataframe_bairros,['Bairro','contador'],"50 bairros com mais registros")
    print(pd.crosstab(dataframe['Bairro'],dataframe['Ano']))
    dataframe_anos(dataframe,2018.0)
    dataframe_anos(dataframe,2019.0)
    dataframe_anos(dataframe,2020.0)
    dataframe_anos(dataframe,2021.0)
    dataframe_anos(dataframe,2022.0)
    dataframe_anos(dataframe,2023.0)
    dataframe_anos(dataframe,2024.0)    
    grafico_box(dataframe, ['Bedrooms','Price'], 'Boxplot do número de quartos com o preço dos apartamentos')
    grafico_box(dataframe, ['Bedrooms','Area'], 'Boxplot do número de quartos com a área dos apartamentos')
    grafico_box(dataframe, ['Bedrooms','Price/area'], 'Boxplot do número de quartos com a relação preço/area dos apartamentos')
    grafico_box(dataframe, ['Bathrooms','Price'], 'Boxplot do número de banheiros com o preço dos apartamentos')
    grafico_box(dataframe, ['Bathrooms','Price/area'], 'Boxplot do número de banheiros com a relação preço/area dos apartamentos')
    grafico_box(dataframe, ['Parking_Spaces','Price'], 'Boxplot do número de vagas de estacionamento com o preço dos apartamentos')
    grafico_box(dataframe, ['Ano','Price'], 'Boxplot dos anos de anúncio com o preço dos apartamentos')
    grafico_box(dataframe, ['Ano','Area'], 'Boxplot dos anos de anúncio com a área dos apartamentos')
    grafico_box(dataframe, ['Ano','Price/area'], 'Boxplot do número de quartos com a relação preço/area dos apartamentos')
    grafico_pontos(dataframe,['Area','Price'],'Avaliação do Preço do apartamento de acordo com a área dos apartamentos') 
    i=0
    lista10_bairros=[]
    while i<10:
        bairro= dataframe_bairros.loc[i,'Bairro']     
        lista10_bairros.append(bairro) 
        i+=1   
    grafico_box_bairros_preco(dataframe,lista10_bairros)
    grafico_box_bairros_area(dataframe,lista10_bairros)   
    grafico_box_bairros_preco_area(dataframe,lista10_bairros) 
main()
