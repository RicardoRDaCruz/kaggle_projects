import pandas as pd
from pandas.plotting import scatter_matrix
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge
from sklearn import svm
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import classification_report, mean_absolute_error, mean_squared_error
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression

def read_csv_to_dataframe(file_path):
    try:
        # Using pandas read_csv function to read the file into a DataFrame
        df = pd.read_csv(file_path)
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

def replace_missing_values(df, lista_colunas):    
    for coluna in lista_colunas:
        df[coluna].fillna("Não Identificado", inplace=True)  
    return df

def grafico_barra(df, lista_colunas,metodo,titulo):
    if metodo=='soma':
        dataframe_plt=df[lista_colunas].groupby(lista_colunas[0]).sum()
    else:
        dataframe_plt=df[lista_colunas].groupby(lista_colunas[0]).mean() 
    dataframe_plt = dataframe_plt.reset_index()
    plt.figure(figsize=(10, 6))
    p=plt.bar(dataframe_plt[lista_colunas[0]],dataframe_plt[lista_colunas[1]])
    plt.bar_label(p,label_type='center')
    plt.title(titulo, y=1.1)
    plt.xlabel(lista_colunas[0])
    plt.ylabel(lista_colunas[1])
    plt.show()

def stacked_bar(df,lista_colunas,titulo):
    df_stacked=df[lista_colunas]
    plt.figure(figsize=(10, 6))
    ax =sns.histplot(df_stacked, x=lista_colunas[0], hue=lista_colunas[2], weights=lista_colunas[1], multiple='stack')
    sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))
    plt.title(titulo, y=1.1)
    plt.xlabel(lista_colunas[0])
    plt.ylabel(lista_colunas[1])
    plt.show()
def grafico_barrah(df, lista_colunas,metodo,titulo):
    if metodo=='soma':
        dataframe_plt=df[lista_colunas].groupby(lista_colunas[0]).sum()
    else:
        dataframe_plt=df[lista_colunas].groupby(lista_colunas[0]).mean() 
    dataframe_plt = dataframe_plt.reset_index()
    plt.figure(figsize=(10, 6))
    p=plt.barh(dataframe_plt[lista_colunas[0]],dataframe_plt[lista_colunas[1]])
    plt.bar_label(p,label_type='center')
    plt.title(titulo, y=1.1)
    plt.xlabel(lista_colunas[0])
    plt.ylabel(lista_colunas[1])
    plt.show()

def grafico_pizza(df,lista_colunas,titulo):
    dataframe_plt=df[lista_colunas].groupby(lista_colunas[0]).sum()
    dataframe_plt = dataframe_plt.reset_index()
    plt.pie(dataframe_plt[lista_colunas[1]],labels=dataframe_plt[lista_colunas[0]], autopct='%1.1f%%')
    plt.title(titulo, y=1.1) 
    plt.show()

def grafico_hist(df,coluna,titulo):
    plt.figure(figsize=(10, 6))
    plt.hist(df[coluna], orientation='horizontal')
    plt.title(titulo, y=1.1)
    plt.xlabel(coluna)
    plt.show()

def grafico_box(df, lista_coluna,titulo):
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df, x=lista_coluna[0], y=lista_coluna[1])
    plt.title(titulo, y=1.1)
    plt.xlabel(lista_coluna[0])
    plt.ylabel(lista_coluna[1])
    plt.show()

def grafico_pontos(df,lista_colunas,metodo,titulo):
    if metodo=='soma':
        dataframe_plt=df[lista_colunas].groupby(lista_colunas[0]).sum()
    else:
        dataframe_plt=df[lista_colunas].groupby(lista_colunas[0]).mean() 
    dataframe_plt = dataframe_plt.reset_index()
    plt.figure(figsize=(10, 6))
    plt.scatter(dataframe_plt[lista_colunas[0]],dataframe_plt[lista_colunas[1]])
    plt.title(titulo, y=1.1)
    plt.xlabel(lista_colunas[0])
    plt.ylabel(lista_colunas[1])
    plt.show()  

def normalize(df, lista_colunas):
    for i in range(0,len(lista_colunas)):
        max = df[lista_colunas[i]].max()    
        if max == 0:
            raise ValueError("All values in the column are zero, cannot normalize.")
        df[lista_colunas[i]] -= max    
        df[lista_colunas[i]] /= max
        return df

def main():
    arquivo="industrial_accidents_in_brazil_from_news.csv"
    dataframe=read_csv_to_dataframe(arquivo)
    print(dataframe.columns)
    print(dataframe.shape) 
    Na_number(dataframe)        
    dataframe['contador']=1        
    dataframe=change_data_types(dataframe, ['vitimas','fatalidades','grau','mes','ano'])   
    dataframe=replace_missing_values(dataframe,['evento_especifico','processo','area_especifica','evento'])
    grafico_barra(dataframe,['ano','contador'],'soma','Número de Acidentes ao longo do período')
    stacked_bar(dataframe,['ano','contador','grau'],'Distribuição do grau dos acidentes ao longo dos anos')    
    grafico_barra(dataframe,['ano','vitimas'],'soma','Número de Vítimas ao longo do período')
    grafico_barra(dataframe,['ano','fatalidades'],'soma','Número de Fatalidades ao longo do período')
    grafico_barra(dataframe,['grau','contador'],'soma','Distribuição do grau dos acidentes')    
    grafico_box(dataframe, ['grau','vitimas'], 'Boxplot do grau dos acidentes com as vítimas resultantes')
    grafico_box(dataframe, ['grau','fatalidades'], 'Boxplot do grau dos acidentes com as fatalidades resultantes')
    grafico_pizza(dataframe,['area_da_industria','contador'],'Separação de acidentes pelas Áreas das Indústrias')
    grafico_hist(dataframe,['area_da_industria'],'Histograma para as Áreas das Indústrias') 
    stacked_bar(dataframe,['ano','contador','area_da_industria'],'Distribuição dos acidentes ao longo dos anos segmentando pela Área das Indústrias')
    grafico_barrah(dataframe,['area_da_industria','grau'],'mean','Média do grau dos acidentes com as Áreas da Indústria')
    stacked_bar(dataframe,['ano','vitimas','area_da_industria'],'Distribuição das vítimas ao longo dos anos segmentando pela Área das Indústrias') 
    grafico_barrah(dataframe,['area_da_industria','vitimas'],'soma','Número das vítimas dos acidentes com as Áreas da Indústria')
    grafico_barrah(dataframe,['area_da_industria','vitimas'],'mean','Média das vítimas dos acidentes com as Áreas da Indústria')
    stacked_bar(dataframe,['ano','fatalidades','area_da_industria'],'Distribuição das fatalidades ao longo dos anos segmentando pela Área das Indústrias')
    grafico_barrah(dataframe,['area_da_industria','fatalidades'],'soma','Número das fatalidades dos acidentes com as Áreas da Indústria')
    grafico_barrah(dataframe,['area_da_industria','fatalidades'],'mean','Média das fatalidades dos acidentes com as Áreas da Indústria')
    grafico_pizza(dataframe[dataframe['area_da_industria']=='Alimentos'],['area_especifica','contador'],'Exploração das áreas específicas envolvidas em acidentes na Área de Alimentos')
    grafico_pizza(dataframe[dataframe['area_da_industria']=='Química'],['area_especifica','contador'],'Exploração das áreas específicas envolvidas em acidentes na Área Química')
    grafico_pizza(dataframe,['processo','contador'],'Separação de acidentes pelos Processos')
    grafico_hist(dataframe,'processo','Histograma para as os Processos')
    stacked_bar(dataframe,['ano','contador','processo'],'Distribuição dos acidentes ao longo dos anos segmentando pelos Processos')
    grafico_pizza(dataframe[dataframe['processo']=='Produção'],['processo_especifico','contador'],'Exploração dos processos específicos envolvidas em acidentes na Produção')
    grafico_barrah(dataframe,['processo','grau'],'mean','Média do grau dos acidentes com os Processos')
    stacked_bar(dataframe,['ano','vitimas','processo'],'Distribuição das vítimas ao longo dos anos segmentando pelos Processos') 
    grafico_barrah(dataframe,['processo','vitimas'],'soma','Número das vítimas dos acidentes com os Processos')
    grafico_barrah(dataframe,['processo','vitimas'],'mean','Média das vítimas dos acidentes com os Processos')
    stacked_bar(dataframe,['ano','fatalidades','processo'],'Distribuição das fatalidades ao longo dos anos segmentando pelos Processos')
    grafico_barrah(dataframe,['processo','fatalidades'],'soma','Número das fatalidades dos acidentes com os Processos')
    grafico_barrah(dataframe,['processo','fatalidades'],'mean','Média das fatalidades dos acidentes com os Processos')
    grafico_pizza(dataframe,['evento','contador'],'Separação de acidentes pelos Eventos')
    grafico_hist(dataframe,['evento'],'Histograma para os Eventos')    
    stacked_bar(dataframe,['ano','contador','evento'],'Distribuição dos acidentes ao longo dos anos segmentando pelos Eventos')
    grafico_barrah(dataframe,['evento','grau'],'mean','Média do grau dos acidentes com os Eventos')
    stacked_bar(dataframe,['ano','vitimas','evento'],'Distribuição das vítimas ao longo dos anos segmentando pelos Eventos') 
    grafico_barrah(dataframe,['evento','vitimas'],'soma','Número das vítimas dos acidentes com os Eventos')
    grafico_barrah(dataframe,['evento','vitimas'],'mean','Média das vítimas dos acidentes com os Eventos')
    stacked_bar(dataframe,['ano','fatalidades','evento'],'Distribuição das fatalidades ao longo dos anos segmentando pelos Eventos')
    grafico_barrah(dataframe,['evento','fatalidades'],'soma','Número das fatalidades dos acidentes com os Eventos')
    grafico_barrah(dataframe,['evento','fatalidades'],'mean','Média das fatalidades dos acidentes com os Eventos')
    grafico_pizza(dataframe,['estado','contador'],'Separação de acidentes pelas Estados')
    grafico_hist(dataframe,['estado'],'Histograma para os Estados')
    grafico_pizza(dataframe[dataframe['estado']=='SP'],['cidade','contador'],'Exploração das cidades envolvidas em acidentes em SP')
    grafico_pontos(dataframe,['grau','vitimas'],'soma','Avaliação do número de vítimas de acordo com o grau dos acidentes')
    grafico_pontos(dataframe,['grau','vitimas'],'mean','Avaliação da média de vítimas de acordo com o grau dos acidentes')
    grafico_pontos(dataframe,['grau','fatalidades'],'soma','Avaliação do número fatalidades de acordo com o grau dos acidentes')    
    grafico_pontos(dataframe,['grau','fatalidades'],'mean','Avaliação da média de fatalidades de acordo com o grau dos acidentes')  
    grafico_pontos(dataframe,['vitimas','fatalidades'],'mean','Avaliação da média de fatalidades de acordo com o numero de vítimas')    
main()