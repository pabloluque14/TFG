#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 18 16:38:11 2019

@author: pabloluque
"""


import time  
import pandas as pd
import random
import numpy as np
import click

hashTableSupp = {}

@click.command()
@click.option('--data_file', '-d', default=None, required=True,
              help=u'Fichero con los datos CSV.')

@click.option('--jerarquia_file', '-j', default='no_file', required=False,
              help=u'Fichero con la gramática de contexto libre')

@click.option('--multi', '-mi',is_flag=True, default=False, show_default=True,
              help=u'Para indicar si la base de datos contiene un formato de multi-instancias o no')

@click.option('--tam_elite', '-e', default=20, type=int , required=True, 
              help=u'tamaño de la élite')

@click.option('--min_support', '-s', default=0.3, type=float , required=True, 
              help=u'soporte mínimo de los patrones a extraer')

@click.option('--it_rs', '-rs', default=1000, type=int , required=False, 
              help=u'iteraciones algoritmo random search')

@click.option('--it_rsb', '-rsb', default=500, type=int , required=False, 
              help=u'iteraciones algoritmo random search balanceado')

@click.option('--it_evo', '-evo', default=14, type=int , required=False, 
              help=u'generaciones algoritmo evolutivo')

@click.option('--tam_torneo', '-k', default=3, type=int , required=True, 
              help=u'tamaño del torneo')

@click.option('--prob_cruce', '-probCruce', default=0.85, type=float , required=False, 
              help=u'generaciones algoritmo evolutivo')

@click.option('--prob_mutacion', '-probMutacion', default=0.7, type=float , required=False, 
              help=u'tamaño del torneo')

@click.option('--tam_poblacion', '-t', default=40, type=int , required=False, 
              help=u'tamaño del torneo')


@click.option('--algoritmo', '-a', default=1, required=True,
              help=u'Algoritmo a utiliar [1: Apriori IN| 2: Apriori POST | 3: RS | 4: RS balanceado | 5: Evolutivo')



def extraer_patrones(data_file,jerarquia_file,multi,tam_elite,min_support,it_rs,it_rsb,it_evo,tam_torneo,prob_cruce,
                     prob_mutacion,tam_poblacion,algoritmo):
    
    if multi :
        
        #Se separa la columna de los ids del resto
        df=pd.read_csv(data_file, sep=',')
        
        df1=df.iloc[:,1:]
        df2=df.iloc[:,:1]

        #caso en el que no se haya introducido jerarquia
        if jerarquia_file == "no_file" :
        
            df_plus = df
            arbol=[]
            
        else :
            
            arbol,df_plus=generar_arbol(df1,jerarquia_file)
            #volvemos a introducir la fila de los ids
            df_plus['id'] = df2.values
            cols = df_plus.columns.tolist()
            cols.remove("id")
            cols2=[]
            cols2.append("id")
            [cols2.append(ele) for ele in cols]
            df_plus = df_plus[cols2]
  
    elif not multi :
        
        df=pd.read_csv(data_file, sep=',')
        
        if jerarquia_file == "no_file" :
        
            df_plus = df.copy()
            arbol=[]
            
        else :
            
            arbol,df_plus=generar_arbol(df,jerarquia_file)
    
    #se genera el archivo con la base de datos modificada
    if jerarquia_file != "no_file":
        name=data_file+"_"+jerarquia_file 
        name+=".csv"
        df_plus.to_csv(name, index=False, encoding='utf8')
    
    
    
    if algoritmo == 1 :
    
        ################     APRIORI VERSION IN     ################
        first=time.time()
        
        frequent_itemsets=ejecutar_aprioriIn(df_plus,arbol,min_support,multi)
        
        print("Apriori In tarda: ",(time.time()-first))         
        print("Patrones en total:",len(frequent_itemsets))
        
        #se selecciona la elite
        df_ordenada=frequent_itemsets.sort_values(by='support', ascending=False)       
        print("Resultado Apriori In: ",np.array(df_ordenada.iloc[:tam_elite,:1]).mean() )
        
        #se genera el archivo con la elite
        name="AprioriIN"+data_file+"_"+jerarquia_file 
        name+=".csv"
        df_ordenada.to_csv(name, index=False, encoding='utf8')

    elif algoritmo == 2:
        
        
        
        ################     APRIORI VERSION POST     ################
        first=time.time()       
    
        frequent_itemsets=ejecutar_aprioriPost(df_plus,arbol,min_support,multi)
        
        print("Apriori Post tarda: ",(time.time()-first))  
        print("Patrones en total:",len(frequent_itemsets))
        
        #se selecciona la elite
        df_ordenada=frequent_itemsets.sort_values(by='support', ascending=False)       
        print("Resultado Apriori Post: ",np.array(df_ordenada.iloc[:tam_elite,:1]).mean() )
        
        #se genera el archivo con la elite
        name="AprioriPOST"+data_file+"_"+jerarquia_file 
        name+=".csv"
        df_ordenada.to_csv(name, index=False, encoding='utf8')
    
    
    elif algoritmo == 3:
        
        ################     RANDOM SEARCH     ################
        first=time.time()
        
        ranking=randomSearch(df_plus,arbol,tam_elite,multi,it_rs) 
        
        print("Resultado RS: ",calcular_media_soporte(ranking,df_plus))
        print("Random Search tarda: ",(time.time()-first))
         
        #se genera el archivo con la elite
        name="RS"+data_file+"_"+jerarquia_file 
        name+=".csv"
        df_ranking=crearDF_ranking(ranking,df_plus,multi)
        df_ranking.to_csv(name, index=False, encoding='utf8')

    
    elif algoritmo == 4:
        
        ################     RANDOM SEARCH BALANCEADO     ################
        first=time.time()
        
        ranking=pseudoRandomSearch(df_plus,arbol,tam_elite,multi,it_rsb)
        
        print("Resultado RS balanceado: ",calcular_media_soporte(ranking,df_plus))    
        print("Random Search Balanceado tarda: ",(time.time()-first))
        
        #se genera el archivo con la elite
        name="RS_balanceado"+data_file+"_"+jerarquia_file 
        name+=".csv"
        df_ranking=crearDF_ranking(ranking,df_plus,multi)
        df_ranking.to_csv(name, index=False, encoding='utf8')
    
    
    
    elif algoritmo == 5:
        
        ################     EVOLUTIVO     ################
        first=time.time()
        
        ranking=evolutivo(df_plus,arbol,tam_elite,multi,tam_poblacion,it_evo,
                          tam_torneo,prob_cruce,prob_mutacion)
        
        print("Resultado evolutivo: ",calcular_media_soporte(ranking,df_plus))
        print("Evolutivo tarda: ",(time.time()-first))
        
        #se genera el archivo con la elite
        name="Evolutivo"+data_file+"_"+jerarquia_file 
        name+=".csv"
        df_ranking=crearDF_ranking(ranking,df_plus,multi)
        df_ranking.to_csv(name, index=False, encoding='utf8')
    



###############################       FUNCIONES GENERALES       ###############################

def crearDF_ranking(ranking,df_plus,multi):  
    """ 
    Crea una DataFrame a partir de la lista con la elite de los patrones de los algoritmos no exhaustivos.
    
    Parámetros
    -----------
    df_plus: DataFrame.  Base de datos.
    ranking: lista. Contiene la élite de los algoritmos.
    multi: Flag por si el problema es multi-instancia.
    
    Devuelve:
    ----------- 
    DataFrame. Elite de los algoritmos (patrones) y sus respectivos soportes calculados 
    """
    
    ranking2=np.array(ranking)
    df2 = pd.DataFrame(ranking2, columns=["elite"])
    
    if multi:
        
        df1=df_plus.iloc[:,1:]
        df3=df_plus.iloc[:,:1]
        
        soportes=[]
        for patron in ranking:
            soportes.append(calculaSoporteMI(df1,df3,patron))
    else:
        soportes=[]
        for patron in ranking:
            soportes.append(calculaSoporte(df_plus,patron))
   
    df2=df2.assign(support=soportes)

    return df2




def calcular_media_soporte(lista,df):
    """ 
    Calcula el soporte medio la elite de los algoritmos no exhaustivos
    Parámetros
    -----------
    df: DataFrame. Base de datos.
    lista: lista
    
    Devuelve:
    ----------- 
    float. Media del soporte de los patrones.
    """
    
    a=[]
    [a.append(calculaSoporte(df,ele)) for ele in lista]
    a= np.array(a)
    return a.mean()

def comprobar(patron,arbol,names):
    """ 
    Parámetros
    -----------
    patron: lista. Lista con nombres de las columnas 
    de la base de datos o los indices que equivale a un patrón
    arbol: lista. Contiene los patrones no válidos.
    names:
    
    Devuelve:
    ----------- 
    bool. rue o False si el patron es válido o no
    """
    
    elements=[]
    for i in patron:
        elements.append(names[i])
    
    elements=frozenset(elements)
    
    for nodo in arbol:            
        if frozenset.issubset(nodo,elements) == True:
            return True
    
    return False


def calculaSoporte(df,patron):
    """
    Calcula el soporte para un patrón, si no ha sido calculado previamente.
    Parámetros
    -----------
    df: DataFrame. Base de datos.
    patron:lista. Lista con nombres de las columnas 
    de la base de datos o los indices que equivale a un patrón
    
    Devuelve:
    -----------
    float entre 0 y 1. Soporte de un patron.
    """
    
    patron.sort()
    if ((str(patron).strip('[]')) in hashTableSupp):
        return hashTableSupp[str(patron).strip('[]')]
    else:
        
        X = df.values
        rows_count = float(X.shape[0])
        together= X[:, patron].all(axis=1)
        support = together.sum() / rows_count
        
        hashTableSupp[str(patron).strip('[]')]=support
        
        return support
    
def calculaSoporteMI(df1,df2,patron):
    """ 
    Calcula el soporte para un patrón si no ha sido calculado 
    previamente para el problema de multi-instancias.
    
    Parámetros
    -----------
    df1: DataFrame. Base de datos.
    df2: DataFrame. Columna con los ids.
    patron: lista. Lista con nombres de las columnas 
    de la base de datos o los indices que equivale a un patrón
    
    Devuelve:
    ----------- 
    float entre 0 y 1. Soporte de un patron.
    """
    
    X=df1.values
    Y=df2.values
    
    patron=list(patron)
    patron.sort()
    if ((str(patron).strip('[]')) in hashTableSupp):
        return hashTableSupp[str(patron).strip('[]')]
    else:
        unique = np.unique(Y, axis=0)
        
        i=0
        sup=0
        filas=[]
        for ele in X:   
            if X[i,patron].all() and Y[i] not in filas:
                sup+=1
                filas.append(Y[i])
            i+=1
                
        support=(sup/unique.shape[0])
        hashTableSupp[str(patron).strip('[]')]=support   
        return support



def quick_sort(lista,df,reverse):
    """ 
    Ordena la lista de forma recursiva.
    Parámetros
    -----------
    lista: lista.
    df: DataFrame. Base de datos.
    reverse: bool. Indica si se ordena en orden descendente o no.
    
    Devuelve:
    -----------
    lista. una nueva lista con los elementos ordenados. 
    """
    
    # Caso base
    if len(lista) < 2:
        return lista
    # Caso recursivo
    menores, medio, mayores = _partition(lista,df,reverse)
    return quick_sort(menores,df,reverse) + medio + quick_sort(mayores,df,reverse)

def _partition(lista,df,reverse):
    """
    Parámetros
    -----------
    lista: lista.
    df: DataFrame. Base de datos.
    reverse: bool. Indica si se ordena en orden descendente o no.
    
    Devuelve:
    ----------- 
    tres listas: menores, medio y mayores. 
    """
    
    pivote = lista[0].copy()
    menores = []
    mayores = []
    for x in range(1, len(lista)):
        if reverse == True:
            if calculaSoporte(df,lista[x]) < calculaSoporte(df,pivote):
                menores.append(lista[x].copy())
            else:
                mayores.append(lista[x].copy())
        else :
            if calculaSoporte(df,lista[x]) > calculaSoporte(df,pivote):
                menores.append(lista[x].copy())
            else:
                mayores.append(lista[x].copy())
            
    return menores, [pivote], mayores

def quick_sortMI(lista,df1,df2,reverse):
    """ 
    Ordena la lista de forma recursiva.
    Parámetros
    -----------
    lista: lista.
    df1: DataFrame. Base de datos.
    df2: DataFrame. Columna con los ids.
    reverse:

    Devuelve:
    -----------
    lista. una nueva lista con los elementos ordenados. 
    """

    # Caso base
    if len(lista) < 2:
        return lista
    # Caso recursivo
    menores, medio, mayores = _partitionMI(lista,df1,df2,reverse)
    return quick_sortMI(menores,df1,df2,reverse) + medio + quick_sortMI(mayores,df1,df2,reverse)

def _partitionMI(lista,df1,df2,reverse):
    """ 
    
    Parámetros
    -----------
    lista: lista. 
    df1: DataFrame. Base de datos.
    df2: DataFrame. Columna con los ids.
    reverse:
    
    Devuelve:
    ----------- 
    tres listas: menores, medio y mayores. 
    """
    
    pivote = lista[0].copy()
    menores = []
    mayores = []
    for x in range(1, len(lista)):
        if reverse == True:
            if calculaSoporteMI(df1,df2,lista[x]) < calculaSoporteMI(df1,df2,pivote):
                menores.append(lista[x].copy())
            else:
                mayores.append(lista[x].copy())
        else :
            if calculaSoporteMI(df1,df2,lista[x]) > calculaSoporteMI(df1,df2,pivote):
                menores.append(lista[x].copy())
            else:
                mayores.append(lista[x].copy())
            
    return menores, [pivote], mayores


def ranking(patron,ranking_patrons,df,mi):
    """ 
    Comprueba para insertar un nuevo patrón en el ranking si hay alguno peor al que sustituir.
    
    Parámetros
    -----------
    patron:
    ranking_patrons:
    df: DataFrame. Base de datos.
    mi: Flag. Indica si la base de datos contiene el problema multi-instancia.
    
    Devuelve:
    -----------
    lista. La élite modifcada o no.
    """
    
    if mi:
        
        df1=df.iloc[:,1:]
        df2=df.iloc[:,:1]
        
        ranking_patrons=quick_sortMI(ranking_patrons,df1,df2,reverse=False)
        encontrado=False
        
        if calculaSoporteMI(df1,df2,patron) > calculaSoporteMI(df1,df2,ranking_patrons[-1]):            
            i=0
            while encontrado == False and i < len(ranking_patrons): 
                if calculaSoporteMI(df1,df2,patron) > calculaSoporteMI(df1,df2,ranking_patrons[i]) and patron not in ranking_patrons :
                    ranking_patrons[i]=patron.copy()
                    encontrado=True
                i+=1
        ranking_patrons=quick_sortMI(ranking_patrons,df1,df2,reverse=False) 
        
        
    else:
    
        #ordenamos según  el soporte
        ranking_patrons=quick_sort(ranking_patrons,df,reverse=False)
        encontrado=False
        
        if calculaSoporte(df,patron) > calculaSoporte(df,ranking_patrons[-1]):            
            i=0
            while encontrado == False and i < len(ranking_patrons): 
                if calculaSoporte(df,patron) > calculaSoporte(df,ranking_patrons[i]) and patron not in ranking_patrons :
                    ranking_patrons[i]=patron.copy()
                    encontrado=True
                i+=1
        ranking_patrons=quick_sort(ranking_patrons,df,reverse=False)
    
    return ranking_patrons
            
        
###############################       RANDOM SEARCH       ###############################

def generarPatronAleatorio(tamPatron,tamPatronMax):
    """
    Genera un patrón aleatorio.
    
    Parámetros
    -----------
    tamPatron: int. tamaño del patrón.
    tamPatronMax: int. Tamaño máximo del patrón.
    
    Devuelve:
    -----------
    lista. Un patrón.
    """
    
    patron=[] 
    for i in range(tamPatron):        
            n=random.randint(0,(tamPatronMax-1))            
            if n not in patron:
                patron.append(n)
    patron.sort() 
    return patron


def randomSearch(df_plus,arbol,tam_elite,multi,numMaxIterations):
    """ 
    Algoritmo de generación de patrones aleatroios Random Search.
    Parámetros
    -----------
    df_plus: DataFrame.  Base de datos.
    arbol: lista. Contiene los patrones no válidos.
    tam_elite: int. Tamaño de la lista con los patrones de la élite.
    multi: Flag por si el problema es multi-instancia.
    numMaxIterations: int. Número de iteraciones del algoritmo.
    
    Devuelve:
    -----------
    lista. Élite de patrones genereados aleatoriamente.
    """
    
    if multi :  
        df2=df_plus.iloc[:,:1] # DataFrame con id
        df_plus=df_plus.iloc[:,1:] # DataFrame con los datos     
    
    patron=[]
    elite=[]
    names = df_plus.columns.tolist()
    tamPatronMax=len(df_plus.columns.tolist())  
    poblacion=[]
    
    #hasta el número de patrones
    for i in range(numMaxIterations):

        #hasta el tamaño del patron
        tamPatron=random.randint(1,tamPatronMax)
        
        patron=generarPatronAleatorio(tamPatron,tamPatronMax)
        
        if len(patron) != 0:            
            if comprobar(patron,arbol,names) == True:
                support = 0
            else:           
                if multi:
                    support=calculaSoporteMI(df_plus,df2,patron)
                else:
                    support=calculaSoporte(df_plus,patron)
                               
            if support > 0:
                if patron not in poblacion:
                    poblacion.append(patron)
        
    #una vez finalizado seleccionamos la élite
    for patron in poblacion:
        patron.sort()
        if len(elite) >= tam_elite:
            elite=ranking(patron,elite,df_plus,multi)         
        else:             
            if patron not in elite:
                    elite.append(patron)
        
        
        
              
    return elite

    
###############################        RANDOM SEARCH BALANCEADO    ###############################
def pseudoRandomSearch(df_plus,arbol,tam_elite,multi,numMaxIterations):
    """ 
    Algoritmo de generación de patrones aleatroios balanceando 
    la probabilidad de ocurrencia.
    
    Parámetros
    -----------
    df_plus: DataFrame.  Base de datos.
    arbol: lista. Contiene los patrones no válidos.
    tam_elite: int. Tamaño de la lista con los patrones de la élite.
    multi: Flag por si el problema es multi-instancia.
    numMaxIterations: int. Número de iteraciones del algoritmo.
    
    Devuelve:
    ----------- 
    lista. Élite de patrones genereados aleatoriamente.
    """
     
    if multi :  
        df2=df_plus.iloc[:,:1]
        df_plus=df_plus.iloc[:,1:]
         
    names = df_plus.columns.tolist()
    tamPatronMax=len(df_plus.columns.tolist())
    elite=[]
    poblacion=[]
    probs=np.empty(tamPatronMax)
    probsAcumulated=np.zeros(tamPatronMax+1)
    probs.fill(0.5)
    occurencys=np.zeros(tamPatronMax)
    min_iterations= int(numMaxIterations/5)
    
    
    
    for i in range(numMaxIterations):
        tamPatron=random.randint(1,tamPatronMax)
        patron=[]
        for j in range(tamPatron):
            prob = np.random.rand()
            if i > min_iterations:
                
                for k in range(len(probsAcumulated)-1):
                    
                    if prob >= probsAcumulated[k] and prob <= probsAcumulated[k+1] and k not in patron:
                        
                        occurencys[k]+=1
                        patron.append(k)
                       
            else: #las primeras iteraciones se comporta como un patrón normal
                aux = np.random.randint(0,tamPatronMax-1)
                if prob <= probs[aux] and aux not in patron:
                    
                    occurencys[aux]+=1
                    patron.append(aux)
        
        if len(patron) != 0:          
            patron.sort()     
            if comprobar(patron,arbol,names) == True:
                support = 0
               
            else:
                if multi:
                        support=calculaSoporteMI(df_plus,df2,patron)
                else:
                        support=calculaSoporte(df_plus,patron)
                
            if support > 0:
                if patron not in poblacion:
                    poblacion.append(patron)           
                
            if i > min_iterations:
                
                #se balacea la probabilidad de ocurrencia una vez pasadas las iteraciones mínimas
                while 0 in occurencys:
                    indice = np.where(occurencys == 0)[0]
                    occurencys[indice]=1
                for i in range(tamPatronMax):
                    probs[i]=1/occurencys[i]
                total=np.sum(probs)
                for i in range(tamPatronMax):
                    probs[i]/=total
                for i in range(1,len(probsAcumulated)):
                    probsAcumulated[i]=probsAcumulated[i-1]+ probs[i-1]
                    
    
    
    for patron in poblacion:
        patron.sort()
        if len(elite) >= tam_elite:
            elite=ranking(patron,elite,df_plus,multi)         
        else:             
            if patron not in elite:
                    elite.append(patron)
    
    return elite
    
 
    
###############################       ALGORITMO EVOLUTIVO       ###############################

def generaPoblacion(df_plus,arbol,multi,tamPoblacion):
    """ 
    Generar una poblacion inicial aleatoria para el evolutivo
    
    Parámetros
    -----------
    df_plus: DataFrame.  Base de datos.
    arbol: lista. Contiene los patrones no válidos.
    multi: Flag por si el problema es multi-instancia.
    tamPoblacion: int. Tamaño de la población inicial.
    
    Devuelve:
    -----------
    lista. Poblacion de patrones aleatorios 
    """
    
    if multi :  
        df_plus=df_plus.iloc[:,1:] # DataFrame con los datos     
    
    patron=[]
    poblacion=[]
    names = df_plus.columns.tolist()
    tamPatronMax=len(df_plus.columns.tolist())  
    
    
    #hasta el número de patrones
    while len(poblacion) < tamPoblacion:

        #hasta el tamaño del patron
        tamPatron=random.randint(1,tamPatronMax)
        
        patron=generarPatronAleatorio(tamPatron,tamPatronMax)
                    
        if comprobar(patron,arbol,names) != True:
            poblacion.append(patron)
        
              
    return poblacion
        
                
def cruzar(padres,probCruce):
    """ 
    Cruzar los hijos dada una probabilidad de cruce.
    Parámetros
    -----------
    padres: lista. Contiene los patrones denominados como padres.
    probCruce: float entre 0 y 1. Probabilidad de cruce.
    
    Devuelve:
    -----------
    lista. Patrones generados a partir de los padres.
    """
   
    hijos=[]
    
    while len(hijos) < len(padres) :
        for i in range(0,len(padres),2): # se cogen padres de dos en dos para cruzarlos entre sí. 
            hijo1=padres[i].copy()
            hijo2=padres[i+1].copy()
                   
            if len(hijo1) < len(hijo2):           
                
                tamMenor=len(hijo1)
            
            else:
                tamMenor=len(hijo2)
                
            cromosoma=random.randint(0,(tamMenor-1))
            
            aux=hijo2.copy()
            
            prob=np.random.rand()
            if prob <= probCruce:
            
                if hijo1[cromosoma] not in aux:
                                 
                    hijo2[cromosoma]=hijo1[cromosoma]
                            
                if aux[cromosoma] not in hijo1:
                                   
                    hijo1[cromosoma]=aux[cromosoma]
        
            hijos.append(hijo1.copy())
            hijos.append(hijo2.copy())
   
    return hijos

def mutar(hijos,tamPatronMax,probMutacion):
    """ 
    Mutar los hijos dada una probabilidad de mutación.
    Parámetros
    -----------
    hijos: lista. Contiene los patrones denominados como hijos.
    tamPatronMax: int. Tamaño máximo del patrón.
    probMutacion: float entre 0 y 1. Probabilidad de mutación.
    
    Devuelve:
    -----------
    lista. Patrones alterados.
    """
    
    for hijo in hijos:
        prob=np.random.rand()
        if prob <= probMutacion:
            n=random.randint(1,3)        
            #añadimos
            if n == 1:
                
                if len(hijo) < tamPatronMax:
                    
                    var=False
                    while not(var):
                        element=random.randint(0,(tamPatronMax-1))
                        if element not in hijo:
                            hijo.append(element)
                            var=True
                
            #quitamos
            elif n == 2:
                
                if len(hijo) > 1:
                    pos = random.randint(0,(len(hijo)-1))          
                    hijo.pop(pos)
                          
            #cambiamos
            else:
                
                if len(hijo)>1:
                    pos = random.randint(0,len(hijo)-1)
                    
                    var=False
                    while not(var):
                        element=random.randint(0,(tamPatronMax-1))
                        if element not in hijo:
                            hijo[pos]=element
                            var=True

    return hijos


def seleccionarYRemplazar(poblacion,hijos,df,multi):
    """ 
    Seleccionar de los hijos a los mejores para reemplazar en la población
    Parámetros
    -----------
    poblacion: lista. Lista con los individuos o patrones.
    hijos: lista. Contiene los patrones denominados como hijos.
    df: DataFrame. Base de datos.
    multi: Flag por si el problema es multi-instancia.
    
    Devuelve:
    ----------- 
    lista. Población modificada
    """
    
    if multi:
        
        df1=df.iloc[:,1:]
        df2=df.iloc[:,:1]
        
        #La ordenamos poniendo el peor en la primera posicion
        poblacion=quick_sortMI(poblacion,df1,df2,reverse=True)
        #Los hijos los ordenamos de mejor a peor
        hijos=quick_sortMI(hijos,df1,df2,reverse=False)
        
        
        i=0
        cambio=0
        while cambio == i and i < len(hijos):
            
            if calculaSoporteMI(df1,df2,poblacion[i]) <  calculaSoporteMI(df1,df2,hijos[i]): 
                cambio+=1
                
                if hijos[i] not in poblacion:           
                    
                    poblacion[i]=hijos[i].copy()
                    
                
        
            i+=1
        poblacion=quick_sortMI(poblacion,df1,df2,reverse=False)   
    
    else:
        
        #La ordenamos poniendo el peor en la primera posicion
        poblacion=quick_sort(poblacion,df,reverse=True)
        #Los hijos los ordenamos de mejor a peor
        hijos=quick_sort(hijos,df,reverse=False)
        
        
        i=0
        cambio=0
        while cambio == i and i < len(hijos):
            
            if calculaSoporte(df,poblacion[i]) <  calculaSoporte(df,hijos[i]): 
                cambio+=1
                
                if hijos[i] not in poblacion:           
                    
                    poblacion[i]=hijos[i].copy()
                    
                
        
            i+=1
        poblacion=quick_sort(poblacion,df,reverse=False)
    
    return poblacion
            

def torneo(tamPoblacion,k,poblacion,df_plus):
    """ 
    Seleccionar de los hijos a los mejores para reemplazar en la población
    Parámetros
    -----------
    df_plus: DataFrame.  Base de datos.
    poblacion: lista. Lista con los individuos o patrones.
    tamPoblacion: int. Tamaño de la población inicial.
    k: int. Tamaño del torneo.

    
    Devuelve:
    ----------- 
    lista. Población modificada
    """
    
    padres=[]
    torneo=[]
    tam_padres=int(tamPoblacion/3)
    if tam_padres%2 != 0:
        tam_padres+=1
    
    
    while len(padres) < tam_padres:
        while len(torneo) < k:
            
            ele=random.randint(0,(len(poblacion)-1))
            #print(ele)
            #selecciono a los individuos para el torneo
            if poblacion[ele] not in torneo:           
                torneo.append(poblacion[ele].copy())
        #print(torneo)        
        torneo=quick_sort(torneo,df_plus,reverse=False)
        
        #selecciono al mejor individuo
        mejor=torneo[0]
                
        #voy metiendo al mejor en los padres, ¿se puede repetir?
        if mejor not in padres:
            padres.append(mejor)
            
        torneo=[]
    return padres
        

def torneoMI(tamPoblacion,k,poblacion,df,df2):
    """ 
    Seleccionar de los hijos a los mejores para reemplazar en la población.
    Opción para el problema multi-instancia.
    Parámetros
    -----------
    df: DataFrame. Base de datos.
    df2: DataFrame. Columna con los ids.
    poblacion: lista. Lista con los individuos o patrones.
    tamPoblacion: int. Tamaño de la población inicial.
    k: int. Tamaño del torneo.
    
    Devuelve:
    -----------
    lista. Población modificada
    """
    
    padres=[]
    torneo=[]
    tam_padres=int(tamPoblacion/3)
    
    if tam_padres%2 != 0:
        tam_padres+=1
    
    while len(padres) < tam_padres:
        
        
        while len(torneo) < k:
            
            ele=random.randint(0,(len(poblacion)-1))
    
            #selecciono a los individuos para el torneo
            if poblacion[ele] not in torneo:           
                torneo.append(poblacion[ele].copy())
        
        
            
        torneo=quick_sortMI(torneo,df,df2,reverse=False)
        
        
        #selecciono al mejor individuo
        mejor=torneo[0]
                
        #voy metiendo al mejor en los padres, ¿se puede repetir?
        if mejor not in padres:
            padres.append(mejor)
            
        torneo=[]
    return padres


    
def evolutivo(df_plus,arbol,tam_elite,multi,tamPoblacion,numMaxIterations,k,probCruce,probMutacion):
    """
    Algoritmo de simulación evolutiva.
    
    Parámetros
    -----------
    df_plus: DataFrame.  Base de datos.
    arbol: lista. Contiene los patrones no válidos.
    tam_elite: int. Tamaño de la lista con los patrones de la élite.
    multi: Flag por si el problema es multi-instancia.
    tamPoblacion: int. Tamaño de la población inicial.
    numMaxIterations: int. Número de iteraciones del algoritmo.
    k: int. Tamaño del torneo.
    probCruce: float entre 0 y 1. Probabilidad de cruce.
    probMutacion: float entre 0 y 1. Probabilidad de mutación.
    
    Devuelve:
    -----------
    lista. Patrones extraídos mediante programación genética.
    """

    poblacion= generaPoblacion(df_plus,arbol,multi,tamPoblacion)
    names=df_plus.columns.tolist()
    
    if multi :  
        df1=df_plus.iloc[:,1:]
        df2=df_plus.iloc[:,:1] 
        names=df1.columns.tolist()
    
    elite=[]
    tamPatronMax=len(names)
    hijos=[]
    
    
    for i in range(numMaxIterations):
        
        #Evaluamos individuos
        for indiviudo in poblacion:
            if multi:
                calculaSoporteMI(df_plus,df2,indiviudo)
            else:
                calculaSoporte(df_plus,indiviudo)
        
        #selección por torneo 
        if not multi:
            
            padres=torneo(tamPoblacion,k,poblacion,df_plus)
        else:
            padres=torneoMI(tamPoblacion,k,poblacion,df1,df2)
        
        
        #Cruce y Mutación   
        hijos=cruzar(padres,probCruce)
        hijos=mutar(hijos,tamPatronMax, probMutacion)
        poblacion=seleccionarYRemplazar(poblacion,hijos,df_plus,multi)
        
        hijos=[]
            
    
         
    for patron in poblacion:
        patron.sort()
        if not(comprobar(patron,arbol,names)):
            if len(elite) >= tam_elite:
                elite=ranking(patron,elite,df_plus,multi)         
            else:             
                if patron not in elite:
                        elite.append(patron)
    
    return elite

###############################       APRIORI       ###############################
def validar(c,names,arbol):  
    """
    Comprueba si un patrón es valido o no.
    
    Parámetros
    -----------
    c: lista. Patron a evaluar.
    names: lista. Nombres de las columnas del DataFrame.
    arbol: lista. Contiene los patrones no válidos.
    
    Devuelve:
    -----------
    bool. True or False si es no válido.
    """
     
    elements=[]
    for i in c:
        elements.append(names[i])
        
    elements=frozenset(elements)
    for nodo in arbol:      
        if frozenset.issubset(nodo,elements):        
            return True
    
    return False


def generate_new_combinations(old_combinations):
    """
    Generador de todas las combinaciones basadas en el último 
    estado de los parámetros del algoritmo Apriori
    
    """

    items_types_in_previous_step = np.unique(old_combinations.flatten())
    for old_combination in old_combinations:
        max_combination = max(old_combination)
        for item in items_types_in_previous_step:
            if item > max_combination:
                res = tuple(old_combination) + (item,)
                yield res


def aprioriDeleteIn(arbol,df,min_support,multi, use_colnames=False,max_len=None, n_jobs=1):
    """
    Extrae todos los itemsets frecuentes de una base de datos.
    Esta versión del algoritmo comprueba durante su ejecución 
    si el patrón encontrado es válido o no.
    
    Parámetros
    -----------
    df : pandas DataFrame.
    min_support : float entre 0 y 1.
    use_colnames : bool 
      Si es verdadero, utiliza los nombres de columna de los DataFrames en el DataFrame devuelto en lugar de los índices de columna.
    max_len : int (default: None)
      Longitud máxima de los conjuntos de elementos generados. 
      Si `Ninguno` (predeterminado) se evalúan todas las longitudes 
      posibles de conjuntos de elementos (bajo la condición apriori).
    
    Devuelve:
    -----------
    DataFrame pandas cuyas columnas son: ['support', 'itemsets'].
    """
    
    if multi :
        df2=df.iloc[:,:1]
        df=df.iloc[:,1:]
        
        
        
    names=df.columns.tolist()
    
    allowed_val = {0, 1, True, False}
    unique_val = np.unique(df.values.ravel())
    for val in unique_val:
        if val not in allowed_val:
            s = ('The allowed values for a DataFrame'
                 ' are True, False, 0, 1. Found value %s' % (val))
            raise ValueError(s)

    is_sparse = hasattr(df, "to_coo")
    if is_sparse:
        X = df.to_coo().tocsc()
        support = np.array(np.sum(X, axis=0) / float(X.shape[0])).reshape(-1)
    else:
        
        X = df.values
        support = (np.sum(X, axis=0) / float(X.shape[0]))

    ary_col_idx = np.arange(X.shape[1])
    support_dict = {1: support[support >= min_support]}
    itemset_dict = {1: ary_col_idx[support >= min_support].reshape(-1, 1)}
    max_itemset = 1
    rows_count = float(X.shape[0])

    if max_len is None:
        max_len = float('inf')

    while max_itemset and max_itemset < max_len:
        next_max_itemset = max_itemset + 1
        combin = generate_new_combinations(itemset_dict[max_itemset])
        frequent_items = []
        frequent_items_support = []

        if is_sparse:
            all_ones = np.ones((X.shape[0], next_max_itemset))
        for c in combin:
            
            
            if validar(c,names,arbol) == False:
                if is_sparse:
                    together = np.all(X[:, c] == all_ones, axis=1)
                else:
                    together = X[:, c].all(axis=1)
                if multi:
                    support= calculaSoporteMI(df,df2,c)
                    
                    
                else:
                    support = together.sum() / rows_count
            
            elif validar(c,names,arbol) == True:
                support=0
                
            
            if support >= min_support:
                frequent_items.append(c)
                frequent_items_support.append(support)

        if frequent_items:
            itemset_dict[next_max_itemset] = np.array(frequent_items)
            support_dict[next_max_itemset] = np.array(frequent_items_support)
            max_itemset = next_max_itemset
        else:
            max_itemset = 0

    all_res = []
    for k in sorted(itemset_dict):
        support = pd.Series(support_dict[k])
        itemsets = pd.Series([frozenset(i) for i in itemset_dict[k]])

        res = pd.concat((support, itemsets), axis=1)
        all_res.append(res)

    res_df = pd.concat(all_res)
    res_df.columns = ['support', 'itemsets']
    if use_colnames:
        mapping = {idx: item for idx, item in enumerate(df.columns)}
        res_df['itemsets'] = res_df['itemsets'].apply(lambda x: frozenset([
                                                      mapping[i] for i in x]))
    res_df = res_df.reset_index(drop=True)

    return res_df


def aprioriDeletePost(df,min_support,multi, use_colnames=False, max_len=None, n_jobs=1):
    """
    Extrae todos los itemsets frecuentes de una base de datos.
    Esta versión del algoritmo extrae todos los patrones sin comprobar
    si el patrón encontrado es válido o no.
    
    Parámetros
    -----------
    df : pandas DataFrame.
    min_support : float entre 0 y 1.
    use_colnames : bool 
      Si es verdadero, utiliza los nombres de columna de los DataFrames en el DataFrame devuelto en lugar de los índices de columna.
    max_len : int (default: None)
      Longitud máxima de los conjuntos de elementos generados. 
      Si `Ninguno` (predeterminado) se evalúan todas las longitudes 
      posibles de conjuntos de elementos (bajo la condición apriori).
    
    Devuelve:
    -----------
    DataFrame pandas cuyas columnas son: ['support', 'itemsets'].
    """
    if multi :
        df2=df.iloc[:,:1]
        df=df.iloc[:,1:]
        
    
    allowed_val = {0, 1, True, False}
    unique_val = np.unique(df.values.ravel())
    for val in unique_val:
        if val not in allowed_val:
            s = ('The allowed values for a DataFrame'
                 ' are True, False, 0, 1. Found value %s' % (val))
            raise ValueError(s)

    is_sparse = hasattr(df, "to_coo")
    if is_sparse:
        X = df.to_coo().tocsc()
        support = np.array(np.sum(X, axis=0) / float(X.shape[0])).reshape(-1)
    else:
        X = df.values
        support = (np.sum(X, axis=0) / float(X.shape[0]))

    ary_col_idx = np.arange(X.shape[1])
    support_dict = {1: support[support >= min_support]}
    itemset_dict = {1: ary_col_idx[support >= min_support].reshape(-1, 1)}
    max_itemset = 1
    rows_count = float(X.shape[0])

    if max_len is None:
        max_len = float('inf')

    while max_itemset and max_itemset < max_len:
        next_max_itemset = max_itemset + 1
        combin = generate_new_combinations(itemset_dict[max_itemset])
        frequent_items = []
        frequent_items_support = []

        if is_sparse:
            all_ones = np.ones((X.shape[0], next_max_itemset))
        for c in combin:
            if is_sparse:
                together = np.all(X[:, c] == all_ones, axis=1)
            else:
                together = X[:, c].all(axis=1)
            if multi:
                support= calculaSoporteMI(df,df2,c)
            else:
                support = together.sum() / rows_count
            
            if support >= min_support:
                frequent_items.append(c)
                frequent_items_support.append(support)

        if frequent_items:
            itemset_dict[next_max_itemset] = np.array(frequent_items)
            support_dict[next_max_itemset] = np.array(frequent_items_support)
            max_itemset = next_max_itemset
        else:
            max_itemset = 0

    all_res = []
    for k in sorted(itemset_dict):
        support = pd.Series(support_dict[k])
        itemsets = pd.Series([frozenset(i) for i in itemset_dict[k]])

        res = pd.concat((support, itemsets), axis=1)
        all_res.append(res)

    res_df = pd.concat(all_res)
    res_df.columns = ['support', 'itemsets']
    if use_colnames:
        mapping = {idx: item for idx, item in enumerate(df.columns)}
        res_df['itemsets'] = res_df['itemsets'].apply(lambda x: frozenset([
                                                      mapping[i] for i in x]))
    res_df = res_df.reset_index(drop=True)

    return res_df




def iniciar_apriori(df_plus,arbol,min_support,multi,delete):
    """
    Inicializa la versión del algoritmo apriori que se le indique.
    
    Parámetros
    -----------
    df_plus: DataFrame.  Base de datos.
    arbol: Lista. Contiene los patrones no válidos.
    min_support: float entre 0 y 1. soporte mínimo para extraer patrones.
    multi: Flag por si el problema es multi-instancia.
    delete: si es para la opción IN o POST.

    Devuelve:
    -----------
    """
    
    if delete:
        
        frequent_itemsets = aprioriDeleteIn(arbol,df_plus,min_support,multi,use_colnames=True)
    
    else:
        
        frequent_itemsets = aprioriDeletePost(df_plus,min_support,multi,use_colnames=True)

    
    return frequent_itemsets




###############################       ARBOL       ###############################             
def generar_arbol(df,jerarquia_file):        
    """
    Modifica la base de datos en base a la grmática de contexto libre introducida
    
    Parámetros
    -----------
    df: DataFrame. Base de datos.
    jerarquia_file: string. Gramática de contetxo libre

    Devuelve:
    -----------
    lista. Lista con los patrones no válidos y DataFrame modificado.
    """
    
    results=[]
    with open(jerarquia_file, newline='') as inputfile:
        for line in inputfile:
            results.append(line.rstrip('\n'))
    
    #En este bucle se separan el nombre de las columnas de sus subtipos, 
    #creando listas de dos elementos
    col=[]
    for r in reversed(results):   
        col.append(r.split("->"))
    
    nodo=[]
    arbol=[]  
    
    for mensaje in col:
        
        
        datos=mensaje[1] #subtipo de la columna 
        
        column=mensaje[0] #nombre de la columna
         
        #eliminamos los espacios en blanco
        datos=datos.replace(" ", "")
        column=column.replace(" ", "")
        
       
        #se contempla dos casos iniciales:
        #       - que haya una OR en la parte de subtipos 
        #       - que no haya ninguna OR en la parte de subtipos
        if "|" in datos:
            
            datos=datos.split('|')
            names = df.columns.tolist()
           
            if len(datos) >= 2:
                
                if column != "empty":
                    
                    df['name']= ([0] * len(df.index))             
                    names = df.columns.tolist() 
                
                    for dato in datos:
                        
                        if "," in dato:
                                                   
                            dato2=dato.split(",")
                            aux = ([1] * len(df.index)) 
                            for dato3 in dato2:
                                aux= aux & df[names[names.index(dato3)]]
                            
                            nodo.append(column)
                            [nodo.append(dat) for dat in dato2]
                            
                            arbol.append(nodo)
                            nodo=[]
                            
                            names = df.columns.tolist() 
                            
                            df['name'] = df['name'] | aux
                        
                        elif "," not in dato:
                            
                            df['name'] = df['name'] | df[names[names.index(dato)]]
                            
                            nodo.append(str(column))
                            nodo.append(str(dato))
                            arbol.append(nodo)
                            nodo=[]
                         
                            
                        
                    
                    names = df.columns.tolist() 
                    
                    names[names.index('name')] = column
                    df.columns = names
                    
                    datos=[]
                    column=[]
                    
        #como solo consta de un elemento la nueva columna a insertar, 
        #se copia los valores correspondientes a la nueva columna    
        elif "|" not in datos:
            
            if "," in datos:
                
                df['name']= ([0] * len(df.index))             
                
                names = df.columns.tolist() 
                
                dato2=datos.split(",")
                
                    
                aux = ([1] * len(df.index)) 
                for dato3 in dato2:
                    aux= aux & df[names[names.index(dato3)]]
                
                nodo.append(column)               
                [nodo.append(dat) for dat in dato2]               
                arbol.append(nodo) 
                
                nodo=[]  
                
                df['name'] = df['name'] | aux
                
                
                names[names.index('name')] = column
                df.columns = names
            
            elif "," not in datos:
                
                names = df.columns.tolist()
            
                df=df.assign(name= df.loc[:, names[names.index(datos)]:names[names.index(datos)]] )
            
                names = df.columns.tolist()
                names[names.index('name')] = column
                df.columns = names
                nodo.append(column)
                nodo.append(str(datos))
                arbol.append(nodo)
                nodo=[]
    
    
    
    #Hacemos ultimo cambio
    aux=[]
    for nodo in arbol:
        for nodo2 in arbol:
            
            if len(nodo)==2:           
                if nodo[1] == nodo2[0]:
                    aux.append(nodo[0])
                    aux.append(nodo2[1])
                    
                    if aux not in arbol:                
                        arbol.append(aux)
                    
                    aux=[] 
    #lo pasamos a frozenset    
    arbol2=[]      
    for nodo in arbol:
        arbol2.append(frozenset(nodo))
    
    
    
    return arbol2, df


def borrar_elementos(arbol,frequent_itemsets):
    """
    Borra los patrones no válidos del Apriori POST
    
    Parámetros
    -----------
    arbol: Lista. Contiene los patrones no válidos.
    frequent_itemsets: DataFrame. Contiene patrones extraidos por el algoritmo Apriori POST

    Devuelve:
    -----------
    """
    

    elements=[]    
    for nodo in arbol:
        for i in range(0, len(frequent_itemsets)):
            nodo2=frequent_itemsets.iloc[i][1]       
            if frozenset.issubset(nodo,nodo2) == True:         
                if i not in elements:
                    elements.append(i)
            
               
    elements.sort()     
    for ele in elements:
        frequent_itemsets=frequent_itemsets.drop([int(ele)],axis=0)

    return frequent_itemsets


def ejecutar_aprioriPost(df_plus,arbol,min_support,multi):
    
    
    frequent_itemsets=iniciar_apriori(df_plus,arbol,min_support,multi,delete=False)
    frequent_itemsets=borrar_elementos(arbol,frequent_itemsets)
    
    return frequent_itemsets


def ejecutar_aprioriIn(df_plus,arbol,min_support,multi):
  
    frequent_itemsets=iniciar_apriori(df_plus,arbol,min_support,multi,delete=True)
    
    return frequent_itemsets



    
   
if __name__ == "__main__":
    extraer_patrones() 
    
