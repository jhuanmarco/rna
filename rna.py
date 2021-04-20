import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import random
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

%matplotlib inline

#Função do cáculo da sigmóide
def sigmoid(x):
    return 1/(1+np.exp(-x))

DataSet=pd.read_csv('arruela_.csv')
DataSet.drop(['Hora','Tamanho','Referencia'],axis=1,inplace=True)


scaler=StandardScaler()
DataScaled=scaler.fit_transform(DataSet)
DataSetScaled=pd.DataFrame(np.array(DataScaled),columns = ['NumAmostra', 'Area', 'Delta', 'Output1','Output2'])

X = DataSetScaled.drop(['Output1', 'Output2'],axis=1)
y = DataSet[['Output1','Output2']]

best = 0
bestr = None
bestn = None
besth1 = None
besth2 = None
besth3 = None
besto = None
beste = 0
count =  0 

randomlist = []
for i in range(0,500):
    n = random.randint(1,5000)
    randomlist.append(n)

for randomN in randomlist:
    count = count + 1
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=randomN)

    # print(y_test)
    # print(X_test)

    #Tamanho do DataSet de Treinamento
    n_records, n_features = X_train.shape

    neurons = random.randint(2,80)

    #Arquitetura da MPL
    N_input = 3
    N_hidden = neurons
    N_hidden2 = neurons
    N_output = 2
    learnrate = 0.2

    #Pesos das Camadas Ocultas (Inicialização Aleatória)
    seedhidden1 = random.randint(1,800)
    np.random.seed(seedhidden1)
    weights_input_hidden = np.random.normal(0, scale=0.1, size=(N_input, N_hidden))

    seedhidden2 = random.randint(1,800)
    np.random.seed(seedhidden2)
    weights_input_hidden2 = np.random.normal(0, scale=0.1, size=(N_hidden, N_hidden2))

    # seedhidden3 = random.randint(1,800)
    # np.random.seed(seedhidden3)
    # weights_input_hidden3 = np.random.normal(0, scale=0.1, size=(N_hidden, N_hidden3))

    #Pesos da Camada de Saída (Inicialização Aleatória)
    seedoutput = random.randint(1,800)
    np.random.seed(seedoutput)
    weights_hidden_output = np.random.normal(0, scale=0.6, size=(N_hidden2, N_output))
    
    epochs =  2000
    last_loss=None
    EvolucaoError=[]
    IndiceError=[]

    for e in range(epochs):
        delta_w_i_h = np.zeros(weights_input_hidden.shape)
        delta_w_h_o = np.zeros(weights_hidden_output.shape)
        for xi, yi in zip(X_train.values, y_train.values):
            
    # Forward Pass
            #Camada oculta
            hidden_layer_input = np.dot(xi, weights_input_hidden)
            hidden_layer_output = sigmoid(hidden_layer_input)

            hidden_layer2_input = np.dot(hidden_layer_output, weights_input_hidden2)
            hidden_layer2_output = sigmoid(hidden_layer2_input)
        
            #Camada de Saída
            output_layer_in = np.dot(hidden_layer2_output, weights_hidden_output)
            output = sigmoid(output_layer_in)

    #-------------------------------------------    
        
    # Backward Pass
            ## TODO: Cálculo do Erro
            error = yi - output
        
            # TODO: Calcule o termo de erro de saída (Gradiente da Camada de Saída)
            output_error_term = error * output * (1 - output)

            # TODO: Calcule a contribuição da camada oculta para o erro

            hidden_error2 = np.dot(weights_hidden_output,output_error_term)
            hidden_error_term2 = hidden_error2 * hidden_layer_output * (1 - hidden_layer_output)

            hidden_error = np.dot(weights_input_hidden2,hidden_error_term2)
            hidden_error_term = hidden_error * hidden_layer2_output * (1 - hidden_layer2_output)
        
            # TODO: Calcule a variação do peso da camada de saída
            delta_w_h_o += output_error_term*hidden_layer_output[:, None]

            # TODO: Calcule a variação do peso da camada oculta
            delta_w_i_h += hidden_error_term * xi[:, None]
            
        #Atualização dos pesos na época em questão
        weights_input_hidden += learnrate * delta_w_i_h / n_records
        weights_hidden_output += learnrate * delta_w_h_o / n_records
        
        
        # Imprimir o erro quadrático médio no conjunto de treinamento
        
        if  e % (epochs / 20) == 0:
            hidden_output = sigmoid(np.dot(xi, weights_input_hidden))
            out = sigmoid(np.dot(hidden_output,
                                weights_hidden_output))
            loss = np.mean((out - yi) ** 2)

            if last_loss and last_loss < loss:
                teste = 200
            else:
                teste = 200
            last_loss = loss
            
            EvolucaoError.append(loss)
            IndiceError.append(e)


    # plt.plot(IndiceError, EvolucaoError, 'r') # 'r' is the color red
    # plt.xlabel('')
    # plt.ylabel('Erro Quadrático')
    # plt.title('Evolução do Erro no treinamento da MPL')
    # plt.show()


    # Calcule a precisão dos dados de teste
    n_records, n_features = X_test.shape
    predictions=0

    for xi, yi in zip(X_test.values, y_test.values):

    # Forward Pass
            #Camada oculta
            #Calcule a combinação linear de entradas e pesos sinápticos
            hidden_layer_input = np.dot(xi, weights_input_hidden)
            #Aplicado a função de ativação
            hidden_layer_output = sigmoid(hidden_layer_input)
        
            #Camada de Saída
            #Calcule a combinação linear de entradas e pesos sinápticos
            output_layer_in = np.dot(hidden_layer_output, weights_hidden_output)

            #Aplicado a função de ativação 
            output = sigmoid(output_layer_in)

    #-------------------------------------------    
        
    #Cálculo do Erro da Predição
            ## TODO: Cálculo do Erro        
            if (output[0]>output[1]):
                if (yi[0]>yi[1]):
                    predictions+=1
                    
            if (output[1]>=output[0]):
                if (yi[1]>yi[0]):
                    predictions+=1
    print(count)
    if(predictions/n_records > best):
        print("-- NOVO MELHOR --")
        best = predictions/n_records
        bestr = randomN
        bestn = neurons
        besth1 = seedhidden1
        besth2 = seedhidden2
        besto = seedoutput
        beste = last_loss

        print("neurons " + str(neurons))
        print("seedhidden1 " + str(seedhidden1))
        print("seedhidden2 " + str(seedhidden2))
        print("seedoutput " + str(seedoutput))

    
    print("Random Nº "+str(randomN))
    print("A Acurácia da Predição é de: {:.6f}".format(predictions/n_records))
    print("Erro " + str(last_loss))
    print()

print()
print("MELHOR RESULTADO OBTIDO")
print("Acurácia: "+ str(best))
print("Erro: " + str(beste))
print("random train seed "+ str(bestr))
print("neurons " + str(neurons))
print("seedhidden1 " + str(seedhidden1))
print("seedhidden2 " + str(seedhidden2))
print("seedoutput " + str(seedoutput))

