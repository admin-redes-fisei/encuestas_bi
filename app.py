from flask import Flask, request, jsonify
import pandas as pd
import requests
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules
from flask_cors import CORS, cross_origin
from sklearn.tree import DecisionTreeClassifier
import itertools
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.utils import shuffle

app = Flask(__name__)
CORS(app)

@app.route('/', methods=['GET'])
def ping():
    try:
        response = requests.get('https://www.google.com', verify=False)
        print("Google Response Status Code:", response.status_code)
        return jsonify({"response": "Google Response Status Code: " + str(response.status_code)})
    except requests.exceptions.RequestException as e:
        print("Request failed:", e)
        return jsonify({"response": "Request failed: " + str(e)})

@app.route('/calcular_regla', methods=['POST'])
@cross_origin()
def calcular_regla():
    # Suponiendo que request.json contiene el JSON enviado por el cliente
    json_data = request.json

    # Convertir la lista de valores en una tupla
    valores = tuple([v for v in json_data["valores"] if v.strip()])  # Eliminar valores en blanco
    headers = tuple([v for v in json_data["headers"] if v.strip()])  # Eliminar valores en blanco
    formulario_id = json_data["formulario_id"]
    target = json_data["target"]
    result_dict = {header: value for header, value in zip(headers, valores)}

    # Hacer una solicitud GET al archivo PHP
    try:
        response = requests.get(f'https://host.docker.internal/encuestas/encuestas_back/obtenerDatasetApriori.php?formulario_id={formulario_id}', verify=False)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        return jsonify({'error': str(e)}), 500
    
    # Comprobar si la solicitud fue exitosa
    if response.status_code == 200:
        # Convertir la respuesta JSON en un DataFrame de pandas
        data = response.json()
        df = pd.DataFrame(data)

        #obtener clase
        clasePredicha = predecirClase(df, target, result_dict)
        # Obtener reglas de asociacion
        filtered_rules = obtenerReglas(df, valores)

        if not filtered_rules.empty:
            # Ordenar las reglas por lift en orden descendente y tomar las primeras 10
            top_rules = filtered_rules.sort_values(by='lift', ascending=False).head(10)
            top_rules_data = {
                'reglas': [{"antecedentes": list(rule['antecedents']), "consecuentes": list(rule['consequents'])} for _, rule in top_rules.iterrows()],
                'soporte': top_rules['support'].tolist(),
                'confianza': top_rules['confidence'].tolist(),
                'lift': top_rules['lift'].tolist()
            }

            respuesta = {"clasificacion": clasePredicha, "asociacion":top_rules_data}
        else:
            respuesta = {"clasificacion": clasePredicha, "asociacion":{'error': 'No se encontraron reglas'}}
            
        return jsonify(respuesta)

    else:
        return jsonify({'error': f'Error: {response.status_code}'})
    
def obtenerReglas(df, valores):
    # Separar los campos unidos por "|"
    for col in df.columns:
        # Dividir cada columna en varias columnas
        split_data = df[col].str.split('|', expand=True)
        split_data = split_data.apply(lambda x: x.str.strip())
        # Reemplazar la columna original con las nuevas columnas
        df = df.drop(col, axis=1)
        df = pd.concat([df] + [split_data[i].rename(f'{col}_{i}') for i in range(split_data.shape[1])], axis=1)

    # Reemplazar NaN por una cadena vacía
    df = df.fillna('')
        
    # Codificar las transacciones con TransactionEncoder
    te = TransactionEncoder()
    te_ary = te.fit(df.values.tolist()).transform(df.values.tolist())
    df = pd.DataFrame(te_ary, columns=te.columns_)

    # Calcular los conjuntos de ítems frecuentes con apriori
    frequent_itemsets = apriori(df, min_support=0.1, use_colnames=True)

    # Calcular las reglas de asociación con association_rules
    rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1)

    # Filtrar las reglas para obtener las que involucran los valores especificados y no tengan valores en blanco
    filtered_rules = rules[(rules['antecedents'].apply(lambda x: set(valores).issubset(x))) &
                           (rules['antecedents'].apply(lambda x: all(v.strip() for v in x))) &
                            (rules['consequents'].apply(lambda x: all(v.strip() for v in x)))]

    return filtered_rules

def predecirClase(df, target, valores):
    features = generate_combinations(df)

    # Convertir la lista de combinaciones en un DataFrame
    features_df = pd.DataFrame(features[1:], columns=features[0])
    features_df_shuffled = shuffle(features_df, random_state=42)

    X = features_df_shuffled.drop(target, axis=1)
    y = features_df_shuffled[target]

    # Dividir los datos en conjuntos de entrenamiento y prueba
    #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Definir la transformación de columnas categóricas a variables dummy (one-hot encoding)
    transformer = ColumnTransformer(
        transformers=[('encoder', OneHotEncoder(), [col for col in X.columns])],
        remainder='passthrough'
    )

    # Crear un pipeline con la transformación y el modelo de árbol de decisión
    model = Pipeline(steps=[
        ('preprocessor', transformer),
        ('classifier', DecisionTreeClassifier())
    ])
    # Entrenar el modelo
    model.fit(X, y) 

     # Entrenar el modelo
    #model.fit(X_train, y_train)

    # Calcular y imprimir el porcentaje de exactitud
    #train_accuracy = model.score(X_train, y_train)
    #test_accuracy = model.score(X_test, y_test)

    #print(f"Precisión de entrenamiento: {train_accuracy * 100:.2f}%")
    #print(f"Precisión de prueba: {test_accuracy * 100:.2f}%")

    for col in X.columns:
        if col not in valores or pd.isnull(valores[col]):
            max_gain_value = X[col].value_counts().idxmax()
            valores[col] = max_gain_value

    # Clasificación de una nueva instancia
    new_instance = valores  # Modifica los valores según tu instancia
    new_instance_df = pd.DataFrame([new_instance])

    predicted_class = model.predict(new_instance_df)[0]

    return {"instancias": new_instance, "clase": predicted_class}


def generate_combinations(matrix):
    headers = list(matrix.columns)  # Obtener los nombres de las columnas como encabezados
    data = matrix.to_numpy().tolist()  # Convertir el DataFrame a una lista de listas
    new_matrix = [headers]

    for row in data:
        # Comprobar si la celda es None antes de aplicar split('|')
        split_values = [cell.split('|') if cell is not None else '' for cell in row]
        for combination in itertools.product(*split_values):
            new_matrix.append(list(combination))
    
    return new_matrix

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, debug=True)
