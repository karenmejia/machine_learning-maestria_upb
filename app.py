import streamlit as st
import pandas as pd
import joblib
import unicodedata

st.set_page_config(layout="wide")

def clean_special_characters(df):
    """
    Limpia caracteres especiales (como tildes), reemplaza espacios por guiones bajos y elimina signos de interrogación
    de todas las columnas de tipo string y de los nombres de las columnas en un DataFrame.
    """
    df_cleaned = df.copy()

    # Limpiar nombres de columnas
    new_columns = []
    for col in df_cleaned.columns:
        cleaned_col = unicodedata.normalize('NFKD', str(col)).encode('ascii', 'ignore').decode('utf-8')
        cleaned_col = cleaned_col.replace(' ', '_')  # Reemplazar espacios por guiones bajos
        cleaned_col = cleaned_col.replace('?', '')   # Eliminar signos de interrogación
        new_columns.append(cleaned_col)
    df_cleaned.columns = new_columns

    # Limpiar valores en columnas de tipo string
    for col in df_cleaned.select_dtypes(include=['object']).columns:
        df_cleaned[col] = df_cleaned[col].apply(lambda x: unicodedata.normalize('NFKD', str(x)).encode('ascii', 'ignore').decode('utf-8') if pd.notna(x) else x)
    return df_cleaned

def preprocess_new_data(new_raw_df, ohe_path='one_hot_encoder.joblib', le_path='label_encoder.joblib'):
    # Cargar encoders
    try:
        ohe = joblib.load(ohe_path)
        le = joblib.load(le_path)
    except FileNotFoundError:
        st.error(f"Error: No se pudieron cargar los archivos de los encoders. Asegúrate de que '{ohe_path}' y '{le_path}' existan.")
        return None, None

    # 1. Limpieza de nombres de columnas y valores
    cleaned_df = clean_special_characters(new_raw_df.copy())

    processed_df = cleaned_df.copy() # Usar la copia limpia como punto de partida

    # 2. Eliminar columnas que no deben ser consideradas por el modelo
    columns_to_drop_for_prediction = ['Nombre', 'Genero', 'Edad', 'A_que_grupo_poblacional_pertences']
    processed_df = processed_df.drop(columns=columns_to_drop_for_prediction, errors='ignore')

    # 3. Definir características categóricas para OHE
    # Las columnas restantes después de eliminar las especificadas son las características categóricas.
    categorical_features_for_ohe = [col for col in processed_df.columns]

    # 4. Asegurarse de que las columnas identificadas sean de tipo string para el OHE
    for col in categorical_features_for_ohe:
        processed_df[col] = processed_df[col].astype('category')

    # 5. Transformar con OneHotEncoder cargado
    encoded_data = ohe.transform(processed_df[categorical_features_for_ohe])

    # 6. Crear DataFrame con nombres de columnas correctos (los que el modelo espera)
    # Asegurarse de que las columnas del DataFrame codificado coincidan con las características esperadas por el modelo.
    # Esto es crucial para evitar errores si las características de entrada no coinciden exactamente.
    expected_features = ohe.get_feature_names_out(categorical_features_for_ohe)
    encoded_df = pd.DataFrame(encoded_data, columns=expected_features)

    return encoded_df, le # Retornamos el le por si se necesita para decodificar

def predict_new_data(new_raw_df, model_path='best_logistic_regression_model.joblib'):
    try:
        model = joblib.load(model_path)
    except FileNotFoundError:
        st.error(f"Error: No se pudo cargar el archivo del modelo. Asegúrate de que '{model_path}' exista.")
        return None

    processed_features, label_encoder = preprocess_new_data(new_raw_df)

    if processed_features is None or label_encoder is None:
        return None

    predictions_numeric = model.predict(processed_features)
    predictions_decoded = label_encoder.inverse_transform(predictions_numeric)
    return predictions_decoded


# Streamlit App Interface
st.title('Predicción de nivel de apropiación TIC para la comunidad de Puerto Gaitan')

st.header('Introduce los datos del individuo para predecir su nivel TIC')

with st.form("prediction_form"):
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        nombre = st.text_input('Nombre', '')
        edad = st.selectbox('Edad', ['0 -18', '18-24', '25-34', '35-44', '45-54', '55-64', '65+'])
        genero = st.selectbox('Genero', ['Masculino', 'Femenino', 'Otros'])

    with col2:
        dispositivos = st.selectbox('¿Con qué dispositivos tecnológicos te conectas regularmente?', 
                                    ['Celular', 'Computador', 'Tablet'])
        uso_internet = st.selectbox('¿Mayormente utilizas internet en tu vida diaria para?', 
                                      ['Estudio', 'Trabajo', 'Redes sociales', 'Compras en linea', 'Plataformas de streaming'])      
        formacion_tic = st.selectbox('¿Has recibido formación formal sobre el uso de Tecnologías de la Información y Comunicaciones?', 
                                     ['Ninguno', 'Autoaprendizaje', 'Talleres o cursos'])
    with col3:
        barreras = st.selectbox('¿Cuáles son las principales barreras que enfrentas para utilizar las Tecnologías de la Información y Comunicaciones?', 
                                      ['Falta de acceso', 'Ninguna', 'Falta de habilidades', 'Poco interés'])
        informacion = st.selectbox('¿Te gustaría recibir formación sobre las Tecnologías de la Información y Comunicaciones?', 
                                      ['Si', 'No', 'Tal vez'])  
        frecuencia = st.selectbox('¿Con qué frecuencia accedes a internet?', 
                                      ['Diariamente', 'Escasamente', 'Semanalmente'])         

    with col4:
        importancia_tic = st.selectbox('¿Qué tan importante consideras el uso de Tecnologías de la Información y Comunicaciones en tu vida diaria?', 
                                         ['Muy importante', 'Importante', 'Poco importante'])
        nivel_educativo = st.selectbox('¿Cuál es tu nivel educativo?', 
                                       ['Primaria', 'Secundaria', 'Técnica o tecnólogo', 'Pregrado', 'Postgrado', 'Sin educación formal'])
        grupo_poblacional = st.selectbox('¿A qué grupo poblacional perteneces?', 
                                          ['Ninguno', 'Indígena', 'Afrocolombiana', 'Campesino', 'Raizal', 'Rom (gitanos)'])

    submitted = st.form_submit_button('Predecir nivel de habilidad para utilizar TIC')

    if submitted:
        # Create a dictionary from user inputs
        user_data = {
            'Nombre': nombre,
            'Edad': edad,
            'Genero': genero,
            '¿Con qué dispositivos tecnológicos te conectas regularmente?': dispositivos,
            '¿Mayormente utilizas internet en tu vida diaria para': uso_internet,
            '¿Has recibido formación formal sobre el uso de Tecnologías de la Información y Comunicaciones?': formacion_tic,
            '¿Cuáles son las principales barreras que enfrentas para utilizar las Tecnologías de la Información y Comunicaciones?': barreras,
            '¿Te gustaría recibir formación sobre las Tecnologías de la Información y Comunicaciones?': informacion,
            '¿Con qué frecuencia accedes a internet?': frecuencia,
            '¿Qué tan importante consideras el uso de Tecnologías de la Información y Comunicaciones en tu vida diaria?': importancia_tic,
            '¿Cuál es tu nivel educativo?': nivel_educativo,
            '¿A que grupo poblacional pertences?': grupo_poblacional
        }

        # Convert to DataFrame
        user_df = pd.DataFrame([user_data])

        # Make prediction
        prediction = predict_new_data(user_df)

        if prediction is not None:
            st.success(f"El nivel de apropiación TIC predicho para {nombre} es: **{prediction[0]}**")
        else:
            st.error("No se pudo realizar la predicción. Por favor, verifica los datos de entrada o la disponibilidad del modelo.")

