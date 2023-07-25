import streamlit as st
import io
import pandas as pd
import matplotlib.pyplot as plt
def main():
        st.title("Algoritmo de Regresión Lineal")

        # Cargar el archivo CSV desde el mismo directorio que main.py
        archivo_csv = "registro.csv"  # Reemplaza "registro.csv" con el nombre real de tu archivo CSV
        datos_sec = pd.read_csv(archivo_csv, encoding='latin-1', on_bad_lines='skip')
        st.write("Datos cargados:")
        st.write(datos_sec)

        # Realiza las visualizaciones de datos y gráficos
        st.write("Aqui podremos saber cuantos datos no existentes o espacios vacios hay por columna:")
        st.write(datos_sec.isna().sum())
        
        # Eliminar columnas que no aportan al objetivo principal (predecir la cant. de participantes)
        datos_sec = datos_sec.drop(columns=['CENTRO POBLADO'])
        datos_sec = datos_sec.drop(columns=['COMUNIDAD CAMPESINA O NATIVA'])
        datos_sec = datos_sec.drop(columns=['DEPARTAMENTO'])
        datos_sec = datos_sec.drop(columns=['NOMBRE DEL EVENTO'])
        datos_sec = datos_sec.drop(columns=['TRIMESTRE'])
        datos_sec = datos_sec.drop(columns=['PROVINCIA'])
        datos_sec = datos_sec.drop(columns=['DISTRITO'])
        
        st.write("Datos limpios sin columnas irrelevantes:")
        st.write(datos_sec)
        
        # Realiza las visualizaciones de datos y gráficos
        st.write("Podemos visualizar datos específicos:")
        st.write(datos_sec["FEMENINO"])
        
        st.write("Información de los datos cargados:")
        st.write(datos_sec.info())


        
        st.write("Información de los datos cargados:")
        st.write(datos_sec.info())

        st.write("Histogramas para la visualización de los datos:")
        st.write(datos_sec.hist())

        st.write("Nueva tabla con solo los datos relevantes para el objetivo:")
        st.write(datos_sec)
        
        st.write("Grafico de dispersión (FEMENINO Vs CANT. DE PARTICIPANTES):")
        fig, ax = plt.subplots()
        ax.scatter(x=datos_sec['FEMENINO'], y=datos_sec['CANT. DE PARTICIPANTES'])
        plt.title('FEMENINO Vs CANT. DE PARTICIPANTES')
        plt.xlabel('FEMENINO')
        plt.ylabel('CANT. DE PARTICIPANTES')
        st.pyplot(fig)
        
        st.write("Grafico de dispersión (MASCULINO Vs CANT. DE PARTICIPANTES):")
        fig, ax = plt.subplots()
        ax.scatter(x=datos_sec['MASCULINO'], y=datos_sec['CANT. DE PARTICIPANTES'])
        plt.title('MASCULINO Vs CANT. DE PARTICIPANTES')
        plt.xlabel('MASCULINO')
        plt.ylabel('CANT. DE PARTICIPANTES')
        st.pyplot(fig)


        st.write("FASE 2. Entrenamiento del modelo")
        datos_entrenamiento = datos_sec.sample(frac=0.8, random_state=0)
        datos_test = datos_sec.drop(datos_entrenamiento.index)
        st.write("Datos de entrenamiento:")
        st.write(datos_entrenamiento)
        
        st.write("Datos de test:")
        st.write(datos_test)
        
        st.write("Quitar columna 'CANT. DE PARTICIPANTES' de datos de entrenamiento y test")
        etiquetas_entrenamiento = datos_entrenamiento.pop('CANT. DE PARTICIPANTES')
        etiquetas_test = datos_test.pop('CANT. DE PARTICIPANTES')
        st.write("Datos de entrenamiento sin columna 'CANT. DE PARTICIPANTES':")
        st.write(datos_entrenamiento)
        st.write("Datos de test sin columna 'CANT. DE PARTICIPANTES':")
        st.write(datos_test)
        
          # FASE 3. PREDICCIONES
        st.write("FASE 3. PREDICCIONES")


        from sklearn.linear_model import LinearRegression
        modelo = LinearRegression()
        modelo.fit(datos_entrenamiento, etiquetas_entrenamiento)

        predicciones = modelo.predict(datos_test)

        st.write("Predicciones:")
        st.write(pd.DataFrame(predicciones, columns=['value']))
        
        # Aqui calculamos el margen de error en la prediccion de la CANT. DE PARTICIPANTES.
        import numpy as np
        from sklearn.metrics import mean_squared_error
        error = np.sqrt(mean_squared_error(etiquetas_test, predicciones))
        st.write("Error porcentual:", error * 100)



      
        from sklearn.ensemble import RandomForestRegressor
        
        modelo_rf = RandomForestRegressor()
        modelo_rf.fit(datos_entrenamiento, etiquetas_entrenamiento)
        
        predicciones_rf = modelo_rf.predict(datos_test)
        st.write("Predicciones (Random Forest):")
        st.write(predicciones_rf)
        
        # Calculamos el margen de error en la predicción de la CANT. DE PARTICIPANTES
        error_rf = np.sqrt(mean_squared_error(etiquetas_test, predicciones_rf))
        st.write("Error porcentual (Random Forest):", error_rf * 100)


        modelo_gb = GradientBoostingRegressor()
        modelo_gb.fit(datos_entrenamiento, etiquetas_entrenamiento)

        predicciones_gb = modelo_gb.predict(datos_test)
        st.write("Predicciones (Gradient Boosting Regressor):")
        st.write(predicciones_gb)

        error_gb = np.sqrt(mean_squared_error(etiquetas_test, predicciones_gb))
        st.write("Error porcentual (Gradient Boosting Regressor): %f" % (error_gb * 100))

if __name__ == "__main__":
    main()
