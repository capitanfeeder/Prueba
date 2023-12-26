"""
FUNCIONES CREADAS PARA EL PROYECTO FINAL DE DATA SCIENCE DE SOY HENRY
                            - NY TAXIS - 

FUNCIONES PARA ALIMENTAR LA API
"""


# Importar las librerías necesarias
from fastapi import HTTPException
from fastapi.responses import HTMLResponse
from fastapi import FastAPI
import pandas as pd
import pyarrow.parquet as pq
from pydantic import BaseModel
from datetime import timedelta
import joblib

# Instanciar la aplicación FastAPI
app = FastAPI()

# Cargar el modelo entrenado desde el archivo pickle
xgb_model = joblib.load('Datasets/xgb_model.pkl')

# Definir la estructura de datos de entrada utilizando Pydantic
class InputData(BaseModel):
    pBoroughID: int
    dayofweek: int
    hour: int

    # Puedes agregar más campos según sea necesario

# Ruta para la página de inicio
@app.get("/", response_class=HTMLResponse)
async def inicio():
    """
    Página de inicio de la API Taxis.

    Realice sus consultas.
    """
    template = """
    <!DOCTYPE html>
    <html>
        <head>
            <title>API Taxi</title>
            <style>
                body {
                    font-family: Arial, sans-serif;
                    padding: 20px;
                }
                h1 {
                    color: #333;
                    text-align: center;
                }
                p {
                    color: #666;
                    text-align: center;
                    font-size: 18px;
                    margin-top: 20px;
                }
            </style>
        </head>
        <body>
            <h1>API de consultas predictivas sobre viajes en Taxi</h1>
            <p>Bienvenido a la API de Taxis. Utiliza la ruta <strong>/predict</strong> para realizar predicciones.</p>
        </body>
    </html>
    """
    return HTMLResponse(content=template)

# Ruta para realizar predicciones
@app.post("/predict")
def predict(data: InputData):
    """
    Endpoint para realizar predicciones.

    Permite realizar predicciones basadas en datos de entrada proporcionados en el cuerpo de la solicitud.

    Parameters:
    - data: Datos de entrada en formato JSON. Debe incluir campos específicos:
        - pBoroughID: Identificador del distrito de recogida (pickup borough). Debe ser un número entero del 1 al 6, donde:
            - 1: Manhattan
            - 2: Bronx
            - 3: Brooklyn
            - 4: Queens
            - 5: Staten Island
            - 6: EWR (Newark Airport)
        - dayofweek: Número que representa el día de la semana. Debe ser un número entero del 1 al 7, donde:
            - 1: Lunes
            - 2: Martes
            - 3: Miércoles
            - 4: Jueves
            - 5: Viernes
            - 6: Sábado
            - 7: Domingo
        - hour: Hora del día. Debe ser un número entero del 0 al 23, representando las horas en formato de 24 horas.

    Returns:
        - JSON con el resultado de la predicción indicando la probabilidad de obtener clientes o un mensaje de error si ingresó un dato erróneo.
    """
    try:
        # Verificar si los datos cumplen con los criterios especificados
        if not (1 <= data.pBoroughID <= 6 and 1 <= data.dayofweek <= 7 and 0 <= data.hour <= 23):
            raise HTTPException(status_code=422, detail="Error, un dato fue ingresado de forma errónea. Por favor verifica que lo ingresado concuerde con los parámetros indicados.")

        # Convertir los datos de entrada a un DataFrame
        input_data = pd.DataFrame([data.model_dump()])

        # Realizar la predicción
        prediction = xgb_model.predict(input_data)

        # Devolver la predicción como JSON
        return {"Probabilidad de conseguir pasajero": prediction.tolist()}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))





# Nueva ruta para obtener el top 3 de vehículos ecológicos y económicos
@app.get("/top_3_vehicles")
def top_3_vehicles(max_price_usd: float):
    """
    Endpoint para obtener el top 3 de vehículos ecológicos de acuerdo al capital.

    Parameters:
    - max_price_usd: Precio máximo en dólares para filtrar los vehículos.

    Returns:
        - JSON con el top 3 de vehículos ecológicos.
    """
    try:
        # Cargar el DataFrame desde el archivo parquet
        vehicles_info = pd.read_parquet('Datasets/vehicles_info.parquet')  

        # Filtrar los vehículos por precio
        filtered_vehicles = vehicles_info[vehicles_info['Price (USD)'] <= max_price_usd]

        # Ordenar los vehículos por CO2 en orden ascendente, luego por Sound Emission y finalmente por Range
        sorted_vehicles = filtered_vehicles.sort_values(by=['CO2 Emission (g/mi)', 'Sound Emission (dB)', 'Range (mi)'], ascending=[True, True, False])

        # Tomar los tres primeros vehículos dentro del rango de precio
        top_3_vehicles = sorted_vehicles.head(3)

        # Crear el formato de salida como una lista de diccionarios
        output_format = []
        for idx, row in enumerate(top_3_vehicles.itertuples(), start=1):
            output_format.append({
                f'Puesto {idx}': f'{row.Manufacturer} {row.Model}',
                'Precio (USD)': row._9,
                'Combustible': row.Fuel,
                'CO2': row._7,
                'dB': row._8,
                'Millas con un tanque lleno': row._10,
                
            })

        return output_format

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))





# Ruta para obtener estadísticas sobre los viajes en taxi
@app.get("/stats")
def get_taxi_stats(pickup_borough: str):
    """
    Endpoint para obtener estadísticas sobre los viajes en taxi para un distrito específico.

    Parameters:
        - pickup_borough: Nombre del distrito para el cual se desean las estadísticas. Debe ser uno de los siguientes:
            - Manhattan
            - Brooklyn
            - Queens
            - Bronx
            - Staten Island
            - EWR

    Returns:
        - JSON con información estadística.
    """
    try:
        # Cargar el DataFrame desde el archivo parquet
        taxis = pd.read_parquet('Datasets/taxis_2023.parquet')

        # Filtrar datos para el distrito específico
        filtered_taxis = taxis[taxis['pickup_borough'] == pickup_borough]

        # Calcular estadísticas sobre los datos de viajes en taxi en el distrito
        total_trips = len(filtered_taxis)
        
        # Calcular la duración promedio del viaje en horas
        duration_timedelta = (filtered_taxis['dropoff_datetime'] - filtered_taxis['pickup_datetime'])
        average_trip_duration = round(duration_timedelta.mean() / timedelta(hours=1), 2)

        # Calcular la media de viajes por día
        daily_trips = round(filtered_taxis.groupby(filtered_taxis['pickup_datetime'].dt.date).size().mean(), 2)

        # Calcular la media de viajes por mes
        monthly_trips = round(filtered_taxis.groupby(filtered_taxis['pickup_datetime'].dt.to_period("M")).size().mean(), 2)

        average_trip_distance = round(filtered_taxis['trip_distance'].mean(), 2)
        average_total_amount = round(filtered_taxis['total_amount'].mean(), 2)

        # Devolver la información estadística como JSON
        return {
            "Viajes Totales": total_trips,
            "Duración Promedio (Hs)": average_trip_duration,
            "Media de Viajes por Día": daily_trips,
            "Media de Viajes por Mes": monthly_trips,
            "Distancia recorrida promedio (millas)": average_trip_distance,
            "Total ganado en promedio (USD)": average_total_amount,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))