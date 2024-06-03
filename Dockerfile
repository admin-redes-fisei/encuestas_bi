# Usa una imagen base oficial de Python
FROM python:3.9-slim

# Establece el directorio de trabajo en el contenedor
WORKDIR /app

# Copia el archivo de requisitos al directorio de trabajo
COPY requirements.txt .

# Instala las dependencias del sistema
RUN apt-get update && \
    apt-get install -y gcc && \
    apt-get clean

# Instala las dependencias de Python
RUN pip install --no-cache-dir -r requirements.txt

# Copia el código de la aplicación al contenedor
COPY . .

# Expone el puerto en el que correrá la aplicación
#EXPOSE 5000

# Define el comando para correr la aplicación
CMD ["python", "app.py"]
