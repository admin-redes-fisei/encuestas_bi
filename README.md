## README para la implementación
### Requisitos previos

* Servidor web Apache
* Docker
* GitHub
* Base de datos MySQL (específicamente `encuestasdb` en el servidor)
* Certificado de seguridad SSL
* Servicios PHP en funcionamiento

### Pasos para la implementación

**1. Acceder al directorio del proyecto:**

```bash
cd /var/www/html/encuestas/fisei-encuestas-docker
```

**2. Verificar archivo docker-compose.yml:**

Asegúrese de que el archivo `docker-compose.yml` exista y contenga el siguiente contenido:

```yaml
version: "3.9"

services:
  backend:
    build:
      context: ./encuestas_bi
      dockerfile: Dockerfile
    ports:
      - "5000:5000"

  frontend:
    build:
      context: ./encuestas-app
      dockerfile: Dockerfile
    ports:
      - "4000:4000"
    depends_on:
      - backend
```

**3. Clonar el repositorio del backend:**

Clone el repositorio del backend de encuestas en la ruta especificada:

```bash
git clone <URL_REPOSITORIO> encuestas_bi
```

**4. Actualizar cambios en el código:**

1. Realice los cambios necesarios en el código del backend y envíelos a la rama `main` del repositorio.
2. En el servidor, navegue a la ruta del backend:

```bash
cd /var/www/html/encuestas/fisei-encuestas-docker/encuestas_bi
```

3. Actualice el repositorio local con los últimos cambios:

```bash
git pull
```

4. Regrese al directorio principal del proyecto:

```bash
cd /var/www/html/encuestas/fisei-encuestas-docker
```

5. Reconstruya y ejecute los contenedores Docker Compose:

```bash
docker compose up -d --build
```

**Nota:** Cualquier cambio en la configuración del contenedor backend debe realizarse en el archivo `Dockerfile`.
