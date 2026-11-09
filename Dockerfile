FROM python:3.11-slim

# System dependencies for geopandas + shapefile support
RUN apt-get update && apt-get install -y \
    gdal-bin \
    libgdal-dev \
    libspatialindex-dev \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY src/ ./src/
COPY config.py .

# GEE credentials will be mounted at runtime (see docker-compose.yml)
# Never bake credentials into the image

CMD ["python", "src/01_gee_export.py", "--help"]