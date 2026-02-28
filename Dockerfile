FROM python:3.11-slim

# Install system dependencies for pyodbc + MS SQL ODBC Driver
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl ca-certificates gnupg2 apt-transport-https \
    unixodbc unixodbc-dev gcc g++ \
 && rm -rf /var/lib/apt/lists/*

# Add Microsoft package repo (for msodbcsql)
RUN curl https://packages.microsoft.com/keys/microsoft.asc | gpg --dearmor > /usr/share/keyrings/microsoft.gpg \
 && echo "deb [arch=amd64 signed-by=/usr/share/keyrings/microsoft.gpg] https://packages.microsoft.com/debian/12/prod bookworm main" \
    > /etc/apt/sources.list.d/mssql-release.list

RUN apt-get update && ACCEPT_EULA=Y apt-get install -y --no-install-recommends msodbcsql18 \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Render sets PORT automatically
CMD gunicorn -b 0.0.0.0:$PORT app:apps