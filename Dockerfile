# Use official Python 3.12 slim image as specified in Section 7 of project guidelines
FROM python:3.12-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    DEBIAN_FRONTEND=noninteractive

# Set working directory
WORKDIR /app

# Install system dependencies required for SQLite and network downloads
RUN apt-get update && apt-get install -y --no-install-recommends \
    sqlite3 \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency files first to leverage Docker layer caching
COPY requirements.txt pyproject.toml* ./

# Upgrade pip and install dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application source code
COPY . .

# Create data directory and build/cache the Chinook SQLite database
RUN mkdir -p /app/data && \
    if [ ! -f /app/data/chinook.db ]; then \
        echo "Downloading Chinook SQL script..." && \
        curl -sSL "https://raw.githubusercontent.com/lerocha/chinook-database/master/ChinookDatabase/DataSources/Chinook_Sqlite.sql" -o /app/data/Chinook_Sqlite.sql && \
        echo "Building Chinook SQLite database..." && \
        sqlite3 /app/data/chinook.db < /app/data/Chinook_Sqlite.sql && \
        rm /app/data/Chinook_Sqlite.sql && \
        echo "Chinook SQLite DB successfully created at /app/data/chinook.db"; \
    fi

# Gradio default port is usually 7860
EXPOSE 7860

# Command to launch your Gradio app via standard Python
CMD ["python", "app.py"]

# To build the Docker image, run the following command in the terminal:
#docker build -t chinook-customer-support-agent .

# To run the Docker container, use the following command:
#docker run -d -p 7860:7860 --name chinook-customer-support-agent --env-file .env chinook-customer-support-agent

#Launch the application from browser by navigating to http://localhost:7860
