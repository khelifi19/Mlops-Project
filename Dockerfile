# Use Python 3.12 slim as the base image
FROM python:3.12-slim

# Set the working directory inside the container
WORKDIR /app

# Copy only the requirements file first (better caching)
COPY requirements.txt /app/

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire project into the container (excluding files in .dockerignore)
COPY . /app/

# Copy model and preprocessing files explicitly
COPY model.pkl scaler.pkl pca.pkl /app/

# Expose FastAPI port
EXPOSE 8000

# Set environment variables for MLflow tracking
ENV MLFLOW_TRACKING_URI=sqlite:///mlflow.db


# Start FastAPI using Uvicorn
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]

