
# Stage 1: Base Image
# We start with the official slim Python image.
FROM python:3.9-slim

# Set working directory inside container
WORKDIR /app

# Copy requirements.txt and install dependencies
COPY requirements.txt requirements.txt

# Install dependencies
# --no-cache-dir to keep image size small
RUN pip install --no-cache-dir -r requirements.txt

# Copy all our application code to working directory
COPY ./app /app/app
COPY ./main.py /app/main.py

# Copy model artifacts
COPY ./best_model.onnx /app/best_model.onnx
COPY ./kolom_model.txt /app/kolom_model.txt

# Tell Docker that container will run on port 8000
EXPOSE 8000

# Command to run application when container starts
# uvicorn.run(app, host="0.0.0.0", port=8000)
# "0.0.0.0" means API will be accessible from outside container
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
