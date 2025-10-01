# Use an official Python runtime as a parent image
FROM jangkuz/hcm202-deps AS runtime

# Set the working directory in the container
WORKDIR /app

# Copy the application code
COPY app app/
COPY rag rag/
COPY .env .

# Expose the port the application will run on
EXPOSE 8000

# Run the command to start the development server
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]