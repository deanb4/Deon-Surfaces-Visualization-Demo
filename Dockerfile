# Base image with optional GPU (switchable)
FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

# set working directory
WORKDIR /app

# Insatll system dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# copy requirements and install python packages
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    && pip install whitenoise 


# Copy Django folder
COPY . .

# Make sure static directory exists
RUN mkdir -p staticfiles

# Collect static files (optional)
RUN python manage.py collectstatic --noinput 

# set environment variable (update as neededO)
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Expose port
EXPOSE 8000

# start Gunicorn server with Djano app
CMD ["gunicorn", "deonSurfacesDemo.wsgi:application", "--bind", "0.0.0.0:8000", "--timeout", "120"]