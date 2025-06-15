FROM continuumio/miniconda3:latest

# Set environment variable for real-time logging
ENV PYTHONUNBUFFERED=1

# 1. Create the botenv environment with ta-lib and pinned numpy
RUN conda create -y -n botenv python=3.10 \
    ta-lib numpy=1.23.5 -c conda-forge && \
    conda clean -afy

# 2. Set working directory
WORKDIR /app

# 3. Copy requirements
COPY requirements.txt .

# 4. Install pip packages without overriding numpy from conda
RUN conda run -n botenv pip install --no-cache-dir --no-deps -r requirements.txt && \
    conda run -n botenv conda install -y numpy=1.23.5

# 5. Copy the rest of the application
COPY . .

# 6. Default command
CMD ["conda", "run", "-n", "botenv", "python", "main.py"]
