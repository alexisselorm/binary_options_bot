FROM continuumio/miniconda3:latest

# 1. Create environment with pinned numpy version
RUN conda create -y -n botenv python=3.10 \
    ta-lib numpy=1.23.5 -c conda-forge && \
    conda clean -afy

# 2. Install requirements with numpy protection
WORKDIR /app
COPY requirements.txt .
RUN conda run -n botenv pip install --no-cache-dir \
    --no-deps -r requirements.txt && \
    conda run -n botenv conda install -y numpy=1.23.5

# 3. Copy application code
COPY . .

# 4. Run with activated environment
CMD ["conda", "run", "-n", "botenv", "python", "main.py"]