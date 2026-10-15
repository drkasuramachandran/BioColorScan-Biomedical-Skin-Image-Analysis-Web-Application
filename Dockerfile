FROM python:3.11-slim

# Hugging Face Spaces runs containers as user 1000
RUN useradd -m -u 1000 user
USER user
ENV PATH=/home/user/.local/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

WORKDIR /home/user/app

COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY --chown=user . .

EXPOSE 7860

# CORS/XSRF protection is disabled so uploads work when the app is embedded
# in an iframe on skincancerimaging.drkasuramachandran.com
CMD ["streamlit", "run", "app.py", \
     "--server.port=7860", "--server.address=0.0.0.0", \
     "--server.enableCORS=false", "--server.enableXsrfProtection=false", \
     "--server.maxUploadSize=200"]
