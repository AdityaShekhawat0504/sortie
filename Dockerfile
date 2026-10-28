FROM python:3.12-slim
WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir ".[api]"
ENV SORTIE_DATA=/data
VOLUME /data
EXPOSE 8000
CMD ["uvicorn", "sortie.api:app", "--host", "0.0.0.0", "--port", "8000"]
