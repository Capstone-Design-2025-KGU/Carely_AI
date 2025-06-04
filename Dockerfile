FROM python:3.10-slim
WORKDIR /app

RUN apt-get update && apt-get install -y openjdk-17-jre-zero

RUN ["pip","install","--upgrade","pip"]
COPY requirements.txt requirements.txt
RUN ["pip","install","-r","requirements.txt"]

COPY /app .

EXPOSE 8000
ENTRYPOINT ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]