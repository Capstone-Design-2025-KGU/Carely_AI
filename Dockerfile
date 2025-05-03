FROM python:3.12-alpine
WORKDIR /app
COPY /app .
RUN ["pip3","install","-r","-user","requirements.txt"]
EXPOSE 8000
ENTRYPOINT ["uvicorn","main:app","--reload"]