FROM python:3.13

LABEL authors="bogdanresetko"

WORKDIR /kursova
COPY . .


CMD ["python3", "main.py"]
