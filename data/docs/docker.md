# Docker: основы

## Образы и контейнеры
Образ — это шаблон. Контейнер — запущенный экземпляр образа.
Скачать образ: docker pull <image>. Запустить: docker run <image>.

## Dockerfile
Dockerfile описывает, как собрать образ. Пример:
FROM python:3.11
WORKDIR /app
COPY . .
RUN pip install -r requirements.txt
CMD ["python", "app.py"]

## Сборка и запуск
Собрать образ: docker build -t myapp . Запустить: docker run -p 8000:8000 myapp.

## Docker Compose
Compose позволяет запускать несколько контейнеров.
