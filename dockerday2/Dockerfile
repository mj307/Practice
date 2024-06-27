FROM python:3.11.4

WORKDIR "/tmp/practicedocker"

COPY . .


RUN pip3 install --no-cache-dir -r requirements.txt

EXPOSE 10017

CMD ["python", "main.py"]
