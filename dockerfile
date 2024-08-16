FROM pytorch/pytorch

WORKDIR /app
COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

RUN chmod +x run.sh
CMD ["./run.sh"]

