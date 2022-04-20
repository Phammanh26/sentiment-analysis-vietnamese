FROM python:3.6
WORKDIR /code

RUN pip install --upgrade pip
RUN pip install tensorflow==2.6.0 -f https://tf.kmtea.eu/whl/stable.html
COPY requirements.txt requirements.txt
RUN pip3 install -r requirements.txt
COPY . . 
ENV FLASK_APP=app.py
ENV FLASK_RUN_HOST=0.0.0.0
EXPOSE 5000
ENTRYPOINT [ "python" ]
CMD [ "app.py" ]