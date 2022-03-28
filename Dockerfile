FROM python:3.6
WORKDIR /code
COPY . .
VOLUME /Users/phamvanmanh/Desktop/sentiment-analyst
RUN pip install --upgrade pip
RUN pip install tensorflow==2.6.0 -f https://tf.kmtea.eu/whl/stable.html
RUN pip3 install -r requirements.txt
ENV FLASK_APP=app.py
ENV FLASK_RUN_HOST=0.0.0.0
EXPOSE 5000
ENTRYPOINT [ "python" ]
CMD [ "app.py" ]