FROM python:3.6.9
LABEL maintainer="Yraimtj, yraimtj@gmail.com"

RUN apt-get update -y
RUN apt-get install -y build-essential
RUN pip install --upgrade pip


WORKDIR /app

ADD ./requirements.txt /app/requirements.txt

RUN pip install -r requirements.txt

ADD . /app

ENTRYPOINT ["python"]
CMD ["api_meero.py"]