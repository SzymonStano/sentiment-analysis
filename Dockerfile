FROM python:3.10

ARG UID
ARG GID

WORKDIR /app

RUN groupadd --gid $GID myuser && useradd -u $UID --gid $GID myuser && chown -R myuser /app

ENV GENSIM_DATA_DIR=/app/.gensim-data

ENV PYTHONPATH=/app

COPY requirements.txt .

RUN pip --no-cache-dir install -r requirements.txt

RUN python -m gensim.downloader --download glove-wiki-gigaword-50
RUN python -m spacy download en_core_web_sm

USER myuser
