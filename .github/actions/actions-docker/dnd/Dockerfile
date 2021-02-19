FROM docker:18.06.1-ce-dind

RUN apk add --no-cache \
        py-pip \
        openssl \
        bash \
        shadow

RUN pip install --upgrade pip && pip install docker-compose
