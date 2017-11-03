FROM gettyimages/spark:2.2.0-hadoop-2.7

# SciPy
RUN set -ex \
 && buildDeps=' \
    libpython3-dev \
    build-essential \
    pkg-config \
    gfortran \
 ' \
 && apt-get update && apt-get install -y --no-install-recommends \
    $buildDeps \
    ca-certificates \
    wget \
    liblapack-dev \
    libopenblas-dev \
 && packages=' \
    numpy \
    pandasql \
    scipy \
    simplejson \
 ' \
 && pip3 install $packages \
 && apt-get purge -y --auto-remove $buildDeps \
 && apt-get clean \
 && rm -rf /var/lib/apt/lists/*

# Zeppelin
ENV ZEPPELIN_PORT 8080
ENV ZEPPELIN_HOME /usr/zeppelin
ENV ZEPPELIN_CONF_DIR $ZEPPELIN_HOME/conf
ENV ZEPPELIN_NOTEBOOK_DIR $ZEPPELIN_HOME/notebook
RUN echo '{ "allow_root": true }' > /root/.bowerrc
RUN set -ex \
 && buildDeps=' \
    git \
    bzip2 \
    npm \
 ' \
 && apt-get update && apt-get install -y --no-install-recommends $buildDeps \
 && curl -sL http://archive.apache.org/dist/maven/maven-3/3.5.0/binaries/apache-maven-3.5.0-bin.tar.gz \
   | gunzip \
   | tar x -C /tmp/

RUN git clone https://github.com/ShamsUlAzeem/zeppelin.git /usr/src/zeppelin

RUN cd /usr/src/zeppelin && git init && git fetch && git checkout 'ipynb-export/import'

RUN cd /usr/src/zeppelin \
 && MAVEN_OPTS="-Xmx2g -XX:MaxPermSize=1024m" /tmp/apache-maven-3.5.0/bin/mvn package -Pbuild-distr -DskipTests \
 && tar xvf /usr/src/zeppelin/zeppelin-distribution/target/zeppelin*.tar.gz -C /usr/ \
 && mv /usr/zeppelin* $ZEPPELIN_HOME \
 && mkdir -p $ZEPPELIN_HOME/logs \
 && mkdir -p $ZEPPELIN_HOME/run \
 && apt-get purge -y --auto-remove $buildDeps \
 && rm -rf /var/lib/apt/lists/* \
 && rm -rf /usr/src/zeppelin \
 && rm -rf /root/.m2 \
 && rm -rf /root/.npm \
 && rm -rf /tmp/*

RUN ln -s -f /usr/bin/pip3 /usr/bin/pip \
 && ln -s -f /usr/bin/python3 /usr/bin/python

RUN curl -sL http://archive.apache.org/dist/maven/maven-3/3.5.0/binaries/apache-maven-3.5.0-bin.tar.gz \
   | gunzip \
   | tar x -C /tmp/

RUN apt-get update && apt-get install dos2unix
COPY notebook_json $ZEPPELIN_HOME/notebook_json
ADD json-folder-ids.py $ZEPPELIN_HOME
RUN cd $ZEPPELIN_HOME && python json-folder-ids.py && mkdir $ZEPPELIN_HOME/otherpoms && cd $ZEPPELIN_HOME/otherpoms
ADD pom.xml $ZEPPELIN_HOME/otherpoms
RUN cd $ZEPPELIN_HOME/otherpoms && /tmp/apache-maven-3.5.0/bin/mvn package
ADD zeppelin-env.sh $ZEPPELIN_HOME/conf
ADD log4j.properties $ZEPPELIN_HOME/conf
RUN dos2unix $ZEPPELIN_HOME/conf/zeppelin-env.sh

EXPOSE 4040
EXPOSE 8080

WORKDIR $ZEPPELIN_HOME
CMD ["bin/zeppelin.sh"]