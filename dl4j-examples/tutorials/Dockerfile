FROM frolvlad/alpine-oraclejdk8:slim

# adapted from https://github.com/P7h/docker-spark/blob/master/Dockerfile

# Scala related variables.
ARG SCALA_VERSION=2.11.8
ARG SCALA_BINARY_ARCHIVE_NAME=scala-${SCALA_VERSION}
ARG SCALA_BINARY_DOWNLOAD_URL=https://downloads.lightbend.com/scala/${SCALA_VERSION}/${SCALA_BINARY_ARCHIVE_NAME}.tgz

# SBT related variables.
ARG SBT_VERSION=0.13.15
ARG SBT_BINARY_ARCHIVE_NAME=sbt-$SBT_VERSION
ARG SBT_BINARY_DOWNLOAD_URL=https://dl.bintray.com/sbt/native-packages/sbt/${SBT_VERSION}/${SBT_BINARY_ARCHIVE_NAME}.tgz

# Configure env variables for Scala, SBT and Spark.
# Also configure PATH env variable to include binary folders of Java, Scala, SBT and Spark.
ENV SCALA_HOME  /usr/local/scala
ENV SBT_HOME    /usr/local/sbt
ENV SPARK_HOME  /usr/local/spark
ENV PATH        $JAVA_HOME/bin:$SCALA_HOME/bin:$SBT_HOME/bin:$SPARK_HOME/bin:$SPARK_HOME/sbin:$PATH

# Download, uncompress and move all the required packages and libraries to their corresponding directories in /usr/local/ folder.
RUN apk update && \
    apk add build-base nodejs nodejs-npm lapack-dev python3-dev git bzip2 openssl gfortran curl bash && \
    rm -rf /var/lib/apt/lists/* && \
    rm -rf /tmp/* && \
    wget -qO - ${SCALA_BINARY_DOWNLOAD_URL} | tar -xz -C /usr/local/ && \
    wget -qO - ${SBT_BINARY_DOWNLOAD_URL} | tar -xz -C /usr/local/  && \
    cd /usr/local/ && \
    ln -s ${SCALA_BINARY_ARCHIVE_NAME} scala \
    && packages=' \
        numpy \
        scipy \
        simplejson \
     ' \
    && pip3 install $packages
#    cp spark/conf/log4j.properties.template spark/conf/log4j.properties && \
#    sed -i -e s/WARN/ERROR/g spark/conf/log4j.properties && \
#    sed -i -e s/INFO/ERROR/g spark/conf/log4j.properties \

# HADOOP
ENV HADOOP_VERSION 2.7.3
ENV HADOOP_HOME /usr/hadoop-$HADOOP_VERSION
ENV HADOOP_CONF_DIR=$HADOOP_HOME/etc/hadoop
ENV PATH $PATH:$HADOOP_HOME/bin
RUN curl -sL --retry 3 \
  "http://archive.apache.org/dist/hadoop/common/hadoop-$HADOOP_VERSION/hadoop-$HADOOP_VERSION.tar.gz" \
  | gunzip \
  | tar -x -C /usr/ \
 && rm -rf $HADOOP_HOME/share/doc \
 && chown -R root:root $HADOOP_HOME

# SPARK
ENV SPARK_VERSION 2.1.0
ENV SPARK_PACKAGE spark-${SPARK_VERSION}-bin-without-hadoop
ENV SPARK_HOME /usr/spark-${SPARK_VERSION}
ENV SPARK_DIST_CLASSPATH="$HADOOP_HOME/etc/hadoop/*:$HADOOP_HOME/share/hadoop/common/lib/*:$HADOOP_HOME/share/hadoop/common/*:$HADOOP_HOME/share/hadoop/hdfs/*:$HADOOP_HOME/share/hadoop/hdfs/lib/*:$HADOOP_HOME/share/hadoop/hdfs/*:$HADOOP_HOME/share/hadoop/yarn/lib/*:$HADOOP_HOME/share/hadoop/yarn/*:$HADOOP_HOME/share/hadoop/mapreduce/lib/*:$HADOOP_HOME/share/hadoop/mapreduce/*:$HADOOP_HOME/share/hadoop/tools/lib/*"
ENV PATH $PATH:${SPARK_HOME}/bin
RUN curl -sL --retry 3 \
  "http://d3kbcqa49mib13.cloudfront.net/${SPARK_PACKAGE}.tgz" \
  | gunzip \
  | tar x -C /usr/ \
 && mv /usr/$SPARK_PACKAGE $SPARK_HOME \
 && chown -R root:root $SPARK_HOME

# Zeppelin
ENV ZEPPELIN_PORT 8080
ENV ZEPPELIN_HOME /usr/zeppelin
ENV ZEPPELIN_CONF_DIR $ZEPPELIN_HOME/conf
ENV ZEPPELIN_NOTEBOOK_DIR $ZEPPELIN_HOME/notebook
RUN echo '{ "allow_root": true }' > /root/.bowerrc
RUN set -ex \
 && curl -sL http://archive.apache.org/dist/maven/maven-3/3.5.0/binaries/apache-maven-3.5.0-bin.tar.gz \
   | gunzip \
   | tar x -C /tmp/

RUN git clone https://github.com/ShamsUlAzeem/zeppelin.git /usr/src/zeppelin

RUN cd /usr/src/zeppelin && git init && git fetch && git checkout 'ipynb-export/import'

RUN cd /usr/src/zeppelin \
 && MAVEN_OPTS="-Xmx2g -XX:MaxPermSize=1024m" /tmp/apache-maven-3.5.0/bin/mvn package -Pbuild-distr -DskipTests \
 -Pspark-1.6 \
# -Pspark-2.1 \
 && tar xvf /usr/src/zeppelin/zeppelin-distribution/target/zeppelin*.tar.gz -C /usr/ \
 && mv /usr/zeppelin* $ZEPPELIN_HOME \
 && mkdir -p $ZEPPELIN_HOME/logs \
 && mkdir -p $ZEPPELIN_HOME/run \
 && rm -rf /var/lib/apt/lists/* \
 && rm -rf /usr/src/zeppelin \
 && rm -rf /root/.m2 \
 && rm -rf /root/.npm \
 && find /tmp -maxdepth 1 -not -name 'apache-maven-3.5.0' -not -name "." -exec rm -rf {} \; \
 # Removing extra interpreter folders
 && find $ZEPPELIN_HOME/interpreter -maxdepth 1 -type d -not -name 'spark' -not -name 'md' -not -name "." -exec rm -rf {} \;

RUN ln -s -f /usr/bin/pip3 /usr/bin/pip \
 && ln -s -f /usr/bin/python3 /usr/bin/python

# moves all additional zeppelin files to their respective locations
# remaining files are notebooks, we move those once everything else is gone
RUN mkdir -p $ZEPPELIN_HOME/otherpoms \
    && mkdir -p $ZEPPELIN_HOME/conf \
    && mkdir -p $ZEPPELIN_HOME/dockerfiles
COPY . $ZEPPELIN_HOME/dockerfiles/.

# switch lines if building for cuda or cpu
RUN mv $ZEPPELIN_HOME/dockerfiles/docker/pom-native_spark_2.xml $ZEPPELIN_HOME/otherpoms/pom.xml
#RUN mv $ZEPPELIN_HOME/dockerfiles/docker/pom-native_spark_1.xml $ZEPPELIN_HOME/otherpoms/pom.xml
#RUN mv $ZEPPELIN_HOME/dockerfiles/docker/pom-cuda-8.0_spark_2.xml $ZEPPELIN_HOME/otherpoms/pom.xml
#RUN mv $ZEPPELIN_HOME/dockerfiles/docker/pom-cuda-8.0_spark_1.xml $ZEPPELIN_HOME/otherpoms/pom.xml

RUN ls $ZEPPELIN_HOME/bin/ \
    && cp -a $ZEPPELIN_HOME/dockerfiles/docker/conf/. $ZEPPELIN_HOME/conf/ \
    && cp -a $ZEPPELIN_HOME/dockerfiles/. $ZEPPELIN_HOME/notebook_json \
    && rm -rf $ZEPPELIN_HOME/notebook_json/docker/ \
    && find $ZEPPELIN_HOME/notebook_json/ -type f -not -name '*.json' -delete \
    && dos2unix $ZEPPELIN_HOME/conf/zeppelin-env.sh \
    && dos2unix $ZEPPELIN_HOME/bin/zeppelin.sh \
    && cd $ZEPPELIN_HOME \
    && mv $ZEPPELIN_HOME/dockerfiles/docker/json-folder-ids.py $ZEPPELIN_HOME \
    && python json-folder-ids.py \
    && cd $ZEPPELIN_HOME/otherpoms \
    && /tmp/apache-maven-3.5.0/bin/mvn package \
    && rm -rf $ZEPPELIN_HOME/dockerfiles/

EXPOSE 8080 4040

WORKDIR $ZEPPELIN_HOME
CMD ["/bin/bash","bin/zeppelin.sh"]