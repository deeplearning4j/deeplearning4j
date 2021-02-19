FROM continuumio/miniconda3

ENV HOME /home/jenkins

RUN groupadd jenkins -g 1000 && useradd -d ${HOME} -u 1000 -g 1000 -m jenkins

RUN export DEBIAN_FRONTEND=noninteractive DEBCONF_NONINTERACTIVE_SEEN=true && \
    apt-get update && \
    apt-get -y --no-install-recommends install \
        dirmngr \
        gnupg \
        build-essential \
        ca-certificates \
        software-properties-common \
        openjdk-8-jdk-headless && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

ENV M2_HOME /opt/maven
RUN mkdir -p ${M2_HOME} && \
    curl -fsSL http://apache.osuosl.org/maven/maven-3/3.6.0/binaries/apache-maven-3.6.0-bin.tar.gz | \
        tar -xzC ${M2_HOME} --strip-components=1 && \
    # Workaround for concurrent safe maven local repository
    curl -O http://repo1.maven.org/maven2/io/takari/aether/takari-local-repository/0.11.2/takari-local-repository-0.11.2.jar && \
    mv takari-local-repository-0.11.2.jar ${M2_HOME}/lib/ext && \
    curl -O http://repo1.maven.org/maven2/io/takari/takari-filemanager/0.8.3/takari-filemanager-0.8.3.jar && \
    mv takari-filemanager-0.8.3.jar ${M2_HOME}/lib/ext && \
    update-alternatives --install "/usr/bin/mvn" "mvn" "/opt/maven/bin/mvn" 0 && \
    update-alternatives --set mvn /opt/maven/bin/mvn

RUN conda config --set always_yes yes --set changeps1 no && \
    conda update -q conda && \
    conda info -a && \
    conda create -q -n test-environment python=3.6 nose

USER jenkins

ENV PATH=/opt/conda/envs/test-environment/bin:/home/jenkins/.local/bin:$PATH \
    JAVA_HOME=/usr/lib/jvm/java-8-openjdk-amd64

RUN echo "source activate test-environment" >> ~/.bashrc

WORKDIR ${HOME}

CMD ["cat"]
