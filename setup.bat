cd ..
git clone https://github.com/deeplearning4j/nd4j.git
git clone https://github.com/deeplearning4j/Canova.git
cd nd4j
mvn clean install -DskipTests -Dmaven.javadoc.skip=true
cd ..
cd Canova
mvn clean install -DskipTests -Dmaven.javadoc.skip=true
cd ..
cd deeplearning4j
mvn clean install -DskipTests -Dmaven.javadoc.skip=true
