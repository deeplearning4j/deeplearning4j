#export TEST_RUNNER_PREFIX=valgrind
mvn clean package
 /home/agibsonccc/Documents/GitHub/deeplearning4j/platform-tests/bin/java -cp target/benchmarks.jar org.nd4j.SmallNDArraysTroubleshoot