@echo off
if not exist %cd%\tests mkdir tests
set currTime=%DATE:/=-%_%TIME::=-%
set outputFile=tests\nd4j-native_%currTime%.log
ECHO ***** Starting tests using nd4j-native backend, piping output to %outputFile% *****
call mvn clean site -Ddependency.locations.enabled=false -P test-nd4j-native -fn -pl !deeplearning4j-cuda > %cd%\%outputFile% 2>&1
ECHO ***** Constructing nd4j-native site at projectRootDirectory\tests\nd4j-native_DATE-TIME *****
call mvn site:stage -DstagingDirectory=%cd%\tests\nd4j-native_%currTime% -pl !deeplearning4j-cuda
ECHO ***** Starting tests using nd4j-cuda-8 backend *****
set outputFile=tests\nd4j-cuda-8_%currTime%.log
call mvn clean site -Ddependency.locations.enabled=false -P test-nd4j-cuda-8.0 -fn > %cd%\%outputFile% 2>&1
ECHO ***** Constructing nd4j-native site at projectRootDirectory\tests\nd4j-cuda-8_DATE-TIME *****
call mvn site:stage -DstagingDirectory=%cd%\tests\nd4j-cuda-8_%currTime%