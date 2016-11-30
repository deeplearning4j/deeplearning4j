@echo off
ECHO ***** Starting tests using nd4j-native backend *****
call mvn clean site -Ddependency.locations.enabled=false -P test-nd4j-native -fn
ECHO ***** Constructing nd4j-native site at projectRootDirectory\site\nd4j-native_DATE-TIME *****
call mvn site:stage -DstagingDirectory=%cd%\site\nd4j-native_%DATE:/=-%_%TIME::=-%
ECHO ***** Starting tests using nd4j-cuda-8 backend *****
call mvn clean site -Ddependency.locations.enabled=false -P test-nd4j-native -fn
ECHO ***** Constructing nd4j-native site at projectRootDirectory\site\nd4j-cuda-8_DATE-TIME *****
call mvn site:stage -DstagingDirectory=%cd%\site\nd4j-cuda-8_%DATE:/=-%_%TIME::=-%