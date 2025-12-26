@echo off
cd /d "%~dp0"
call mvn -q clean compile exec:java -Dexec.mainClass="org.deeplearning4j.tools.CppDependencyAnalyzer" -Dexec.args="%*"
