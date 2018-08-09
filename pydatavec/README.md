# pydatavec : Python interface for DataVec

## Installation

Clone deeplearning4j mono repo:

```bash
git clone https://www.github.com/deeplearning4j/deeplearning4j.git
```

Switch to datavec directory:

```bash
cd deeplearning4j/datavec
```

Add shade plug-in to datavec's `pom.xml`:

```xml
<plugin>
<groupId>org.apache.maven.plugins</groupId>
<artifactId>maven-shade-plugin</artifactId>
<version>3.1.1</version>
<configuration>
</configuration>
<executions>
<execution>
<phase>package</phase>
<goals>
<goal>shade</goal>
</goals>
</execution>
</executions>
</plugin>
```

Build datavec:

```bash
mvn clean install -D skipTests
```

Find `datavec-api-1.0.0-SNAPSHOT.jar` in `/deeplearning4j/datavec/datavec-api/target`

set `DATAVEC_CLASS_PATH` to the parent directory of this file.

This would look something like
```bash
export DATAVEC_CLASS_PATH=.../deeplearning4j/datavec/datavec-api/target
```

Clone pydatavec:

```bash
git clone https://www.github.com/deeplearning4j/pydatavec.git
```

Install pydatavec:

```bash
cd pydatavec
python setup.py install
```

Run example:

```bash
cd examples
python basic.py
```

