# Jumpy
--------------------------

Jumpy is the python interface for [nd4j](https://nd4j.org).

Jumpy allows for python to use nd4j via pointers (no network communication required like a lot of python libraries).

Jumpy is a thin wrapper around numpy and [jnius](https://github.com/kivy/pyjnius).

To setup jumpy, you need to set a jumpy classpath via the environment variable:

```
JUMPY_CLASS_PATH
```

The JUMPY_CLASSPATH can be set to a list of jar files that contain
the necessary jar files for running an [nd4j backend](http://nd4j.org/backend.html)


Install:
```{python}
pip install jumpy
```

Setting up the classpath
--------------------------------------------------

Build an uberjar from the following pom.xml using `maven package`.

```xml
<project xmlns="http://maven.apache.org/POM/4.0.0" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
         xsi:schemaLocation="http://maven.apache.org/POM/4.0.0 http://maven.apache.org/maven-v4_0_0.xsd">
    <modelVersion>4.0.0</modelVersion>
    <groupId>org.deeplearning4j</groupId>
    <artifactId>dl4j</artifactId>
    <packaging>jar</packaging>
    <version>1.0-SNAPSHOT</version>
    <name>dl4j</name>
    <url>http://maven.apache.org</url>

    <properties>
        <dl4j.version>0.9.2-SNAPSHOT</dl4j.version>
    </properties>

    <dependencies>
        <dependency>
            <groupId>org.deeplearning4j</groupId>
            <artifactId>deeplearning4j-core</artifactId>
            <version>${dl4j.version}</version>
        </dependency>
        <dependency>
            <groupId>org.datavec</groupId>
            <artifactId>datavec-api</artifactId>
            <version>${dl4j.version}</version>
        </dependency>
        <dependency>
            <groupId>org.nd4j</groupId>
            <artifactId>nd4j-native</artifactId>
            <version>${dl4j.version}</version>
        </dependency>
    </dependencies>

    <build>
        <plugins>
            <plugin>
            <groupId>org.apache.maven.plugins</groupId>
            <artifactId>maven-shade-plugin</artifactId>
            <version>3.1.0</version>
            <executions>
                <execution>
                <phase>package</phase>
                <goals>
                    <goal>shade</goal>
                </goals>
                <configuration>
                    <transformers>
                    <transformer implementation="org.apache.maven.plugins.shade.resource.ManifestResourceTransformer">
                        <mainClass>org.deeplearning4j.example.App</mainClass>
                    </transformer>
                    </transformers>
                </configuration>
                </execution>
            </executions>
            </plugin>
            <plugin>
                <groupId>org.apache.maven.plugins</groupId>
                <artifactId>maven-compiler-plugin</artifactId>
                <version>3.1</version>
                <configuration>
                    <source>1.8</source>
                    <target>1.8</target>
                </configuration>
            </plugin>
        </plugins>
    </build>

<repositories>
    <repository>
        <id>snapshots-repo</id>
        <url>https://oss.sonatype.org/content/repositories/snapshots</url>
        <releases>
            <enabled>false</enabled>
        </releases>
        <snapshots>
            <enabled>true</enabled>
            <updatePolicy>daily</updatePolicy>  <!-- Optional, update daily -->
        </snapshots>
    </repository>
</repositories>

</project>
```

This will give you a JAR in the target directory. The target directory should contain a very large JAR with -bin in the name.

Finally,  run (not exactly this but close):
```
export JUMPY_CLASS_PATH=/path/to/jar
```

This should allow you to get started with Jumpy.

Please file [issues](https://github.com/deeplearning4j/jumpy/issues) to this repo if there are problems.
