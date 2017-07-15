# Jumpy
--------------------------

Jumpy is the python interface for [nd4j](https://nd4j.org)

Jumpy allows for python to use nd4j via pointers
(no network communication required like a lot of python libraries)


Jumpy is a thin wrapper around numpy and [jnius](https://github.com/kivy/pyjnius).


To setup jumpy, you need to set a jumpy classpath via the enviroment variable:

JUMPY_CLASSPATH


The JUMPY_CLASSPATH can be set to a list of jar files that contain
the necessary jar files for running an [nd4j backend](http://nd4j.org/backend.html)

Install:
pip install jumpy

Setting up the classpath
--------------------------------------------------

Jumpy currently requires building using snapshots. In order to use snapshots:

Clone the [examples](https://github.com/deeplearning4j/dl4j-examples)
```
git clone https://github.com/deeplearning4j/dl4j-examples
```

Modify the versions to include 0.8.1-SNAPSHOT:
https://github.com/deeplearning4j/dl4j-examples/blob/master/pom.xml#L21

Change the nd4j backend to be either (nd4j-native or nd4j-cuda-8.0) for cpu and gpu respectively:
https://github.com/deeplearning4j/dl4j-examples/blob/master/pom.xml#L14

Paste:
```
            <plugin>
                <groupId>org.apache.maven.plugins</groupId>
                <artifactId>maven-shade-plugin</artifactId>
                <version>${maven-shade-plugin.version}</version>
                <configuration>
                    <shadedArtifactAttached>true</shadedArtifactAttached>
                    <shadedClassifierName>${shadedClassifier}</shadedClassifierName>
                    <createDependencyReducedPom>true</createDependencyReducedPom>
                    <filters>
                        <filter>
                            <artifact>*:*</artifact>
                            <excludes>
                                <exclude>org/datanucleus/**</exclude>
                                <exclude>META-INF/*.SF</exclude>
                                <exclude>META-INF/*.DSA</exclude>
                                <exclude>META-INF/*.RSA</exclude>
                            </excludes>
                        </filter>
                    </filters>
                </configuration>
                <executions>
                    <execution>
                        <phase>package</phase>
                        <goals>
                            <goal>shade</goal>
                        </goals>
                        <configuration>
                            <transformers>
                                <transformer implementation="org.apache.maven.plugins.shade.resource.AppendingTransformer">
                                    <resource>reference.conf</resource>
                                </transformer>
                                <transformer implementation="org.apache.maven.plugins.shade.resource.ServicesResourceTransformer"/>
                                <transformer implementation="org.apache.maven.plugins.shade.resource.ManifestResourceTransformer">
                                </transformer>
                            </transformers>
                        </configuration>
                    </execution>
                </executions>
            </plugin>
```

in tho the pom.xml. This will make maven build 1 jar you can use with jumpy.

Run:
```
cd nd4j-examples && mvn clean install -DskipTests
```

This will give you a jar in the target directory. The target directory should contain a very large jar with -bin in the name.

Finally,  run (not exactly this but close):
```
export JUMPY_CLASS_PATH=/path/to/jar
```

This should allow you to get started with jumpy.

File issues if there are problems.

