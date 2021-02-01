#  /* ******************************************************************************
#   *
#   *
#   * This program and the accompanying materials are made available under the
#   * terms of the Apache License, Version 2.0 which is available at
#   * https://www.apache.org/licenses/LICENSE-2.0.
#   *
#   * Unless required by applicable law or agreed to in writing, software
#   * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
#   * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
#   * License for the specific language governing permissions and limitations
#   * under the License.
#   *
#   * SPDX-License-Identifier: Apache-2.0
#   ******************************************************************************/

################################################################################
#
# This program and the accompanying materials are made available under the
# terms of the Apache License, Version 2.0 which is available at
# https://www.apache.org/licenses/LICENSE-2.0.
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations
# under the License.
#
# SPDX-License-Identifier: Apache-2.0
################################################################################


def dl4j_core_dependencies():
    return """<dependency>
            <groupId>org.deeplearning4j</groupId>
            <artifactId>deeplearning4j-core</artifactId>
            <version>${project.version}</version>
        </dependency>"""


def spark_dependencies():
    return """<dependency>
            <groupId>org.deeplearning4j</groupId>
            <artifactId>dl4j-spark-parameterserver_{scala.binary.version}</artifactId>
            <version>{dl4j.spark.version}</version>
        </dependency>"""


def datavec_dependencies():
    return """<dependency>
            <groupId>org.datavec</groupId>
            <artifactId>datavec-api</artifactId>
            <version>${project.version}</version>
        </dependency>
        <dependency>
            <groupId>org.datavec</groupId>
            <artifactId>datavec-local</artifactId>
            <version>${project.version}</version>
        </dependency>
        <dependency>
            <groupId>org.datavec</groupId>
            <artifactId>datavec-arrow</artifactId>
            <version>${project.version}</version>
        </dependency>
        <dependency>
            <groupId>org.datavec</groupId>
            <artifactId>datavec-camel</artifactId>
            <version>${project.version}</version>
        </dependency>
        <dependency>
            <groupId>org.datavec</groupId>
            <artifactId>datavec-excel</artifactId>
            <version>${project.version}</version>
        </dependency>
        <dependency>
            <groupId>org.datavec</groupId>
            <artifactId>datavec-geo</artifactId>
            <version>${project.version}</version>
        </dependency>
        <dependency>
            <groupId>org.datavec</groupId>
            <artifactId>datavec-hadoop</artifactId>
            <version>${project.version}</version>
        </dependency>
        <dependency>
            <groupId>org.datavec</groupId>
            <artifactId>datavec-jdbc</artifactId>
            <version>${project.version}</version>
        </dependency>
        <dependency>
            <groupId>org.datavec</groupId>
            <artifactId>datavec-perf</artifactId>
            <version>${project.version}</version>
        </dependency>
        <dependency>
            <groupId>org.datavec</groupId>
            <artifactId>datavec-spark_{scala.binary.version}</artifactId>
            <version>{dl4j.spark.version}</version>
        </dependency>
"""


def pom_template():
    return """<?xml version="1.0" encoding="UTF-8"?>
<!--~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  ~ Copyright (c) 2015-2018 Skymind, Inc.
  ~
  ~ This program and the accompanying materials are made available under the
  ~ terms of the Apache License, Version 2.0 which is available at
  ~ https://www.apache.org/licenses/LICENSE-2.0.
  ~
  ~ Unless required by applicable law or agreed to in writing, software
  ~ distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
  ~ WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
  ~ License for the specific language governing permissions and limitations
  ~ under the License.
  ~
  ~ SPDX-License-Identifier: Apache-2.0
  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~-->

<project xmlns="http://maven.apache.org/POM/4.0.0"
    xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
    xsi:schemaLocation="http://maven.apache.org/POM/4.0.0 http://maven.apache.org/maven-v4_0_0.xsd">


    <modelVersion>4.0.0</modelVersion>
    <groupId>org.deeplearning4j</groupId>
    <artifactId>pydl4j</artifactId>
    <version>{dl4j.version}</version>
    <packaging>jar</packaging>

    <name>pydl4j</name>

    <properties>
        <maven-shade-plugin.version>3.0.0</maven-shade-plugin.version>
        <javacpp.version>1.5.4</javacpp.version>
        <javacpp-presets.version>1.5.4</javacpp-presets.version>
        <openblas.version>0.3.10</openblas.version>
    </properties>

    <licenses>
        <license>
            <name>Apache License, Version 2.0</name>
            <url>http://www.apache.org/licenses/LICENSE-2.0.txt</url>
            <distribution>repo</distribution>
        </license>
    </licenses>

    <dependencies>
        <dependency>
            <groupId>org.bytedeco</groupId>
            <artifactId>openblas-platform</artifactId>
            <version>${openblas.version}-${javacpp-presets.version}</version>
        </dependency>
        <dependency>
            <groupId>org.nd4j</groupId>
            <artifactId>{nd4j.platform.backend}</artifactId>
            <version>${project.version}</version>
        </dependency>
        <dependency>
            <groupId>org.nd4j</groupId>
            <artifactId>{nd4j.backend}</artifactId>
            <version>${project.version}</version>
            <classifier>linux-x86_64</classifier>
        </dependency>
        <dependency>
            <groupId>org.nd4j</groupId>
            <artifactId>{nd4j.backend}</artifactId>
            <version>${project.version}</version>
            <classifier>windows-x86_64</classifier>
        </dependency>
        {dl4j.core.dependencies}
        {spark.dependencies}
        {datavec.dependencies}
        <dependency>
            <groupId>ch.qos.logback</groupId>
            <artifactId>logback-classic</artifactId>
            <version>1.2.3</version>
        </dependency>
    </dependencies>

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

    <build>
        <plugins>
            <plugin>
                <groupId>org.apache.maven.plugins</groupId>
                <artifactId>maven-shade-plugin</artifactId>
                <version>${maven-shade-plugin.version}</version>
                <configuration>
                    <shadedArtifactAttached>true</shadedArtifactAttached>
                    <shadedClassifierName>bin</shadedClassifierName>
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
            <plugin>
                <groupId>org.apache.maven.plugins</groupId>
                <artifactId>maven-compiler-plugin</artifactId>
                <version>3.1</version>
                <configuration>
                    <source>1.8</source>
                    <target>1.8</target>
                </configuration>
            </plugin>
            <plugin>
                <groupId>org.apache.maven.plugins</groupId>
                <artifactId>maven-jar-plugin</artifactId>
                <configuration>
                    <forceCreation>true</forceCreation>
                </configuration>
                <executions>
                    <execution>
                        <id>empty-javadoc-jar</id>
                        <phase>package</phase>
                        <goals>
                            <goal>jar</goal>
                        </goals>
                        <configuration>
                            <skip>false</skip>
                            <classifier>javadoc</classifier>
                            <classesDirectory>${basedir}/javadoc</classesDirectory>
                        </configuration>
                    </execution>
                    <execution>
                        <id>empty-sources-jar</id>
                        <phase>package</phase>
                        <goals>
                            <goal>jar</goal>
                        </goals>
                        <configuration>
                            <classifier>sources</classifier>
                            <classesDirectory>${basedir}/src</classesDirectory>
                        </configuration>
                    </execution>
                </executions>
            </plugin>
        </plugins>
    </build>
</project>


    """
