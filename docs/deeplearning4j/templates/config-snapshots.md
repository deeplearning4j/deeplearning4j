---
title: Snapshots and daily builds
short_title: Snapshots
description: Using daily builds for access to latest Eclipse Deeplearning4j features.
category: Configuration
weight: 10
---

## Contents

* [Introduction to Snapshots](#Introduction)
* [Setup Instructions](#Setup_Instructions)
* [Limitations](#Limitations)
* [Configuration of ND4J Backend](#ND4J_Backend)
* [Note to Gradle Users](#Note_to_gradle_users)

## <a name="Introduction">Overview/Introduction</a>

We provide automated daily builds of repositories such as ND4J, DataVec, DeepLearning4j, RL4J etc. So all the newest functionality and most recent bug fixes are released daily.

Snapshots work like any other Maven dependency. The only difference is that they are served from a custom repository rather than from Maven Central.

**Due to ongoing development, snapshots should be considered less stable than releases: breaking changes or bugs can in principle be introduced at any point during the course of normal development. Typically, releases (not snapshots) should be used when possible, unless a bug fix or new feature is required.**

## <a name="Setup_Instructions">Setup Instructions</a>

**Step 1:**
To use snapshots in your project, you should add snapshot repository information like this to your `pom.xml` file:

```
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
```

**Step 2:**
Make sure to specify the snapshot version. We follow a simple rule: If the latest stable release version is `A.B.C`, the snapshot version will be `A.B.(C+1)-SNAPSHOT`. The current snapshot version is `1.0.0-SNAPSHOT`.
For more details on the repositories section of the pom.xml file, see [Maven documentation](https://maven.apache.org/settings.html#Repositories)

If using properties like the DL4J examples, change:
From version:
```
<dl4j.version>1.0.0-beta2</dl4j.version>
<nd4j.version>1.0.0-beta2</nd4j.version>
```
To version:
```
<dl4j.version>1.0.0-SNAPSHOT</dl4j.version>
<nd4j.version>1.0.0-SNAPSHOT</nd4j.version>
```

For Spark dependencies, change as follows:
```
<dl4j.spark.version>1.0.0-beta2_spark_2</dl4j.spark.version>
```
to
```
<dl4j.spark.version>1.0.0_spark_2-SNAPSHOT</dl4j.spark.version>
```

**Sample pom.xml using Snapshots**

A sample pom.xml is provided here: [sample pom.xml using snapshots](https://gist.github.com/AlexDBlack/28b0c9a72bce562c8782be326a6e2aaa)
This has been taken from the DL4J standalone sample project and modified using step 1 and 2 above. The original (using the last release) can be found [here](https://github.com/eclipse/deeplearning4j-examples/blob/master/standalone-sample-project/pom.xml)


## <a name="Limitations">Limitations</a>

Both `-platform` (all operating systems) and single OS (non-platform) snapshot dependencies are released.
Due to the multi-platform build nature of snapshots, it is possible (though rare) for the `-platform` artifacts to temporarily get out of sync, which can cause build issues.

If you are building and deploying on just one platform, it is safter use the non-platform artifacts, such as:
```
        <dependency>
            <groupId>org.nd4j</groupId>
            <artifactId>nd4j-native</artifactId>
            <version>${nd4j.version}</version>
        </dependency>
```

    
## <a name="mavencommands">Useful Maven Commands for Snapshots</a>

Two commands that might be useful when using snapshot dependencies in Maven is as follows:
1. ```-U``` - for example, in ```mvn package -U```. This ```-U``` option forces Maven to check (and if necessary, download) of new snapshot releases. This can be useful if you need the be sure you have the absolute latest snapshot release.
2. ```-nsu``` - for example, in ```mvn package -nsu```. This ```-nsu``` option stops Maven from checking for snapshot releases. Note however your build will only succeed with this option if you have some snapshot dependencies already downloaded into your local Maven cache (.m2 directory) 

An alternative approach to (1) is to set ```<updatePolicy>always</updatePolicy>``` in the ```<repositories>``` section found earlier in this page.
An alternative approach to (2) is to set ```<updatePolicy>never</updatePolicy>``` in the ```<repositories>``` section found earlier in this page.

## <a name="Note_to_gradle_users">Note to Gradle users</a>

Snapshots will not work with Gradle. You must use Maven to download the files. After that, you may try using your local Maven repository with `mavenLocal()`.

A bare minimum file like this:

```Gradle
version '1.0-SNAPSHOT'
 
apply plugin: 'java'
 
sourceCompatibility = 1.8
 
repositories {
    maven { url "https://oss.sonatype.org/content/repositories/snapshots" }
    mavenCentral()
}
 
dependencies {
    compile group: 'org.deeplearning4j', name: 'deeplearning4j-core', version: '1.0.0-SNAPSHOT'
    compile group: 'org.deeplearning4j', name: 'deeplearning4j-modelimport', version: '1.0.0-SNAPSHOT'
    compile "org.nd4j:nd4j-native:1.0.0-SNAPSHOT"
    // Use windows-x86_64 or linux-x86_64 if you are not on macos
    compile "org.nd4j:nd4j-native:1.0.0-SNAPSHOT:macosx-x86_64"
    testCompile group: 'junit', name: 'junit', version: '4.12'
 
}
```

should work in theory, but it does not. This is due to [a bug in Gradle](https://github.com/gradle/gradle/issues/2882). Gradle with snapshots *and* Maven classifiers appears to be a problem.

 Of note when using the nd4j-native backend on Gradle (and SBT - but not Maven), you need to add openblas as a dependency. We do this for you in the -platform pom. Reference the -platform pom [here](https://github.com/eclipse/deeplearning4j/blob/master/nd4j/nd4j-backends/nd4j-backend-impls/nd4j-native-platform/pom.xml#L19) to double check your dependencies. Note that these are version properties. See the ```<properties>``` section of the pom for current versions of the openblas and javacpp presets required to run nd4j-native.
