---
title: Using snapshots for daily updated builds of DL4J in your app
layout: default
---

# DL4J and ND4J Snapshots

Contents

* [Introduction to Snapshots](#Introduction)
* [Setup Instructions](#Setup_Instructions)
* [Limitations](#Limitations)
* [Configuration of ND4J Backend](#ND4J_Backend)
* [Note to Gradle Users](#Note_to_gradle_users)

## <a name="Introduction">Overview/Introduction</a>

We provide automated daily builds of our repositories: ND4J/DataVec/DeepLearning4j/RL4J etc. So, all newest functionality & bug fixes are delivered daily.

Snapshots work like any other maven dependencies, just served from custom repository, instead of Maven Central.

**Note that due to ongoing development, snapshots should be considered less stable than releases: breaking changes or bugs can in principle be introduced at any point during the course of normal development. Typically, releases (not snapshots) should be used when possible, unless a bug fix or new feature is required.**

## <a name="Setup_Instructions">Setup Instructions</a>

Basically to use snapshots in your project, you should just add snapshot repository information to your `pom.xml`, like shown below:

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

And specify snapshot version. We follow simple rule. If latest stable release version is `A.B.C`, snapshot version will be `A.B.(C+1)-SNAPSHOT`. i.e. as of writing this, latest stable version is `0.9.1`, and snapshot version is `0.9.2-SNAPSHOT`

## <a name="Limitations">Limitations</a>

Primary limitation when using snapshots, is absence of `-platform` artifacts. So, if you've been using `nd4j-native-platform` as your backend, you should be using `nd4j-native` with snapshots.

## <a name="ND4J_Backend">ND4J_Backend</a>

If your pom.xml has a dependency for `nd4j-native-platform` and you switch to using snapshots to get access to a recent feature you will have to switch your `nd4j-backend` to `nd4j-native`

## <a name="Note_to_gradle_users">Note to gradle users</a>

Snapshots will not work with gradle. You will have to use maven to download the files. After that you may try using your local maven repository with mavenLocal().

A bare minimum file like

```Gradle
version '1.0-SNAPSHOT'
 
apply plugin: 'java'
 
sourceCompatibility = 1.8
 
repositories {
    maven { url "https://oss.sonatype.org/content/repositories/snapshots" }
    mavenCentral()
}
 
dependencies {
    compile group: 'org.deeplearning4j', name: 'deeplearning4j-core', version: '0.9.2-SNAPSHOT'
    compile group: 'org.deeplearning4j', name: 'deeplearning4j-modelimport', version: '0.9.2-SNAPSHOT'
//  also tried:
//  compile group: 'org.nd4j', name: 'nd4j-native', version: '0.9.2-SNAPSHOT'
    compile "org.nd4j:nd4j-native:0.9.2-SNAPSHOT"
    compile "org.nd4j:nd4j-native:0.9.2-SNAPSHOT:macosx-x86_64"
    testCompile group: 'junit', name: 'junit', version: '4.12'
 
}
```

should work but does not. This is due to [a bug in gradle](https://github.com/gradle/gradle/issues/2882)


Gradle with snapshots *and* maven classifiers appears to be a problem. 

 
