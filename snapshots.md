---
title: Using snapshots for daily updated builds of DL4J in your app
layout: default
---

# Snapshots Aren't Working

Contents

* [Introduction to Snapshots](#Introduction)
* [Setup Instructions](#Setup_Instructions)
* [Limitations](#Limitations)
* [Confiiguration of ND4J Backend](#ND4J_Backend)

## <a name="Introduction">Overview/Introduction</a>

**Please use Maven to build 0.9.1, or compile from source.**

We provide automated daily builds of our repositories: ND4J/DataVec/DeepLearning4j/RL4J etc. So, all newest functionality & bug fixes are delivered daily.

Snapshots work like any other maven depenedencies, just served from custom repository, instead of Maven Central. 

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

And specify snapshot version. We follow simple rule. If latest stable release version is `A.B.C`, snapshot version will be `A.B.(C+1)-SNAPSHOT`. I.e. as of writing this, latest stable version is `0.9.1`, and snapshot version is `0.9.2-SNAPSHOT`

## <a name="Limitations">Limitations</a>

Primary limitation when using snapshots, is absence of `-platform` artifacts. So, if you've been using `nd4j-native-platform` as your backend, you should be using `nd4j-native` with snapshots.

## <a name="ND4J_Backend">ND4J_Backend</a>

If your pom.xml has a dependency for `nd4j-native-platform` and you switch to using snapshots to get access to a recent feature you will have to switch your `nd4j-backend` to `nd4j-native`
 
