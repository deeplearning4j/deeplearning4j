---
title: Using snapshots in your app
layout: default
---

# What's "snapshots"?

We provide automated daily builds of our repositories: ND4J/DataVec/DeepLearning4j/RL4J etc. So, all newest functionality & bug fixes are delivered daily.


# Setup instructions

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

And specify snapshot version. We follow simple rule. If latest stable release version is A.B.C, snapshot version will be A.B.(C+1)-SNAPSHOT. I.e. as of writing this, latest stable version is `0.9.1`, and snapshot version is `0.9.2-SNAPSHOT`


# Limitations

Primary limitation when using snapshots, is absence of `-platform` artifacts. So, if you've been using `nd4j-native-platform` as your backend, you should be using `nd4j-native` with snapshots.
 