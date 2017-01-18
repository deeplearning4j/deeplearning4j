---
title: Deeplearning4j Version Check
layout: default
---

# Deeplearning4j's Version Check

Deeplearning4j includes a version check, which may produce HTTP client debug messages in your logs. The version check helps us know whether an up-to-date versions of the library is being used. The version check can be turned off by adding the following line of code to the Java file where you configure your model:

    Heartbeat.getInstance().disableHeartbeat();

While we realize that some users may prefer to turn off the version check, please keep in mind that Deeplearning4j is free, and allowing us to check the version is one way that members of the community can help us. 
