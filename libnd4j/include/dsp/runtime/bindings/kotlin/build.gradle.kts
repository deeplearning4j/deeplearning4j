plugins {
    kotlin("jvm") version "1.9.24"
}

group = "org.nd4j"
version = "0.0.1-SNAPSHOT"

repositories {
    mavenCentral()
    mavenLocal()
}

dependencies {
    implementation("net.java.dev.jna:jna:5.14.0")
}

sourceSets {
    main {
        java {
            // Staged SDK layout: the Java wrapper source is packaged as a
            // sibling wrappers/java/ directory. In the libnd4j source tree the
            // canonical copy lives in the nd4j-sdx Maven module instead —
            // Gradle silently skips whichever srcDir does not exist.
            srcDir("../java/src/main/java")
            srcDir("../../../../../../nd4j/nd4j-backends/nd4j-backend-impls/nd4j-sdx/src/main/java")
        }
    }
}

kotlin {
    jvmToolchain(11)
}
