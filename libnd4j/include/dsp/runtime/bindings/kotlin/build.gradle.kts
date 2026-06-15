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
            srcDir("../java/src/main/java")
        }
    }
}

kotlin {
    jvmToolchain(11)
}
