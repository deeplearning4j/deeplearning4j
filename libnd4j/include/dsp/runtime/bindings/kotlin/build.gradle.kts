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
    implementation("org.nd4j:sdx-runtime-java-bindings:0.0.1-SNAPSHOT")
}

kotlin {
    jvmToolchain(11)
}
