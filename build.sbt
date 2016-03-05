lazy val nd4jVersion = SettingKey[String]("nd4jVersion")

def scalaTestDependency(v: String): ModuleID = "org.scalatest" %% "scalatest" % v

lazy val root = (project in file(".")).settings(
  scalaVersion := "2.11.7",
  crossScalaVersions := Seq("2.10.6", "2.11.7", "2.12.0-M3"),
  name := "nd4s",
  version := "0.4-rc3.9-SNAPSHOT",
  organization := "org.nd4j",
  resolvers += "Local Maven Repository" at "file://" + Path.userHome.absolutePath + "/.m2/repository",
  nd4jVersion := "0.4-rc3.8",
  libraryDependencies ++= Seq(
    "org.nd4j" % "nd4j-api" % nd4jVersion.value,
    "org.nd4j" % "nd4j-x86" % nd4jVersion.value % Test,
    "ch.qos.logback" % "logback-classic" % "1.1.3" % Test,
    "org.scalacheck" %% "scalacheck" % "1.12.5" % Test,
    "org.scalanlp" %% "breeze" % "0.12" % Test,
    "com.github.julien-truffaut" %% "monocle-core" % "1.2.0" % Test
  ),
  libraryDependencies <+= scalaVersion {
    case x if x startsWith "2.12" => scalaTestDependency("2.2.5-M3")
    case x => scalaTestDependency("2.2.4")
  },
  scalacOptions ++= Seq("-unchecked", "-deprecation", "-feature", "-language:implicitConversions", "-language:higherKinds", "-language:postfixOps"),
  publishMavenStyle := true,
  publishArtifact in Test := false,
  pomIncludeRepository := { _ => false },
  publishTo <<= version {
    v =>
      val nexus = "https://oss.sonatype.org/"
      if (v.trim.endsWith("SNAPSHOT"))
        Some("snapshots" at nexus + "content/repositories/snapshots")
      else
        Some("releases" at nexus + "service/local/staging/deploy/maven2")
  },
  pomExtra := {
    <url>http://nd4j.org/</url>
      <licenses>
        <license>
          <name>Apache License, Version 2.0</name>
          <url>http://www.apache.org/licenses/LICENSE-2.0.html</url>
          <distribution>repo</distribution>
        </license>
      </licenses>
      <scm>
        <connection>scm:git@github.com:SkymindIO/deeplearning4j.git</connection>
        <developerConnection>scm:git:git@github.com:SkymindIO/deeplearning4j.git</developerConnection>
        <url>git@github.com:deeplearning4j/deeplearning4j.git</url>
        <tag>HEAD</tag>
      </scm>
      <developers>
        <developer>
          <id>agibsonccc</id>
          <name>Adam Gibson</name>
          <email>adam@skymind.io</email>
        </developer>
        <developer>
          <id>taisukeoe</id>
          <name>Taisuke Oe</name>
          <email>oeuia.t@gmail.com</email>
        </developer>
      </developers>
  },
  credentials += Credentials(Path.userHome / ".ivy2" / ".credentials"),
  releasePublishArtifactsAction := com.typesafe.sbt.pgp.PgpKeys.publishSigned.value,
  releaseCrossBuild := true,
  initialCommands in console := "import org.nd4j.linalg.factory.Nd4j; import org.nd4s.Implicits._")

