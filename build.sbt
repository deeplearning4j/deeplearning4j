import com.typesafe.sbt.pgp.PgpKeys

lazy val root = (project in file(".")).settings(
  scalaVersion := "2.11.7",
  crossScalaVersions := Seq("2.10.5", "2.11.7", "2.12.0-M1"),
  name := "nd4s",
  version := "0.4-rc1",
  organization := "org.nd4j",
  resolvers += "Local Maven Repository" at "file://" + Path.userHome.absolutePath + "/.m2/repository",
  libraryDependencies ++= Seq(
    "org.nd4j" % "nd4j-api" % "0.4-rc1",
    "org.nd4j" % "nd4j-jblas" % "0.4-rc1" % Test,
    "ch.qos.logback" % "logback-classic" %  "1.1.3" % Test,
    "org.scalatest" %% "scalatest" % "2.2.4" % Test cross CrossVersion.binaryMapped {
      case x if x startsWith "2.12" => "2.11"
      case x => x
    }
  ),
  scalacOptions ++= Seq("-unchecked", "-deprecation", "-feature", "-language:implicitConversions", "-language:higherKinds", "-language:postfixOps"),
  publishMavenStyle := true,
  publishArtifact in Test := false,
  pomIncludeRepository := {_ => false},
  publishTo <<= version {
    v =>
      val nexus = "https://oss.sonatype.org/"
      if (v.trim.endsWith("SNAPSHOT"))
        Some("snapshots" at nexus + "content/repositories/snapshots")
      else
        Some("releases" at nexus + "service/local/staging/deploy/maven2")
  },
 pomExtra := {
   <url>http://nd4s.org/</url>
     <licenses>
       <license>
         <name>Apache License, Version 2.0</name>
         <url>http://www.apache.org/licenses/LICENSE-2.0.html</url>
         <distribution>repo</distribution>
       </license>
     </licenses>
     <scm>
       <connection>scm:git@github.com:SkymindIO/nd4s.git</connection>
       <developerConnection>scm:git:git@github.com:SkymindIO/nd4s.git</developerConnection>
       <url>git@github.com:deeplearning4j/nd4s.git</url>
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
  releasePublishArtifactsAction := PgpKeys.publishSigned.value,
  releaseCrossBuild := true,
  initialCommands in console := "import org.nd4j.linalg.factory.Nd4j; import org.nd4s.Implicits._"
)
