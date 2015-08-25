lazy val root = (project in file(".")).settings(
  scalaVersion := "2.11.7",
  crossScalaVersions := Seq("2.10.5", "2.11.7", "2.12.0-M1"),
  name := "nd4s",
  version := "0.4-rc1-SNAPSHOT",
  organization := "org.nd4j",
  resolvers += "Local Maven Repository" at "file://" + Path.userHome.absolutePath + "/.m2/repository",
  libraryDependencies ++= Seq(
    "org.nd4j" % "nd4j-api" % "0.4-rc1-SNAPSHOT",
    "org.nd4j" % "nd4j-jblas" % "0.4-rc1-SNAPSHOT" % Test,
    "ch.qos.logback" % "logback-classic" %  "1.1.3" % Test,
    "org.scalatest" %% "scalatest" % "2.2.4" % Test cross CrossVersion.binaryMapped {
      case x if x startsWith "2.12" => "2.11"
      case x => x
    }
  ),
  scalacOptions ++= Seq("-unchecked", "-deprecation", "-feature", "-language:implicitConversions", "-language:higherKinds", "-language:postfixOps"),
  initialCommands in console := "import org.nd4j.linalg.factory.Nd4j; import org.nd4j.api.Implicits._"
)
