import scala.sys.process._

name := "ScalNet"
version := "1.0.0-SNAPSHOT"
description := "A Scala wrapper for Deeplearning4j, inspired by Keras. Scala + DL + Spark + GPUs"

scalaVersion := "2.11.12"

resolvers in ThisBuild ++= Seq(
  Resolver.sonatypeRepo("snapshots")
)

cleanFiles += baseDirectory.value / "lib"
val mvnInstall = Seq("mvn", "install", "-q", "-f", "sbt-pom.xml")
val operatingSystem = sys.props("os.name").toLowerCase.substring(0, 3)
update := {
  operatingSystem match {
    case "win" => { Seq("cmd", "/C") ++ mvnInstall !; update.value }
    case _     => { mvnInstall !; update.value }
  }
}

libraryDependencies ++= {

  val dl4j = "1.0.0-SNAPSHOT"
  val logback = "1.2.3"
  val scalaCheck = "1.13.5"
  val scalaTest = "3.0.5"

  Seq(
    "org.deeplearning4j" % "deeplearning4j-core" % dl4j,
    "org.slf4j" % "slf4j-api" % "1.7.25",
    "ch.qos.logback" % "logback-classic" % logback,
    "org.nd4j" % "nd4j-native" % dl4j % "test",
    "org.scalacheck" %% "scalacheck" % scalaCheck % "test",
    "org.scalatest" %% "scalatest" % scalaTest % "test"
  )
}

scalacOptions in ThisBuild ++= Seq("-language:postfixOps",
                                   "-language:implicitConversions",
                                   "-language:existentials",
                                   "-feature",
                                   "-deprecation")

lazy val standardSettings = Seq(
  organization := "org.deeplearning4j",
  organizationName := "Skymind",
  startYear := Some(2016),
  licenses += ("Apache-2.0", url("http://www.apache.org/licenses/LICENSE-2.0.html")),
  homepage := Some(url("https://github.com/deeplearning4j/ScalNet")),
  crossScalaVersions := Seq("2.11.12", "2.10.7"),
  scalacOptions ++= Seq(
    "-encoding",
    "UTF-8",
    "-Xlint",
    "-deprecation",
    "-Xfatal-warnings",
    "-feature",
    "-language:postfixOps",
    "-unchecked"
  )
)

parallelExecution in Test := false
scalafmtOnCompile in ThisBuild := true
scalafmtTestOnCompile in ThisBuild := true
test in assembly := {}
assemblyMergeStrategy in assembly := {
  case PathList("META-INF", xs @ _*) => MergeStrategy.discard
  case x                             => MergeStrategy.first
}

lazy val root = (project in file("."))
  .enablePlugins(AutomateHeaderPlugin)
  .settings(standardSettings)
  .settings(
    name := "ScalNet",
    fork := true
  )
