import com.typesafe.sbt.SbtGit.{ GitKeys => git }

name := "ScalNet"
version := "0.9.2-SNAPSHOT"
description := "A Scala wrapper for Deeplearning4j, inspired by Keras. Scala + DL + Spark + GPUs"

scalaVersion := "2.11.12"

resolvers ++= Seq(
  Resolver.sonatypeRepo("snapshots")
)

javaCppPresetLibs ++= Seq("openblas" -> "0.2.20")

libraryDependencies ++= {

  val dl4j = "0.9.2-SNAPSHOT"
  val scalaLogging = "3.8.0"
  val logbackClassic = "1.2.3"
  val scalaCheck = "1.13.5"
  val scalaTest = "3.0.5"
  val typesafe = "1.3.2"

  Seq(
    "org.deeplearning4j" % "deeplearning4j-core" % dl4j,
    "com.typesafe.scala-logging" %% "scala-logging" % scalaLogging,
    "ch.qos.logback" % "logback-classic" % logbackClassic,
    "com.typesafe" % "config" % typesafe,
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
  licenses += ("Apache-2.0", url("http://www.apache.org/licenses/LICENSE-2.0.html")),
  homepage := Some(url("https://github.com/deeplearning4j/ScalNet")),
  scmInfo := Some(
    ScmInfo(url("https://github.com/deeplearning4j/ScalNet"), "scm:git:git@github.com:deeplearning4j/ScalNet.git")
  ),
  crossScalaVersions := Seq("2.11.12", "2.10.6"),
  pomExtra := (
    <developers>
    <developer>
      <id>turambar</id>
      <name>Dave Kale</name>
    </developer>
  </developers>
  ),
  publishMavenStyle := true,
  publishTo := {
    if (version.value.trim.endsWith("SNAPSHOT")) Some(Opts.resolver.sonatypeSnapshots)
    else Some(Opts.resolver.sonatypeStaging)
  },
  git.gitRemoteRepo := "git@github.com:deeplearning4j/ScalNet.git",
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

scalafmtOnCompile in ThisBuild := true
scalafmtTestOnCompile in ThisBuild := false

lazy val root = (project in file("."))
  .settings(standardSettings)
  .settings(
    name := "ScalNet"
  )
