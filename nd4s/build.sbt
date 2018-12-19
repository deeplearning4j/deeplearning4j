lazy val currentVersion = SettingKey[String]("currentVersion")
lazy val nd4jVersion = SettingKey[String]("nd4jVersion")
lazy val publishSomeThing = sys.props.getOrElse("repoType", default = "local").toLowerCase match {
  case repoType if repoType.contains("nexus") => publishNexus
  case repoType if repoType.contains("bintray") => publishBintray
  case repoType if repoType.contains("sonatype") => publishSonatype
  case _ => publishLocalLocal
}

val nexusStagingRepoId = sys.props.getOrElse("stageRepoId", default = "deploy/maven2")
lazy val releaseRepositoryId = sys.props.getOrElse("stageRepoId", default = "deploy/maven2") match {
  case stageRepoId if stageRepoId.equals("") => "deploy/maven2"
  case stageRepoId if stageRepoId.equals("deploy/maven2") => "deploy/maven2"
  case _ => "deployByRepositoryId/" + nexusStagingRepoId
}

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

lazy val commonSettings = Seq(
  scalaVersion := "2.11.8",
  crossScalaVersions := Seq("2.10.6", "2.11.8"),
  name := "nd4s",
  version := sys.props.getOrElse("currentVersion", default = "1.0.0-SNAPSHOT"),
  organization := "org.nd4j",
  resolvers += Resolver.mavenLocal,
  resolvers in ThisBuild ++= Seq(Opts.resolver.sonatypeSnapshots),
  nd4jVersion := sys.props.getOrElse("nd4jVersion", default = "1.0.0-SNAPSHOT"),
  libraryDependencies ++= Seq(
    "com.nativelibs4java" %% "scalaxy-loops" % "0.3.4",
//    "org.nd4j" % "nd4j-api" % nd4jVersion.value,
//    "org.nd4j" % "nd4j-native-platform" % nd4jVersion.value % Test,
    "org.scalatest" %% "scalatest" % "2.2.6" % Test,
    "ch.qos.logback" % "logback-classic" % "1.2.1" % Test,
    "org.scalacheck" %% "scalacheck" % "1.12.5" % Test,
    "org.scalanlp" %% "breeze" % "0.12" % Test,
    "com.github.julien-truffaut" %% "monocle-core" % "1.2.0" % Test
  ),
  scalacOptions ++= Seq("-unchecked", "-deprecation", "-feature", "-language:implicitConversions", "-language:higherKinds", "-language:postfixOps"),
  publishMavenStyle := true,
  publishArtifact in Test := false,
  pomIncludeRepository := { _ => false },
  useGpg := true,
  pgpPassphrase := Some(Array()),
  credentials += Credentials(Path.userHome / ".ivy2" / ".credentials"),
  releasePublishArtifactsAction := com.typesafe.sbt.pgp.PgpKeys.publishSigned.value,
  releaseCrossBuild := true,
  initialCommands in console := "import org.nd4j.linalg.factory.Nd4j; import org.nd4s.Implicits._"
)

lazy val publishNexus = Seq(
  publishTo := {
    val nexus = "https://nexus.ci.skymind.io/"
    if (isSnapshot.value)
      Some("snapshots" at nexus + "content/repositories/maven-snapshots")
    else
      Some("releases" at nexus + "service/local/staging/" + releaseRepositoryId)
  }
)

lazy val publishBintray = Seq(
  publishTo := {
    val jfrog = "https://oss.jfrog.org/artifactory/"
    if (isSnapshot.value)
      Some("snapshots" at jfrog + "oss-snapshot-local")
    else
      Some("releases" at jfrog + "oss-release-local")
  }
)

lazy val publishSonatype = Seq(
  publishTo := {
    val nexus = "https://oss.sonatype.org/"
    if (isSnapshot.value)
      Some("snapshots" at nexus + "content/repositories/snapshots")
    else
      Some("releases" at nexus + "service/local/staging/" + releaseRepositoryId)
  }
)

lazy val publishLocalLocal = Seq(
  publish := {},
  publishLocal := {}
)

lazy val root = (project in file(".")).settings(
  commonSettings,
  publishSomeThing
)
