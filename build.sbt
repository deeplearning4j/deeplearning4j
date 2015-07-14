lazy val root = (project in file(".")).settings(
       scalaVersion := "2.10.5",
       name := "nd4s",
       version := "0.0.3.5.5.6-SNAPSHOT",
       organization := "org.nd4j",
       resolvers += "Local Maven Repository" at "file://"+Path.userHome.absolutePath+"/.m2/repository", 
       libraryDependencies ++= Seq( 
              "org.scala-lang" % "scala-compiler" % "2.10.5",
              "org.nd4j" % "nd4j-api" % "0.0.3.5.5.6-SNAPSHOT",  
              "org.nd4j" % "nd4j-jblas" % "0.0.3.5.5.6-SNAPSHOT" % Test,
              "org.scalatest" %% "scalatest" % "2.2.4" % Test
       ),
       scalacOptions ++= Seq("-unchecked", "-deprecation", "-feature", "-language:implicitConversions", "-language:higherKinds")
)
