package org.nd4j.api.scala.repl

import scala.tools.nsc.Settings
import scala.tools.nsc.interpreter.{ILoop, SimpleReader}



/**
 * @author Adam Gibson
 */
object RunRepl {

  def main(args : Array[String]): Unit = {

    val repl = new Nd4jLoop
    repl.settings = new Settings
    repl.settings.usejavacp.value = true
    repl.settings.Yreplsync.value = true
    repl.in = SimpleReader()
    repl.settings.prompt.value = true

    repl.createInterpreter()
    repl.intp.addImports("org.nd4j.linalg.factory.Nd4j", "org.nd4j.api.linalg.DSL._")

    // start the interpreter and then close it after you :quit
    repl.loop()
    try {
      repl.closeInterpreter()

    }
    catch {case e : Exception => }

  }

  class Nd4jLoop extends ILoop {
    override def prompt = "nd4j=> "

    addThunk {
      intp.beQuietDuring {
        intp.addImports("org.nd4j.linalg.factory.Nd4j", "org.nd4j.api.linalg.DSL._")
      }
    }

    override def printWelcome() {
      echo("Welcome to nd4j. Please remember to always specify a backend when invoking this.")
    }
  }

}
