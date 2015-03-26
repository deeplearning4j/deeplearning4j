package org.tribbloid.ispark

import java.lang.management.ManagementFactory

trait ScalaUtil {
  def scalaVersion = scala.util.Properties.versionNumberString
}

trait ByteUtil {
  def hex(bytes: Seq[Byte]): String = bytes.map("%02x" format _).mkString
}

trait OSUtil {
  def getpid(): Int = {
    val name = ManagementFactory.getRuntimeMXBean.getName
    name.takeWhile(_ != '@').toInt
  }
}

trait ConsoleUtil {
  val origOut = System.out
  val origErr = System.err

  def log[T](message: => T) {
    origOut.println(message)
  }

  def debug[T](message: => T) {
    if (Util.options.verbose) {
      origOut.println(message)
    }
  }

  def warn[T](message: => T) {
    origErr.println(message)
  }
}

trait StringUtil {
  /** Find longest common prefix of a list of strings.
    */
  def commonPrefix(xs: List[String]): String = {
    if (xs.isEmpty || xs.contains("")) ""
    else xs.head.head match {
      case ch =>
        if (xs.tail forall (_.head == ch)) "" + ch + commonPrefix(xs map (_.tail))
        else ""
    }
  }

  /** Find longest string that is a suffix of `head` and prefix of `tail`.
    *
    *  Example:
    *
    *    isInstance
    *  x.is
    *    ^^
    *
    *  >>> Util.suffixPrefix("x.is", "isInstance")
    *  "is"
    */
  def suffixPrefix(head: String, tail: String): String = {
    var prefix = head
    while (!tail.startsWith(prefix)) {
      prefix = prefix.drop(1)
    }
    prefix
  }
}

object Util extends ScalaUtil with ByteUtil with OSUtil with ConsoleUtil with StringUtil {

  var options: Options = _
  var daemon: Main = _
}
