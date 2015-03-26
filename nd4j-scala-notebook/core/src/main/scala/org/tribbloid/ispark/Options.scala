package org.tribbloid.ispark

import java.io.File
import joptsimple.{OptionParser,OptionSpec}

class Options(args: Seq[String]) {
  private val parser = new OptionParser()
  private val _help = parser.accepts("help", "show help").forHelp()
  private val _verbose = parser.accepts("verbose", "print debugging information")
  private val _parent = parser.accepts("parent", "indicate that IPython started this engine")
  private val _profile = parser.accepts("profile", "path to connection file").withRequiredArg().ofType(classOf[File])
  private val options = parser.parse(args: _*)

  def tail = args.dropWhile(_ != "--").drop(1).toList

  private def has[T](spec: OptionSpec[T]): Boolean =
    options.has(spec)

  private def get[T](spec: OptionSpec[T]): Option[T] =
    Some(options.valueOf(spec)).filter(_ != null)

  val help: Boolean = has(_help)
  val verbose: Boolean = has(_verbose)
  val parent: Boolean = has(_parent)
  val profile: Option[File] = get(_profile)

  if (help) {
    parser.printHelpOn(System.out)
    sys.exit()
  }
}
