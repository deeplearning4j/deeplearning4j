package org.tribbloid.ispark.interpreters

import org.tribbloid.ispark.display.Data

/**
 * Created by peng on 12/30/14.
 */
object Results {
  sealed trait Result
  sealed trait Success extends Result
  sealed trait Failure extends Result

  case class Value(value: Any, tpe: String, repr: Data) extends Success
  case object NoValue extends Success

  case class Exception(exception: Throwable) extends Failure {
//    def traceback = exception.getStackTraceString
  }
  case object Error extends Failure
  case object Incomplete extends Failure
  case object Cancelled extends Failure
}
