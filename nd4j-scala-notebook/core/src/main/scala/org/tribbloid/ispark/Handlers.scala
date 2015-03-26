package org.tribbloid.ispark

import org.tribbloid.ispark.db.DB
import org.tribbloid.ispark.interpreters.Results
import org.tribbloid.ispark.msg._
import org.tribbloid.ispark.msg.formats._
import org.zeromq.ZMQ

abstract class Handler[T <: FromIPython](parent: Parent) extends ((ZMQ.Socket, Msg[T]) => Unit)

class ExecuteHandler(parent: Parent) extends Handler[execute_request](parent) {
  import parent.{interpreter, ipy}

  private def capture[T](msg: Msg[_])(block: => T): T = {
    val size = 10240

    class WatchStream(input: java.io.InputStream, name: String) extends Thread {
      override def run() {
        val buffer = new Array[Byte](size)

        try {
          while (true) {
            val n = input.read(buffer)
            ipy.send_stream(msg, name, new String(buffer.take(n)))

            if (n < size) {
              Thread.sleep(50) // a little delay to accumulate output
            }
          }
        } catch {
          case _: java.io.IOException => // stream was closed so job is done
        }
      }
    }

    val stdoutIn = new java.io.PipedInputStream(size)
    val stdoutOut = new java.io.PipedOutputStream(stdoutIn)
    val stdout = new java.io.PrintStream(stdoutOut)

    val stderrIn = new java.io.PipedInputStream(size)
    val stderrOut = new java.io.PipedOutputStream(stderrIn)
    val stderr = new java.io.PrintStream(stderrOut)

    // This is a heavyweight solution to start stream watch threads per
    // input, but currently it's the cheapest approach that works well in
    // multiple thread setup. Note that piped streams work only in thread
    // pairs (producer -> consumer) and we start one thread per execution,
    // so technically speaking we have multiple producers, which completely
    // breaks the earlier intuitive approach.

    new WatchStream(stdoutIn, "stdout").start()
    new WatchStream(stderrIn, "stderr").start()

    try {
      val result =
        Console.withOut(stdout) {
          Console.withErr(stderr) {
            block
          }
        }

      stdoutOut.flush()
      stderrOut.flush()

      // Wait until both streams get dry because we have to
      // send messages with streams' data before execute_reply
      // is send. Otherwise there will be no output in clients
      // or it will be incomplete.
      while (stdoutIn.available > 0 || stderrIn.available > 0)
        Thread.sleep(10)

      result
    } finally {
      // This will effectively terminate threads.
      stdoutOut.close()
      stderrOut.close()
      stdoutIn.close()
      stderrIn.close()
    }
  }

  def apply(socket: ZMQ.Socket, msg: Msg[execute_request]) {
    import parent.n

    val content = msg.content
    val code = content.code.replaceAll("\\s+$", "")
    val silent = content.silent || code.endsWith(";")
    val store_history = content.store_history getOrElse !silent

    if (code.trim.isEmpty) {
      ipy.send_ok(msg, n)
      return
    }

    parent.nextInput()
    parent.storeInput(code)

    ipy.publish(msg.pub(MsgType.pyin,
      pyin(
        execution_count=n,
        code=code)))

    ipy.busy {
      interpreter.resetOutput()

      code match {
        case Magic(name, input, Some(magic)) =>
          val ir = capture(msg) {
            magic(interpreter, input)
          }

          ir match {
            case Some(error) =>
              ipy.send_error(msg, n, error)
            case None =>
              val output = interpreter.output.toString

              if (!output.trim.isEmpty)
                ipy.send_error(msg, n, output)
              else
                ipy.send_ok(msg, n)
          }
        case Magic(name, _, None) =>
          ipy.send_error(msg, n, s"ERROR: Line magic function `%$name` not found.")
        case _ =>
          capture(msg) { interpreter.interpret(code) } match {
            case result @ Results.Value(value, tpe, repr) if !silent =>
              if (store_history) {
                repr.default foreach { output =>
                  parent.storeOutput(result, output)
                }
              }

              val resultMsg =  msg.pub(
                MsgType.pyout,
                pyout(
                  execution_count=n,
                  data=repr
                )
              )

              ipy.publish(resultMsg)

              ipy.send_ok(msg, n)
            case Results.NoValue =>
              ipy.send_ok(msg, n)
            case exc @ Results.Exception(exception) =>
              ipy.send_error(msg, pyerr.fromThrowable(n, exception))
            case Results.Error =>
              ipy.send_error(msg, n, interpreter.output.toString)
            case Results.Incomplete =>
              ipy.send_error(msg, n, "incomplete")
            case Results.Cancelled =>
              ipy.send_abort(msg, n)
          }
      }
    }
  }
}

class CompleteHandler(parent: Parent) extends Handler[complete_request](parent) {
  import parent.{interpreter, ipy}

  def apply(socket: ZMQ.Socket, msg: Msg[complete_request]) {
    val text = if (msg.content.text.isEmpty) {
      // Notebook only gives us line and cursor_pos
      val pos = msg.content.cursor_pos
      val upToCursor = msg.content.line.splitAt(pos)._1
      upToCursor.split("""[^\w.%]""").last // FIXME java.util.NoSuchElementException
    } else {
      msg.content.text
    }

    val matches = if (msg.content.line.startsWith("%")) {
      val prefix = text.stripPrefix("%")
      Magic.magics.map(_.name.name).filter(_.startsWith(prefix)).map("%" + _)
    } else {
      val completions = interpreter.completions(text)
      val common = Util.commonPrefix(completions)
      var prefix = Util.suffixPrefix(text, common)
      completions.map(_.stripPrefix(prefix)).map(text + _)
    }

    ipy.send(
      socket, msg.reply(
        MsgType.complete_reply,
        complete_reply(
          status=ExecutionStatus.ok,
          matches=matches,
          matched_text=text
        )
      )
    )
  }
}

class KernelInfoHandler(parent: Parent) extends Handler[kernel_info_request](parent) {
  import parent.ipy

  def apply(socket: ZMQ.Socket, msg: Msg[kernel_info_request]) {
    val scalaVersion = Util.scalaVersion
      .split(Array('.', '-'))
      .take(3)
      .map(_.toInt)
      .toList

    ipy.send(socket, msg.reply(MsgType.kernel_info_reply,
      kernel_info_reply(
        protocol_version=(4, 0),
        language_version=scalaVersion,
        language="scala")))
  }
}

class ConnectHandler(parent: Parent) extends Handler[connect_request](parent) {
  import parent.ipy

  def apply(socket: ZMQ.Socket, msg: Msg[connect_request]) {
    ipy.send(socket, msg.reply(MsgType.connect_reply,
      connect_reply(
        shell_port=parent.profile.shell_port,
        iopub_port=parent.profile.iopub_port,
        stdin_port=parent.profile.stdin_port,
        hb_port=parent.profile.hb_port)))
  }
}

class ShutdownHandler(parent: Parent) extends Handler[shutdown_request](parent) {
  import parent.ipy

  def apply(socket: ZMQ.Socket, msg: Msg[shutdown_request]) {
    ipy.send(socket, msg.reply(MsgType.shutdown_reply,
      shutdown_reply(
        restart=msg.content.restart)))
    sys.exit()
  }
}

class ObjectInfoHandler(parent: Parent) extends Handler[object_info_request](parent) {
  import parent.ipy

  def apply(socket: ZMQ.Socket, msg: Msg[object_info_request]) {
    ipy.send(socket, msg.reply(MsgType.object_info_reply,
      object_info_notfound_reply(
        name=msg.content.oname)))
  }
}

class HistoryHandler(parent: Parent) extends Handler[history_request](parent) {
  import parent.ipy

  def apply(socket: ZMQ.Socket, msg: Msg[history_request]) {
    import scala.slick.driver.SQLiteDriver.simple._
    import Database.dynamicSession

    val raw = msg.content.raw

    var query = for {
      (input, output) <- DB.History leftJoin DB.OutputHistory on ((in, out) => in.session === out.session && in.line === out.line)
    } yield (input.session, input.line, if (raw) input.source_raw else input.source, output.output.?)

    msg.content.hist_access_type match {
      case HistAccessType.range =>
        val session = msg.content.session getOrElse 0

        val actualSession =
          if (session == 0) parent.session.id
          else if (session > 0) session
          else parent.session.id - session

        query = query.filter(_._1 === actualSession)

        for (start <- msg.content.start)
          query = query.filter(_._2 >= start)

        for (stop <- msg.content.stop)
          query = query.filter(_._2 < stop)
      case HistAccessType.tail | HistAccessType.search =>
        // TODO: add support for `pattern` and `unique`
        query = query.sortBy(r => (r._1.desc, r._2.desc))

        for (n <- msg.content.n)
          query = query.take(n)
    }

    val rawHistory = DB.db.withDynSession { query.list }
    val history =
      if (msg.content.output)
        rawHistory.map { case (session, line, input, output) => (session, line, Right((input, output))) }
      else
        rawHistory.map { case (session, line, input, output) => (session, line, Left(input)) }

    ipy.send(socket, msg.reply(MsgType.history_reply,
      history_reply(
        history=history)))
  }
}

class CommOpenHandler(parent: Parent) extends Handler[comm_open](parent) {
  def apply(socket: ZMQ.Socket, msg: Msg[comm_open]) { println(msg) }
}

class CommMsgHandler(parent: Parent) extends Handler[comm_msg](parent) {
  def apply(socket: ZMQ.Socket, msg: Msg[comm_msg]) { println(msg) }
}

class CommCloseHandler(parent: Parent) extends Handler[comm_close](parent) {
  def apply(socket: ZMQ.Socket, msg: Msg[comm_close]) { println(msg) }
}
