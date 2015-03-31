package org.tribbloid.ispark

import org.tribbloid.ispark.Util.{debug, getpid, log}
import org.tribbloid.ispark.display.{Data, IScala}
import org.tribbloid.ispark.interpreters.{Results, SparkInterpreter}
import org.tribbloid.ispark.json.JsonUtil._
import org.tribbloid.ispark.msg._
import org.zeromq.ZMQ
import sun.misc.{Signal, SignalHandler}

import scala.collection.mutable
import scalax.file.Path

trait Parent {
  val profile: Profile
  val ipy: Communication
  val interpreter: SparkInterpreter //TODO: support multiple interpreter

  var in: mutable.Map[Int, String] = mutable.Map()
  var out: mutable.Map[Int, Any] = mutable.Map()

  val session: Session = new Session
  var n: Int = 0

  /**
   * Public API
   */
  def nextInput(): Int = {
    n += 1
    n
  }

  /**
   * Public API
   */
  def storeInput(input: String) {
    in(n) = input
    session.addHistory(n, input)
  }

  /**
   * Public API
   */
  def storeOutput(result: Results.Value, output: String) {
    out(n) = result.value
    session.addOutputHistory(n, output)
    interpreter.bind("_" + n, result.tpe, result.value)
  }
}

class Main(options: Options) extends Parent {
  val profile = options.profile match {
    case Some(path) => Path(path).string.as[Profile]
    case None =>
      val file = Path(s"profile-${getpid()}.json")
      log(s"connect ipython with --existing ${file.toAbsolute.path}")
      val profile = Profile.default
      file.write(toJSON(profile))
      profile
  }

  override lazy val interpreter =  new SparkInterpreter(options.tail.toArray)

  val zmq = new Sockets(profile)
  val ipy = new Communication(zmq, profile)

  def welcome() {
    import scala.util.Properties._
    log(s"Welcome to Scala $versionNumberString ($javaVmName, Java $javaVersion)")
  }

  Runtime.getRuntime.addShutdownHook(
    new Thread() {
      override def run() {
        debug("Terminating Main")
        interpreter.close()

        session.endSession(n)
      }
    }
  )

  Signal.handle(new Signal("INT"), new SignalHandler {
    private var previously = System.currentTimeMillis

    def handle(signal: Signal) {
      if (!options.parent) {
        val now = System.currentTimeMillis
        if (now - previously < 500) sys.exit() else previously = now
      }

      interpreter.cancel()
    }
  })

  class HeartBeat extends Thread {
    override def run() {
      ZMQ.proxy(zmq.heartbeat, zmq.heartbeat, null)
    }
  }

  (options.profile, options.parent) match {
    case (Some(file), true) =>
      // This setup means that this kernel was started by IPython. Currently
      // IPython is unable to terminate Main without explicitly killing it
      // or sending shutdown_request. To fix that, Main watches the profile
      // file whether it exists or not. When the file is removed, Main is
      // terminated.

      class FileWatcher(file: java.io.File, interval: Int) extends Thread {
        override def run() {
          while (true) {
            if (file.exists) Thread.sleep(interval)
            else sys.exit()
          }
        }
      }

      val fileWatcher = new FileWatcher(file, 1000)
      fileWatcher.setName(s"FileWatcher(${file.getPath})")
      fileWatcher.start()
    case _ =>
  }

  val ExecuteHandler    = new ExecuteHandler(this)
  val CompleteHandler   = new CompleteHandler(this)
  val KernelInfoHandler = new KernelInfoHandler(this)
  val ObjectInfoHandler = new ObjectInfoHandler(this)
  val ConnectHandler    = new ConnectHandler(this)
  val ShutdownHandler   = new ShutdownHandler(this)
  val HistoryHandler    = new HistoryHandler(this)
  val CommOpenHandler   = new CommOpenHandler(this)
  val CommMsgHandler    = new CommMsgHandler(this)
  val CommCloseHandler  = new CommCloseHandler(this)

  class Conn(msg: Msg[_]) extends display.Conn {
    def display_data(data: Data) {
      ipy.send_display_data(msg, data)
    }
  }

  class EventLoop(socket: ZMQ.Socket) extends Thread {
    def dispatch[T <: FromIPython](msg: Msg[T]) {
      IScala.withConn(new Conn(msg)) {
        msg.header.msg_type match {
          case MsgType.execute_request     => ExecuteHandler(socket, msg.asInstanceOf[Msg[execute_request]])
          case MsgType.complete_request    => CompleteHandler(socket, msg.asInstanceOf[Msg[complete_request]])
          case MsgType.kernel_info_request => KernelInfoHandler(socket, msg.asInstanceOf[Msg[kernel_info_request]])
          case MsgType.object_info_request => ObjectInfoHandler(socket, msg.asInstanceOf[Msg[object_info_request]])
          case MsgType.connect_request     => ConnectHandler(socket, msg.asInstanceOf[Msg[connect_request]])
          case MsgType.shutdown_request    => ShutdownHandler(socket, msg.asInstanceOf[Msg[shutdown_request]])
          case MsgType.history_request     => HistoryHandler(socket, msg.asInstanceOf[Msg[history_request]])
          case MsgType.comm_open           => CommOpenHandler(socket, msg.asInstanceOf[Msg[comm_open]])
          case MsgType.comm_msg            => CommMsgHandler(socket, msg.asInstanceOf[Msg[comm_msg]])
          case MsgType.comm_close          => CommCloseHandler(socket, msg.asInstanceOf[Msg[comm_close]])
          case _                           =>
        }
      }
    }

    override def run() {
      try {
        while (true) {
          ipy.recv(socket).foreach(dispatch)
        }
      } catch {
        case exc: Exception =>
          zmq.terminate() // this will gracefully terminate heartbeat
          throw exc
      }
    }
  }

  val heartBeat = new HeartBeat
  heartBeat.setName("HeartBeat")
  heartBeat.start()

  debug("Starting kernel event loop")
  ipy.send_status(ExecutionState.starting)

  val requestsLoop = new EventLoop(zmq.requests)
  requestsLoop.setName("RequestsEventLoop")
  requestsLoop.start()

  welcome()
}


object Main {

  def main (args: Array[String]) {
    Util.options = new Options(args)
    Util.daemon = new Main(Util.options)
    Util.daemon.heartBeat.join()
  }
}