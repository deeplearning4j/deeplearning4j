package org.tribbloid.ispark

import org.tribbloid.ispark.Util.{debug, warn}
import org.tribbloid.ispark.display.Data
import org.tribbloid.ispark.json.JsonUtil._
import org.tribbloid.ispark.msg._
import org.tribbloid.ispark.msg.formats._
import org.zeromq.{ZMQ, ZMQException}
import play.api.libs.json.{JsResultException, Writes}

class Communication(zmq: Sockets, connection: Profile) {
  private val hmac = HMAC(connection.key, connection.signature_scheme)

  private val DELIMITER = "<IDS|MSG>"

  def send[T<:ToIPython:Writes](socket: ZMQ.Socket, msg: Msg[T]) {
    val idents = msg.idents
    val header = toJSON(msg.header)
    val parent_header = msg.parent_header.map(toJSON(_)).getOrElse("{}")
    val metadata = toJSON(msg.metadata)
    val content = toJSON(msg.content)

    socket.synchronized {
      debug(s"sending: $msg")
      idents.foreach(socket.send(_, ZMQ.SNDMORE))
      socket.send(DELIMITER, ZMQ.SNDMORE)
      socket.send(hmac(header, parent_header, metadata, content), ZMQ.SNDMORE)
      socket.send(header, ZMQ.SNDMORE)
      socket.send(parent_header, ZMQ.SNDMORE)
      socket.send(metadata, ZMQ.SNDMORE)
      socket.send(content)
    }
  }

  def recv(socket: ZMQ.Socket): Option[Msg[FromIPython]] = {
    val (idents, signature, header, parent_header, metadata, content) = socket.synchronized {
      (Stream.continually { socket.recvStr() }.takeWhile(_ != DELIMITER).toList,
        socket.recvStr(),
        socket.recvStr(),
        socket.recvStr(),
        socket.recvStr(),
        socket.recvStr())
    }

    val expectedSignature = hmac(header, parent_header, metadata, content)

    if (signature != expectedSignature) {
      error(s"Invalid HMAC signature, got $signature, expected $expectedSignature")
      None
    } else try {
      val _header = header.as[Header]
      val _parent_header = parent_header.as[Option[Header]]
      val _metadata = metadata.as[Metadata]
      val _content = _header.msg_type match {
        case MsgType.execute_request     => Some(content.as[execute_request])
        case MsgType.complete_request    => Some(content.as[complete_request])
        case MsgType.kernel_info_request => Some(content.as[kernel_info_request])
        case MsgType.object_info_request => Some(content.as[object_info_request])
        case MsgType.connect_request     => Some(content.as[connect_request])
        case MsgType.shutdown_request    => Some(content.as[shutdown_request])
        case MsgType.history_request     => Some(content.as[history_request])
        case MsgType.input_reply         => Some(content.as[input_reply])
        case MsgType.comm_open           => Some(content.as[comm_open])
        case MsgType.comm_msg            => Some(content.as[comm_msg])
        case MsgType.comm_close          => Some(content.as[comm_close])
        case _                           =>
          warn(s"Unexpected message type: ${_header.msg_type}")
          None
      }
      _content.map { _content =>
        val msg = Msg(idents, _header, _parent_header, _metadata, _content)
        debug(s"received: $msg")
        msg
      }
    } catch {
      case e: JsResultException =>
        error(s"JSON deserialization error: ${e.getMessage}")
        None
    }
  }

  def publish[T<:ToIPython:Writes](msg: Msg[T]) = send(zmq.publish, msg)

  def send_status(state: ExecutionState) {
    publish(Msg(
      "status" :: Nil,
      Header(msg_id=UUID.uuid4(),
        username="scala_kernel",
        session=UUID.uuid4(),
        msg_type=MsgType.status),
      None,
      Metadata(),
      status(
        execution_state=state)))
  }

  def send_ok(msg: Msg[_], execution_count: Int) {
    send(zmq.requests, msg.reply(MsgType.execute_reply,
      execute_ok_reply(
        execution_count=execution_count,
        payload=Nil,
        user_expressions=Map.empty)))
  }

  def send_error(msg: Msg[_], execution_count: Int, error: String) {
    send_error(msg, pyerr(execution_count, "", "", error.split("\n").toList))
  }

  def send_error(msg: Msg[_], err: pyerr) {
    publish(msg.pub(MsgType.pyerr, err))
    send(zmq.requests, msg.reply(MsgType.execute_reply,
      execute_error_reply(
        execution_count=err.execution_count,
        ename=err.ename,
        evalue=err.evalue,
        traceback=err.traceback)))
  }

  def send_abort(msg: Msg[_], execution_count: Int) {
    send(zmq.requests, msg.reply(MsgType.execute_reply,
      execute_abort_reply(
        execution_count=execution_count)))
  }

  def send_stream(msg: Msg[_], name: String, data: String) {
    publish(msg.pub(MsgType.stream, stream(name=name, data=data)))
  }

  def send_stdin(msg: Msg[_], prompt: String) {
    send(zmq.stdin, msg.reply(MsgType.input_request, input_request(prompt=prompt)))
  }

  def recv_stdin(): Option[Msg[FromIPython]] = recv(zmq.stdin)

  def send_display_data(msg: Msg[_], data: Data) {
    publish(msg.pub(MsgType.display_data, display_data(source="", data=data, metadata=Map.empty)))
  }

  def silently[T](block: => T) {
    try {
      block
    } catch {
      case _: ZMQException =>
    }
  }

  def busy[T](block: => T): T = {
    send_status(ExecutionState.busy)

    try {
      block
    } finally {
      send_status(ExecutionState.idle)
    }
  }
}
