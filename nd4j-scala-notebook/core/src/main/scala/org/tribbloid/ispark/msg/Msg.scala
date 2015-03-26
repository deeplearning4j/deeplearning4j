package org.tribbloid.ispark.msg

import org.tribbloid.ispark.UUID
import org.tribbloid.ispark.display.{Data, MIME}

object ExecutionStatus extends Enumeration {
  type ExecutionStatus = Value
  val ok = Value
  val error = Value
  val abort = Value
}

object HistAccessType extends Enumeration {
  type HistAccessType = Value
  val range = Value
  val tail = Value
  val search = Value
}

object ExecutionState extends Enumeration {
  type ExecutionState = Value
  val busy = Value
  val idle = Value
  val starting = Value
}

object MsgType extends Enumeration {
  type MsgType = Value

  val execute_request,
  execute_reply,
  object_info_request,
  object_info_reply,
  complete_request,
  complete_reply,
  history_request,
  history_reply,
  connect_request,
  connect_reply,
  kernel_info_request,
  kernel_info_reply,
  shutdown_request,
  shutdown_reply,
  stream,
  display_data,
  pyin,
  pyout,
  pyerr,
  status,
  input_request,
  input_reply,
  comm_open,
  comm_msg,
  comm_close = Value
}

sealed trait Content
sealed trait FromIPython extends Content
sealed trait ToIPython extends Content

case class Header(
                   msg_id: UUID,
                   username: String,
                   session: UUID,
                   msg_type: MsgType)

case class Msg[+T <: Content](
                               idents: List[String], // XXX: Should be List[UUID]?
                               header: Header,
                               parent_header: Option[Header],
                               metadata: Metadata,
                               content: T) {

  private def replyHeader(msg_type: MsgType): Header =
    header.copy(msg_id=UUID.uuid4(), msg_type=msg_type)

  private def replyMsg[T <: ToIPython](idents: List[String], msg_type: MsgType, content: T, metadata: Metadata): Msg[T] =
    Msg(idents, replyHeader(msg_type), Some(header), metadata, content)

  def pub[T <: ToIPython](msg_type: MsgType, content: T, metadata: Metadata=Metadata()): Msg[T] = {
    val tpe = content match {
      case content: stream => content.name
      case _               => msg_type.toString
    }
    replyMsg(tpe :: Nil, msg_type, content, metadata)
  }

  def reply[T <: ToIPython](msg_type: MsgType, content: T, metadata: Metadata=Metadata()): Msg[T] =
    replyMsg(idents, msg_type, content, metadata)
}

case class execute_request(
                            // Source code to be executed by the kernel, one or more lines.
                            code: String,

                            // A boolean flag which, if True, signals the kernel to execute
                            // this code as quietly as possible.  This means that the kernel
                            // will compile the code with 'exec' instead of 'single' (so
                            // sys.displayhook will not fire), forces store_history to be False,
                            // and will *not*:
                            //   - broadcast exceptions on the PUB socket
                            //   - do any logging
                            //
                            // The default is False.
                            silent: Boolean,

                            // A boolean flag which, if True, signals the kernel to populate history
                            // The default is True if silent is False.  If silent is True, store_history
                            // is forced to be False.
                            store_history: Option[Boolean]=None,

                            // A dict mapping names to expressions to be evaluated in the user's dict. The
                            // rich display-data representation of each will be evaluated after execution.
                            // See the display_data content for the structure of the representation data.
                            user_expressions: Map[String, String],

                            // Some frontends (e.g. the Notebook) do not support stdin requests. If
                            // raw_input is called from code executed from such a frontend, a
                            // StdinNotImplementedError will be raised.
                            allow_stdin: Boolean) extends FromIPython

sealed trait execute_reply extends ToIPython {
  // One of: 'ok' OR 'error' OR 'abort'
  val status: ExecutionStatus

  // The global kernel counter that increases by one with each request that
  // stores history.  This will typically be used by clients to display
  // prompt numbers to the user.  If the request did not store history, this will
  // be the current value of the counter in the kernel.
  val execution_count: Int
}

case class execute_ok_reply(
                             execution_count: Int,

                             // 'payload' will be a list of payload dicts.
                             // Each execution payload is a dict with string keys that may have been
                             // produced by the code being executed.  It is retrieved by the kernel at
                             // the end of the execution and sent back to the front end, which can take
                             // action on it as needed.  See main text for further details.
                             payload: List[Map[String, String]],

                             // Results for the user_expressions.
                             user_expressions: Map[String, String]) extends execute_reply {

  val status = ExecutionStatus.ok
}

case class execute_error_reply(
                                execution_count: Int,

                                // Exception name, as a string
                                ename: String,
                                // Exception value, as a string
                                evalue: String,

                                // The traceback will contain a list of frames, represented each as a
                                // string.  For now we'll stick to the existing design of ultraTB, which
                                // controls exception level of detail statefully.  But eventually we'll
                                // want to grow into a model where more information is collected and
                                // packed into the traceback object, with clients deciding how little or
                                // how much of it to unpack.  But for now, let's start with a simple list
                                // of strings, since that requires only minimal changes to ultratb as
                                // written.
                                traceback: List[String]) extends execute_reply {

  val status = ExecutionStatus.error
}

case class execute_abort_reply(
                                execution_count: Int) extends execute_reply {

  val status = ExecutionStatus.abort
}

case class object_info_request(
                                // The (possibly dotted) name of the object to be searched in all
                                // relevant namespaces
                                oname: String,

                                // The level of detail desired.  The default (0) is equivalent to typing
                                // 'x?' at the prompt, 1 is equivalent to 'x??'.
                                detail_level: Int) extends FromIPython

case class ArgSpec(
                    // The names of all the arguments
                    args: List[String],
                    // The name of the varargs (*args), if any
                    varargs: String,
                    // The name of the varkw (**kw), if any
                    varkw: String,
                    // The values (as strings) of all default arguments.  Note
                    // that these must be matched *in reverse* with the 'args'
                    // list above, since the first positional args have no default
                    // value at all.
                    defaults: List[String])

sealed trait object_info_reply extends ToIPython {
  // The name the object was requested under
  val name: String

  // Boolean flag indicating whether the named object was found or not.  If
  // it's false, all other fields will be empty.
  val found: Boolean
}

case class object_info_notfound_reply(
                                       name: String) extends object_info_reply {

  val found = false
}

case class object_info_found_reply(
                                    name: String,

                                    // Flags for magics and system aliases
                                    ismagic: Boolean,
                                    isalias: Boolean,

                                    // The name of the namespace where the object was found ('builtin',
                                    // 'magics', 'alias', 'interactive', etc.)
                                    namespace: String,

                                    // The type name will be type.__name__ for normal Python objects, but it
                                    // can also be a string like 'Magic function' or 'System alias'
                                    type_name: String,

                                    // The string form of the object, possibly truncated for length if
                                    // detail_level is 0
                                    string_form: String,

                                    // For objects with a __class__ attribute this will be set
                                    base_class: String,

                                    // For objects with a __len__ attribute this will be set
                                    length: String,

                                    // If the object is a function, class or method whose file we can find,
                                    // we give its full path
                                    file: String,

                                    // For pure Python callable objects, we can reconstruct the object
                                    // definition line which provides its call signature.  For convenience this
                                    // is returned as a single 'definition' field, but below the raw parts that
                                    // compose it are also returned as the argspec field.
                                    definition: String,

                                    // The individual parts that together form the definition string.  Clients
                                    // with rich display capabilities may use this to provide a richer and more
                                    // precise representation of the definition line (e.g. by highlighting
                                    // arguments based on the user's cursor position).  For non-callable
                                    // objects, this field is empty.
                                    argspec: ArgSpec,

                                    // For instances, provide the constructor signature (the definition of
                                    // the __init__ method):
                                    init_definition: String,

                                    // Docstrings: for any object (function, method, module, package) with a
                                    // docstring, we show it.  But in addition, we may provide additional
                                    // docstrings.  For example, for instances we will show the constructor
                                    // and class docstrings as well, if available.
                                    docstring: String,

                                    // For instances, provide the constructor and class docstrings
                                    init_docstring: String,
                                    class_docstring: String,

                                    // If it's a callable object whose call method has a separate docstring and
                                    // definition line:
                                    call_def: String,
                                    call_docstring: String,

                                    // If detail_level was 1, we also try to find the source code that
                                    // defines the object, if possible.  The string 'None' will indicate
                                    // that no source was found.
                                    source: String) extends object_info_reply {

  val found = true
}

case class complete_request(
                             // The text to be completed, such as 'a.is'
                             // this may be an empty string if the frontend does not do any lexing,
                             // in which case the kernel must figure out the completion
                             // based on 'line' and 'cursor_pos'.
                             text: String,

                             // The full line, such as 'print a.is'.  This allows completers to
                             // make decisions that may require information about more than just the
                             // current word.
                             line: String,

                             // The entire block of text where the line is.  This may be useful in the
                             // case of multiline completions where more context may be needed.  Note: if
                             // in practice this field proves unnecessary, remove it to lighten the
                             // messages.

                             block: Option[String],

                             // The position of the cursor where the user hit 'TAB' on the line.
                             cursor_pos: Int) extends FromIPython

case class complete_reply(
                           // The list of all matches to the completion request, such as
                           // ['a.isalnum', 'a.isalpha'] for the above example.
                           matches: List[String],

                           // the substring of the matched text
                           // this is typically the common prefix of the matches,
                           // and the text that is already in the block that would be replaced by the full completion.
                           // This would be 'a.is' in the above example.
                           matched_text: String,

                           // status should be 'ok' unless an exception was raised during the request,
                           // in which case it should be 'error', along with the usual error message content
                           // in other messages.
                           status: ExecutionStatus) extends ToIPython

case class history_request(
                            // If True, also return output history in the resulting dict.
                            output: Boolean,

                            // If True, return the raw input history, else the transformed input.
                            raw: Boolean,

                            // So far, this can be 'range', 'tail' or 'search'.
                            hist_access_type: HistAccessType,

                            // If hist_access_type is 'range', get a range of input cells. session can
                            // be a positive session number, or a negative number to count back from
                            // the current session.
                            session: Option[Int],

                            // start and stop are line numbers within that session.
                            start: Option[Int],
                            stop: Option[Int],

                            // If hist_access_type is 'tail' or 'search', get the last n cells.
                            n: Option[Int],

                            // If hist_access_type is 'search', get cells matching the specified glob
                            // pattern (with * and ? as wildcards).
                            pattern: Option[String],

                            // If hist_access_type is 'search' and unique is true, do not
                            // include duplicated history.  Default is false.
                            unique: Option[Boolean]) extends FromIPython

case class history_reply(
                          // A list of 3 tuples, either:
                          // (session, line_number, input) or
                          // (session, line_number, (input, output)),
                          // depending on whether output was False or True, respectively.
                          history: List[(Int, Int, Either[String, (String, Option[String])])]) extends ToIPython

case class connect_request() extends FromIPython

case class connect_reply(
                          // The port the shell ROUTER socket is listening on.
                          shell_port: Int,
                          // The port the PUB socket is listening on.
                          iopub_port: Int,
                          // The port the stdin ROUTER socket is listening on.
                          stdin_port: Int,
                          // The port the heartbeat socket is listening on.
                          hb_port: Int) extends ToIPython

case class kernel_info_request() extends FromIPython

case class kernel_info_reply(
                              // Version of messaging protocol (mandatory).
                              // The first integer indicates major version.  It is incremented when
                              // there is any backward incompatible change.
                              // The second integer indicates minor version.  It is incremented when
                              // there is any backward compatible change.
                              protocol_version: (Int, Int),

                              // IPython version number (optional).
                              // Non-python kernel backend may not have this version number.
                              // The last component is an extra field, which may be 'dev' or
                              // 'rc1' in development version.  It is an empty string for
                              // released version.
                              ipython_version: Option[(Int, Int, Int, String)]=None,

                              // Language version number (mandatory).
                              // It is Python version number (e.g., [2, 7, 3]) for the kernel
                              // included in IPython.
                              language_version: List[Int],

                              // Programming language in which kernel is implemented (mandatory).
                              // Kernel included in IPython returns 'python'.
                              language: String) extends ToIPython

case class shutdown_request(
                             // whether the shutdown is final, or precedes a restart
                             restart: Boolean) extends FromIPython

case class shutdown_reply(
                           // whether the shutdown is final, or precedes a restart
                           restart: Boolean) extends ToIPython

case class stream(
                   // The name of the stream is one of 'stdout', 'stderr'
                   name: String,

                   // The data is an arbitrary string to be written to that stream
                   data: String) extends ToIPython

case class display_data(
                         // Who create the data
                         source: String,

                         // The data dict contains key/value pairs, where the kids are MIME
                         // types and the values are the raw data of the representation in that
                         // format.
                         data: Data,

                         // Any metadata that describes the data
                         metadata: Metadata) extends ToIPython

case class pyin(
                 // Source code to be executed, one or more lines
                 code: String,

                 // The counter for this execution is also provided so that clients can
                 // display it, since IPython automatically creates variables called _iN
                 // (for input prompt In[N]).
                 execution_count: Int) extends ToIPython

case class pyout(
                  // The counter for this execution is also provided so that clients can
                  // display it, since IPython automatically creates variables called _N
                  // (for prompt N).
                  execution_count: Int,

                  // data and metadata are identical to a display_data message.
                  // the object being displayed is that passed to the display hook,
                  // i.e. the *result* of the execution.
                  data: Data,
                  metadata: Metadata = Metadata()) extends ToIPython

case class pyerr(
                  execution_count: Int,

                  // Exception name, as a string
                  ename: String,
                  // Exception value, as a string
                  evalue: String,

                  // The traceback will contain a list of frames, represented each as a
                  // string.  For now we'll stick to the existing design of ultraTB, which
                  // controls exception level of detail statefully.  But eventually we'll
                  // want to grow into a model where more information is collected and
                  // packed into the traceback object, with clients deciding how little or
                  // how much of it to unpack.  But for now, let's start with a simple list
                  // of strings, since that requires only minimal changes to ultratb as
                  // written.
                  traceback: List[String]) extends ToIPython

object pyerr {
  // XXX: can't use apply(), because of https://github.com/playframework/playframework/issues/2031
  def fromThrowable(execution_count: Int, exception: Throwable): pyerr = {
    val name = exception.getClass.getName
    val value = Option(exception.getMessage) getOrElse ""
    val stacktrace = exception
      .getStackTrace
      .takeWhile(_.getFileName != "<console>")
      .toList
    val traceback = s"$name: $value" :: stacktrace.map("    " + _)

    pyerr(execution_count=execution_count,
      ename=name,
      evalue=value,
      traceback=traceback)
  }
}

case class status(
                   // When the kernel starts to execute code, it will enter the 'busy'
                   // state and when it finishes, it will enter the 'idle' state.
                   // The kernel will publish state 'starting' exactly once at process startup.
                   execution_state: ExecutionState) extends ToIPython

case class clear_output(
                         // Wait to clear the output until new output is available.  Clears the
                         // existing output immediately before the new output is displayed.
                         // Useful for creating simple animations with minimal flickering.
                         _wait: Boolean) extends ToIPython

case class input_request(
                          prompt: String) extends ToIPython

case class input_reply(
                        value: String) extends FromIPython

import play.api.libs.json.JsObject

case class comm_open(
                      comm_id: UUID,
                      target_name: String,
                      data: JsObject) extends ToIPython with FromIPython

case class comm_msg(
                     comm_id: UUID,
                     data: JsObject) extends ToIPython with FromIPython

case class comm_close(
                       comm_id: UUID,
                       data: JsObject) extends ToIPython with FromIPython

// XXX: This was originally in src/main/scala/Formats.scala, but due to
// a bug in the compiler related to `knownDirectSubclasses` and possibly
// also other bugs (e.g. `isCaseClass`), formats had to be moved here
// and explicit type annotations had to be added for formats of sealed
// traits. Otherwise no known subclasses will be reported.

import org.tribbloid.ispark.json.{EnumJson, Json}
import play.api.libs.json.{JsObject, Writes}

package object formats {

  implicit val MIMEFormat = new Writes[MIME] {
    def writes(mime: MIME) = implicitly[Writes[String]].writes(mime.name)
  }

  implicit val DataFormat = new Writes[Data] {
    def writes(data: Data) = {
      JsObject(data.items.map { case (mime, value) =>
        mime.name -> implicitly[Writes[String]].writes(value)
      })
    }
  }

  import org.tribbloid.ispark.json.JsonImplicits._

  implicit val MsgTypeFormat = EnumJson.format(MsgType)
  implicit val HeaderFormat = Json.format[Header]

  implicit val ExecutionStatusFormat = EnumJson.format(ExecutionStatus)
  implicit val ExecutionStateFormat = EnumJson.format(ExecutionState)
  implicit val HistAccessTypeFormat = EnumJson.format(HistAccessType)

  implicit val ArgSpecFormat = Json.format[ArgSpec]

  implicit val ExecuteRequestJSON = Json.format[execute_request]
  implicit val ExecuteReplyJSON: Writes[execute_reply] = Json.writes[execute_reply]

  implicit val ObjectInfoRequestJSON = Json.format[object_info_request]
  implicit val ObjectInfoReplyJSON: Writes[object_info_reply] = Json.writes[object_info_reply]

  implicit val CompleteRequestJSON = Json.format[complete_request]
  implicit val CompleteReplyJSON = Json.format[complete_reply]

  implicit val HistoryRequestJSON = Json.format[history_request]
  implicit val HistoryReplyJSON = Json.format[history_reply]

  implicit val ConnectRequestJSON = Json.noFields[connect_request]
  implicit val ConnectReplyJSON = Json.format[connect_reply]

  implicit val KernelInfoRequestJSON = Json.noFields[kernel_info_request]
  implicit val KernelInfoReplyJSON = Json.format[kernel_info_reply]

  implicit val ShutdownRequestJSON = Json.format[shutdown_request]
  implicit val ShutdownReplyJSON = Json.format[shutdown_reply]

  implicit val StreamJSON = Json.writes[stream]
  implicit val DisplayDataJSON = Json.writes[display_data]
  implicit val PyinJSON = Json.writes[pyin]
  implicit val PyoutJSON = Json.writes[pyout]
  implicit val PyerrJSON = Json.writes[pyerr]
  implicit val StatusJSON = Json.writes[status]
  implicit val ClearOutputJSON = new Writes[clear_output] {
    def writes(obj: clear_output) = {
      // NOTE: `wait` is a final member on Object, so we have to go through hoops
      JsObject(Seq("wait" -> implicitly[Writes[Boolean]].writes(obj._wait)))
    }
  }

  implicit val InputRequestJSON = Json.format[input_request]
  implicit val InputReplyJSON = Json.format[input_reply]

  implicit val CommOpenJSON = Json.format[comm_open]
  implicit val CommMsgJSON = Json.format[comm_msg]
  implicit val CommCloseJSON = Json.format[comm_close]
}
