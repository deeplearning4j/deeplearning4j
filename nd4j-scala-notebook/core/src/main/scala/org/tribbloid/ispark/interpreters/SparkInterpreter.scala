package org.tribbloid.ispark.interpreters

import org.apache.spark.SparkContext
import org.apache.spark.repl._
import org.tribbloid.ispark.Util
import org.tribbloid.ispark.display.Data

import scala.collection.immutable
import scala.language.postfixOps
import scala.tools.nsc.Settings
import scala.tools.nsc.interpreter._
import scala.tools.nsc.util.ScalaClassLoader


class SparkInterpreter(args: Array[String], usejavacp: Boolean=true, val appName: String = "ISpark") {

  val output = new java.io.StringWriter
  val printer = new java.io.PrintWriter(output)

  val iLoop = new SparkILoop(None, printer, None)
  assert(process(args))

  var sc: SparkContext = _

  def intp = iLoop.intp

  /** process command-line arguments and do as they request */
  private def process(args: Array[String]): Boolean = {
    val cl = new SparkCommandLine(args.toList, println(_))
    cl.settings.embeddedDefaults[this.type]
    cl.settings.usejavacp.value = usejavacp

    cl.ok && process(cl.settings)
  }

  private def process(settings: Settings): Boolean = ScalaClassLoader.savingContextLoader {
    if (getMaster == "yarn-client") System.setProperty("SPARK_YARN_MODE", "true")

    iLoop.settings = settings

    iLoop.createInterpreter()

    initializeSpark()

    // it is broken on startup; go ahead and exit
    if (intp.reporter.hasErrors) return false

    intp.initializeSynchronous()

    true
  }

  protected def initializeSpark() {
    sc = iLoop.createSparkContext()
    quietBind(NamedParam[SparkContext]("sc", sc), immutable.List("@transient")) match {
      case IR.Success =>
      case _ => throw new RuntimeException("Spark failed to initialize")
    }

    interpret("""
                |import org.apache.spark.SparkContext._
                |import org.tribbloid.ispark.display.Display
              """.stripMargin) match {
      case _: Results.Success =>
      case Results.Exception(ee) => throw new RuntimeException("SparkContext failed to be imported", ee)
      case _ => throw new RuntimeException("SparkContext failed to be imported\n"+this.output.toString)
    }
  }

  private def getMaster: String = {
    val envMaster = sys.env.get("MASTER")
    val propMaster = sys.props.get("spark.master")
    propMaster.orElse(envMaster).getOrElse("local[*]")
  }


  protected def sparkCleanUp(): Unit = {
    if (sc != null)
      sc.stop()
  }

  def close(): Unit = {
    if (intp ne null) {
      sparkCleanUp()
      intp.close()
      iLoop.intp = null
    }
  }

  override def finalize() {
    try{
      close()
    }catch{
      case e: Throwable => Util.log("FINALIZE FAILED! " + e);
    }finally{
      super.finalize()
    }
  }

  def resetOutput() {
    output.getBuffer.setLength(0)
  }

  private val completion = new SparkJLineCompletion(intp)

  def completions(input: String): List[String] = completion.topLevelFor(Parsed.dotted(input, input.length))

  def interpret(line: String, synthetic: Boolean = false): Results.Result = {

    val res = try{
      intp.interpret(line, synthetic)
    }
    catch {
      case e: Throwable => return Results.Exception(e)
    }

    res match {
      case IR.Incomplete => Results.Incomplete
      case IR.Error =>
        Results.Error
      case IR.Success =>
        val mostRecentVar = intp.mostRecentVar
        if (mostRecentVar == "") Results.NoValue
        else {
          val value = intp.valueOfTerm(mostRecentVar)
          value match {
            case None => Results.NoValue
            case Some(v) =>
              val tpe = intp.typeOfTerm(mostRecentVar)

              val data = Data.parse(v)
              val result = Results.Value(v, tpe.toString(), data)
              result
          }
        }
    }
  }

  def bind(name: String, boundType: String, value: Any, modifiers: List[String] = Nil): IR.Result = intp.bind(name, boundType, value, modifiers)

  def bind(p: NamedParam, modifiers: List[String]): IR.Result = bind(p.name, p.tpe, p.value, modifiers)

  def quietBind(p: NamedParam, modifiers: List[String]): IR.Result = intp.beQuietDuring(bind(p, modifiers))

  //  def compile(code: String): Boolean = {
  //    val intp = intp
  //
  //    val imports = intp.definedTypes ++ intp.definedTerms match {
  //      case Nil => "/* imports */"
  //      case names => names.map(intp.pathToName).map("import " + _).mkString("\n  ")
  //    }
  //
  //    val source = s"""
  //            |$imports
  //            |
  //            |$code
  //            """.stripMargin
  //
  //    val bindRep = new intp.ReadEvalPrint()
  //    bindRep.compile(source)
  //  }

  def cancel() = {}

  def stringify(obj: Any): String = unmangle(obj.toString)

  private def unmangle(string: String): String = intp.naming.unmangle(string)

  def typeInfo(code: String, deconstruct: Boolean): Option[String] = {

    val intp0 = this.intp

    import intp0.global._
    val symbol = intp0.symbolOfLine(code)

    if (symbol.exists) {
      Some(afterTyper {
        val info = symbol.info match {
          case NullaryMethodType(restpe) if symbol.isAccessor => restpe
          case _                                           => symbol.info
        }
        stringify(if (deconstruct) intp0.deconstruct.show(info) else info)
      })
    } else None
  }
}

