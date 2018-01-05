package org.deeplearning4j.spark.ml.utils

import org.apache.hadoop.fs.Path
import org.apache.spark.SparkContext
import org.apache.spark.ml.param.{ParamPair, Params}
import org.apache.spark.ml.util.{MLReadable, MLReader}
import org.apache.spark.sql.types._
import org.deeplearning4j.spark.impl.paramavg.ParameterAveragingTrainingMaster
import org.json4s.JsonDSL._
import org.json4s.jackson.JsonMethods.{compact, parse, render}
import org.json4s.{JObject, JValue, _}

/**
  * This is imported from Spark with a few minor changes, since the libraries are private.
  * This is a known constraint in spark around params. https://issues.apache.org/jira/browse/SPARK-7146
  */
object SparkDl4jUtil {

    def saveMetadata(instance: Params,
                     path: String,
                     sc: SparkContext,
                     extraMetadata: Option[JObject] = None,
                     paramMap: Option[JValue] = None): Unit = {
        val uid = instance.uid
        val cls = instance.getClass.getName
        val params = instance.extractParamMap().toSeq.asInstanceOf[Seq[ParamPair[Any]]]
        val jsonParams = paramMap.getOrElse(render(params.map { case ParamPair(p, v) =>
            p.name -> parse(p.jsonEncode(v))
        }.toList))
        val basicMetadata = ("class" -> cls) ~
            ("timestamp" -> System.currentTimeMillis()) ~
            ("sparkVersion" -> sc.version) ~
            ("uid" -> uid) ~
            ("paramMap" -> jsonParams)
        val metadata = extraMetadata match {
            case Some(jObject) =>
                basicMetadata ~ jObject
            case None =>
                basicMetadata
        }
        val metadataPath = new Path(path + "_metadata").toString
        val metadataJson = compact(render(metadata))
        sc.parallelize(Seq(metadataJson), 1).saveAsTextFile(metadataPath)
    }

    /**
      * All info from metadata file.
      * @param params  paramMap, as a [[JValue]]
      * @param metadata  All metadata, including the other fields
      * @param metadataJson  Full metadata file String (for debugging)
      */
    case class Metadata(
                           className: String,
                           uid: String,
                           timestamp: Long,
                           sparkVersion: String,
                           params: JValue,
                           metadata: JValue,
                           metadataJson: String)

    /**
      * Load metadata from file.
      * @param expectedClassName  If non empty, this is checked against the loaded metadata.
      * @throws IllegalArgumentException if expectedClassName is specified and does not match metadata
      */
    def loadMetadata(path: String, sc: SparkContext, expectedClassName: String = ""): Metadata = {
        val metadataPath = new Path(path + "_metadata").toString
        val metadataStr = sc.textFile(metadataPath, 1).first()
        val metadata = parse(metadataStr)

        implicit val format = DefaultFormats
        val className = (metadata \ "class").extract[String]
        val uid = (metadata \ "uid").extract[String]
        val timestamp = (metadata \ "timestamp").extract[Long]
        val sparkVersion = (metadata \ "sparkVersion").extract[String]
        val params = metadata \ "paramMap"
        if (expectedClassName.nonEmpty) {
            require(className == expectedClassName, s"Error loading metadata: Expected class name" +
                s" $expectedClassName but found class name $className")
        }

        Metadata(className, uid, timestamp, sparkVersion, params, metadata, metadataStr)
    }

    /**
      * Extract Params from metadata, and set them in the instance.
      * This works if all Params implement [[org.apache.spark.ml.param.Param.jsonDecode()]].
      */
    def getAndSetParams(instance: Params, metadata: Metadata): Unit = {
        implicit val format = DefaultFormats
        metadata.params match {
            case JObject(pairs) =>
                pairs.foreach { case (paramName, jsonValue) =>
                    val param = instance.getParam(paramName)
                    val value = param.jsonDecode(compact(render(jsonValue)))
                    instance.set(param, value)
                }
            case _ =>
                throw new IllegalArgumentException(
                    s"Cannot recognize JSON metadata: ${metadata.metadataJson}.")
        }
    }

    /**
      * Load a [[Params]] instance from the given path, and return it.
      * This assumes the instance implements [[MLReadable]].
      */
    def loadParamsInstance[T](path: String, sc: SparkContext): T = {
        val metadata = loadMetadata(path, sc)
        val cls = classForName(metadata.className)
        cls.getMethod("read").invoke(null).asInstanceOf[MLReader[T]].load(path)
    }

    def classForName(className: String): Class[_] = {
        Class.forName(className, true, getContextOrSparkClassLoader)
        // scalastyle:on classforname
    }

    def getContextOrSparkClassLoader: ClassLoader =
        Option(Thread.currentThread().getContextClassLoader).getOrElse(getSparkClassLoader)

    def getSparkClassLoader: ClassLoader = getClass.getClassLoader

    def createScheme() : StructType = {
        new StructType(Array(
            StructField("mlc", DataTypes.StringType, false),
            StructField("params", ArrayType.apply(DataTypes.DoubleType), false)
        ))
    }
}

object SchemaUtils {
    def appendColumn(schema: StructType, colName: String, dataType: DataType, nullable: Boolean = false): StructType = {
        if (colName.isEmpty) return schema
        appendColumn(schema, StructField(colName, dataType, nullable))
    }

    def appendColumn(schema: StructType, col: StructField): StructType = {
        require(!schema.fieldNames.contains(col.name), s"Column ${col.name} already exists.")
        StructType(schema.fields :+ col)
    }
}

trait ParamSerializer extends Serializable {
    def apply() : ParameterAveragingTrainingMaster
}
