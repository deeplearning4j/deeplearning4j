package org.tribbloid.ispark.json

import org.tribbloid.ispark.UUID
import org.tribbloid.ispark.macros.JsonImpl
import play.api.libs.json.{Format, JsArray, JsError, JsObject, JsPath, JsResult, JsString, JsSuccess, JsValue, Json => PlayJson, OWrites, Reads, Writes}

import scala.reflect.ClassTag

object JsonUtil {
    def toJSON[T:Writes](obj: T): String =
        PlayJson.stringify(PlayJson.toJson(obj))

    def fromJSON[T:Reads](json: String): T =
        PlayJson.parse(json).as[T]

    implicit class JsonString(json: String) {
        def as[T:Reads] = fromJSON[T](json)
    }
}

object Json extends JsonImpl {
    import play.api.libs.json.Json.JsValueWrapper

    def fromJson[T:Reads](json: JsValue): JsResult[T] = PlayJson.fromJson(json)
    def toJson[T:Writes](obj: T): JsValue = PlayJson.toJson(obj)

    def obj(fields: (String, JsValueWrapper)*): JsObject = PlayJson.obj(fields: _*)
    def arr(fields: JsValueWrapper*): JsArray = PlayJson.arr(fields: _*)

    def noFields[A:ClassTag]: Format[A] = NoFields.format
}

object NoFields {
    def reads[T:ClassTag]: Reads[T] = new Reads[T] {
        def reads(json: JsValue) = json match {
            case JsObject(seq) if seq.isEmpty =>
                JsSuccess(implicitly[ClassTag[T]].runtimeClass.newInstance.asInstanceOf[T])
            case _ =>
                JsError("Not an empty object")
        }
    }

    def writes[T]: OWrites[T] = new OWrites[T] {
        def writes(t: T) = JsObject(Nil)
    }

    def format[T:ClassTag]: Format[T] = {
        Format(reads, writes)
    }
}

object EnumJson {
    def reads[E <: Enumeration](enum: E): Reads[E#Value] = new Reads[E#Value] {
        def reads(json: JsValue): JsResult[E#Value] = json match {
            case JsString(string) =>
                try {
                    JsSuccess(enum.withName(string))
                } catch {
                    case _: NoSuchElementException =>
                        JsError(s"Enumeration expected of type: ${enum.getClass}, but it does not appear to contain the value: $string")
                }
            case _ =>
                JsError("Value of type String expected")
        }
    }

    def writes[E <: Enumeration]: Writes[E#Value] = new Writes[E#Value] {
        def writes(value: E#Value): JsValue = JsString(value.toString)
    }

    def format[E <: Enumeration](enum: E): Format[E#Value] = {
        Format(reads(enum), writes)
    }
}

trait EitherJson {
    implicit def EitherReads[T1:Reads, T2:Reads]: Reads[Either[T1, T2]] = new Reads[Either[T1, T2]] {
        def reads(json: JsValue) = {
            implicitly[Reads[T1]].reads(json) match {
                case JsSuccess(left, _) => JsSuccess(Left(left))
                case _ =>
                    implicitly[Reads[T2]].reads(json) match {
                        case JsSuccess(right, _) => JsSuccess(Right(right))
                        case _ => JsError("Either[T1, T2] failed")
                    }
            }
        }
    }

    implicit def EitherWrites[T1:Writes, T2:Writes]: Writes[Either[T1, T2]] = new Writes[Either[T1, T2]] {
        def writes(t: Either[T1, T2]) = t match {
            case Left(left) => implicitly[Writes[T1]].writes(left)
            case Right(right) => implicitly[Writes[T2]].writes(right)
        }
    }
}

trait TupleJson {
    implicit def Tuple1Reads[T1:Reads]: Reads[Tuple1[T1]] = new Reads[Tuple1[T1]] {
        def reads(json: JsValue) = json match {
            case JsArray(List(j1)) =>
                (implicitly[Reads[T1]].reads(j1)) match {
                    case JsSuccess(v1, _) => JsSuccess(new Tuple1(v1))
                    case e1: JsError => e1
            }
            case _ => JsError("Not an array")
        }
    }

    implicit def Tuple1Writes[T1:Writes]: Writes[Tuple1[T1]] = new Writes[Tuple1[T1]] {
        def writes(t: Tuple1[T1]) = JsArray(Seq(implicitly[Writes[T1]].writes(t._1)))
    }

    implicit def Tuple2Reads[T1:Reads, T2:Reads]: Reads[(T1, T2)] = new Reads[(T1, T2)] {
        def reads(json: JsValue) = json match {
            case JsArray(List(j1, j2)) =>
                (implicitly[Reads[T1]].reads(j1),
                 implicitly[Reads[T2]].reads(j2)) match {
                    case (JsSuccess(v1, _), JsSuccess(v2, _)) => JsSuccess((v1, v2))
                    case (e1: JsError, _) => e1
                    case (_, e2: JsError) => e2
            }
            case _ => JsError("Not an array")
        }
    }

    implicit def Tuple2Writes[T1:Writes, T2:Writes]: Writes[(T1, T2)] = new Writes[(T1, T2)] {
        def writes(t: (T1, T2)) = JsArray(Seq(implicitly[Writes[T1]].writes(t._1),
                                              implicitly[Writes[T2]].writes(t._2)))
    }

    implicit def Tuple3Reads[T1:Reads, T2:Reads, T3:Reads]: Reads[(T1, T2, T3)] = new Reads[(T1, T2, T3)] {
        def reads(json: JsValue) = json match {
            case JsArray(List(j1, j2, j3)) =>
                (implicitly[Reads[T1]].reads(j1),
                 implicitly[Reads[T2]].reads(j2),
                 implicitly[Reads[T3]].reads(j3)) match {
                    case (JsSuccess(v1, _), JsSuccess(v2, _), JsSuccess(v3, _)) => JsSuccess((v1, v2, v3))
                    case (e1: JsError, _, _) => e1
                    case (_, e2: JsError, _) => e2
                    case (_, _, e3: JsError) => e3
            }
            case _ => JsError("Not an array")
        }
    }

    implicit def Tuple3Writes[T1:Writes, T2:Writes, T3:Writes]: Writes[(T1, T2, T3)] = new Writes[(T1, T2, T3)] {
        def writes(t: (T1, T2, T3)) = JsArray(Seq(implicitly[Writes[T1]].writes(t._1),
                                                  implicitly[Writes[T2]].writes(t._2),
                                                  implicitly[Writes[T3]].writes(t._3)))
    }

    implicit def Tuple4Reads[T1:Reads, T2:Reads, T3:Reads, T4:Reads]: Reads[(T1, T2, T3, T4)] = new Reads[(T1, T2, T3, T4)] {
        def reads(json: JsValue) = json match {
            case JsArray(List(j1, j2, j3, j4)) =>
                (implicitly[Reads[T1]].reads(j1),
                 implicitly[Reads[T2]].reads(j2),
                 implicitly[Reads[T3]].reads(j3),
                 implicitly[Reads[T4]].reads(j4)) match {
                    case (JsSuccess(v1, _), JsSuccess(v2, _), JsSuccess(v3, _), JsSuccess(v4, _)) => JsSuccess((v1, v2, v3, v4))
                    case (e1: JsError, _, _, _) => e1
                    case (_, e2: JsError, _, _) => e2
                    case (_, _, e3: JsError, _) => e3
                    case (_, _, _, e4: JsError) => e4
            }
            case _ => JsError("Not an array")
        }
    }

    implicit def Tuple4Writes[T1:Writes, T2:Writes, T3:Writes, T4:Writes]: Writes[(T1, T2, T3, T4)] = new Writes[(T1, T2, T3, T4)] {
        def writes(t: (T1, T2, T3, T4)) = JsArray(Seq(implicitly[Writes[T1]].writes(t._1),
                                                      implicitly[Writes[T2]].writes(t._2),
                                                      implicitly[Writes[T3]].writes(t._3),
                                                      implicitly[Writes[T4]].writes(t._4)))
    }
}

trait MapJson {
    implicit def MapReads[V:Reads]: Reads[Map[String, V]] = new Reads[Map[String, V]] {
        def reads(json: JsValue) = json match {
            case JsObject(obj) =>
                var hasErrors = false

                val r = obj.map { case (key, value) =>
                    implicitly[Reads[V]].reads(value) match {
                        case JsSuccess(v, _) => Right(key -> v)
                        case JsError(e) =>
                            hasErrors = true
                            Left(e.map { case (p, valerr) => (JsPath \ key) ++ p -> valerr })
                    }
                }

                if (hasErrors) {
                    JsError(r.collect { case Left(t) => t }.reduceLeft((acc, v) => acc ++ v))
                } else {
                    JsSuccess(r.collect { case Right(t) => t }.toMap)
                }
            case _ => JsError("Not an object")
        }
    }

    implicit def MapWrites[V:Writes]: OWrites[Map[String, V]] = new OWrites[Map[String, V]] {
        def writes(t: Map[String, V]) =
            JsObject(t.map { case (k, v) => (k, implicitly[Writes[V]].writes(v)) }.toList)
    }
}

trait UUIDJson {
    implicit val UUIDReads: Reads[UUID] = new Reads[UUID] {
        def reads(json: JsValue) = json match {
            case JsString(uuid) =>
                UUID.fromString(uuid)
                    .map(JsSuccess(_))
                    .getOrElse{JsError(s"Invalid UUID string: $uuid")}
            case _ => JsError("Not a string")
        }
    }

    implicit val UUIDWrites: Writes[UUID] = new Writes[UUID] {
        def writes(t: UUID) = JsString(t.toString)
    }
}

trait JsonImplicits extends EitherJson with TupleJson with MapJson with UUIDJson
object JsonImplicits extends JsonImplicits
