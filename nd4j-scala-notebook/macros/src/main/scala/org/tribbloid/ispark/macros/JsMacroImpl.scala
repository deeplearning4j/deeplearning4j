package org.tribbloid.ispark.macros

import play.api.libs.json.{Format, JsMacroImpl => PlayMacroImpl, Reads, Writes}

import scala.language.experimental.macros
import scala.reflect.macros.Context

trait JsonImpl {
    def reads[A]:  Reads[A]  = macro PlayMacroImpl.readsImpl[A]
    def writes[A]: Writes[A] = macro JsMacroImpl.sealedWritesImpl[A]
    def format[A]: Format[A] = macro PlayMacroImpl.formatImpl[A]
}

object JsMacroImpl {

    /* JSON writer for sealed traits.
     *
     * This macro generates code equivalent to:
     * ```
     * new Writes[T] {
     *     val $writes$T_1 = Json.writes[T_1]
     *     ...
     *     val $writes$T_n = Json.writes[T_n]
     *
     *     def writes(obj: T) = (obj match {
     *         case o: T_1 => $writes$T_1.writes(o)
     *         ...
     *         case o: T_n => $writes$T_n.writes(o)
     *     }) ++ JsObject(List(
     *         ("field_1", Json.toJson(obj.field_1)),
     *         ...
     *         ("field_n", Json.toJson(obj.field_n))))
     * }
     * ```
     *
     * `T` is a sealed trait with case subclasses `T_1`, ... `T_n`. Fields `field_1`,
     * ..., `field_n` are `T`'s vals that don't appear in `T_i` constructors.
     */
    def sealedWritesImpl[T: c.WeakTypeTag](c: Context): c.Expr[Writes[T]] = {
        import c.universe._

        val tpe = weakTypeOf[T]
        val symbol = tpe.typeSymbol

        if (!symbol.isClass) {
            c.abort(c.enclosingPosition, "expected a class or trait")
        }

        val cls = symbol.asClass

        if (!cls.isTrait) {
            PlayMacroImpl.writesImpl[T](c)
        } else if (!cls.isSealed) {
            c.abort(c.enclosingPosition, "expected a sealed trait")
        } else {
            val children = cls.knownDirectSubclasses.toList

            if (children.isEmpty) {
                c.abort(c.enclosingPosition, "trait has no subclasses")
            } else if (!children.forall(_.isClass) || !children.map(_.asClass).forall(_.isCaseClass)) {
                c.abort(c.enclosingPosition, "all children must be case classes")
            } else {
                val named = children.map { child =>
                    (child, newTermName("$writes$" + child.name.toString))
                }

                val valDefs = named.map { case (child, name) =>
                    q"val $name = play.api.libs.json.Json.writes[$child]"
                }

                val caseDefs = named.map { case (child, name) =>
                    CaseDef(
                        Bind(newTermName("o"), Typed(Ident(nme.WILDCARD),
                             Ident(child))),
                        EmptyTree,
                        q"$name.writes(o)")
                }

                val names = children.flatMap(
                    _.typeSignature
                     .declaration(nme.CONSTRUCTOR)
                     .asMethod
                     .paramss(0)
                     .map(_.name.toString)
                 ).toSet

                val fieldNames = cls.typeSignature
                   .declarations
                   .toList
                   .filter(_.isMethod)
                   .map(_.asMethod)
                   .filter(_.isStable)
                   .filter(_.isPublic)
                   .map(_.name.toString)
                   .filterNot(names contains _)

                val fieldDefs = fieldNames.map { fieldName =>
                    val name = newTermName(fieldName)
                    q"($fieldName, play.api.libs.json.Json.toJson(obj.$name))"
                }

                val matchDef = Match(q"obj", caseDefs)

                c.Expr[Writes[T]](
                    q"""
                    new Writes[$symbol] {
                        ..$valDefs

                        def writes(obj: $symbol) =
                            $matchDef ++ play.api.libs.json.JsObject(List(..$fieldDefs))
                    }
                    """)
            }
        }
    }
}
