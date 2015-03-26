package org.tribbloid.ispark.macros

import scala.annotation.StaticAnnotation
import scala.language.experimental.macros
import scala.reflect.macros.Context

trait EnumType {
    val name: String = toString
}

trait LowerCase { self: EnumType =>
    override val name = toString.toLowerCase
}

trait SnakeCase { self: EnumType =>
    override val name = Utils.snakify(toString)
}

trait Enumerated[T <: EnumType] {
    type ValueType = T

    val values: Set[T]
    val fromString: PartialFunction[String, T]

    final def unapply(name: String): Option[T] = fromString.lift(name)

    override def toString: String = {
        val name = getClass.getSimpleName.stripSuffix("$")
        s"$name(${values.map(_.name).mkString(", ")})"
    }
}

class enum extends StaticAnnotation {
    def macroTransform(annottees: Any*): Any = macro EnumImpl.enumTransformImpl
}

object EnumImpl {
    def enumTransformImpl(c: Context)(annottees: c.Expr[Any]*): c.Expr[Any] = {
        import c.universe._

        annottees.map(_.tree) match {
            case ModuleDef(mods, name, tpl @ Template(parents, sf, body)) :: Nil =>
                val enumImpl = reify { EnumImpl }
                val methods = List(
                    q"final val values: Set[ValueType] = $enumImpl.values[ValueType]",
                    q"final val fromString: PartialFunction[String, ValueType] = $enumImpl.fromString[ValueType]")
                val module = ModuleDef(mods, name, Template(parents, sf, body ++ methods))
                c.Expr[Any](Block(module :: Nil, Literal(Constant(()))))
            case _ => c.abort(c.enclosingPosition, "@enum annotation can only be applied to an object")
        }
    }

    private def children[T <: EnumType : c.WeakTypeTag](c: Context): Set[c.universe.Symbol] = {
        import c.universe._

        val tpe = weakTypeOf[T]
        val cls = tpe.typeSymbol.asClass

        if (!cls.isSealed) c.error(c.enclosingPosition, "must be a sealed trait or class")
        val children = tpe.typeSymbol.asClass.knownDirectSubclasses
        if (children.isEmpty) c.error(c.enclosingPosition, "no enumerations found")

        children
    }

    def values[T <: EnumType]: Set[T] = macro EnumImpl.valuesImpl[T]

    def valuesImpl[T <: EnumType : c.WeakTypeTag](c: Context): c.Expr[Set[T]] = {
        import c.universe._

        val tpe = weakTypeOf[T]
        val values = children[T](c).map(_.name.toTermName)

        c.Expr[Set[T]](q"Set[$tpe](..$values)")
    }

    def fromString[T <: EnumType]: PartialFunction[String, T] = macro EnumImpl.fromStringImpl[T]

    def fromStringImpl[T <: EnumType : c.WeakTypeTag](c: Context): c.Expr[PartialFunction[String, T]] = {
        import c.universe._

        val tpe = weakTypeOf[T]
        val cases = children[T](c).map { child => cq"${child.name.toTermName}.name => ${child.name.toTermName}" }

        c.Expr[PartialFunction[String, T]](q"{ case ..$cases }: PartialFunction[String, $tpe]")
    }
}
