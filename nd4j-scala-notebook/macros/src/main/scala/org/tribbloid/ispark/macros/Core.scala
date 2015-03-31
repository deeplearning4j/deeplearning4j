package org.tribbloid.ispark.macros

import scala.language.experimental.macros
import scala.reflect.macros.Context

object Core {
    def implicitlyOptImpl[T: c.WeakTypeTag](c: Context): c.Expr[Option[T]] = {
        import c.universe._

        c.inferImplicitValue(weakTypeOf[T]) match {
            case EmptyTree => reify { None }
            case impl      => reify { Some(c.Expr[T](impl).splice) }
        }
    }

    def implicitlyOpt[T]: Option[T] = macro Core.implicitlyOptImpl[T]
}
