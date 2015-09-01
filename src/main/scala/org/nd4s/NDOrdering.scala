package org.nd4s

import org.nd4j.linalg.factory.NDArrayFactory

sealed trait NDOrdering {
  val value:Char
}
object NDOrdering{
  case object Fortran extends NDOrdering{
    override val value: Char = NDArrayFactory.FORTRAN
  }
  case object C extends NDOrdering{
    override val value: Char = NDArrayFactory.C
  }
  def apply(char:Char):NDOrdering = char.toLower match{
    case 'c' => C
    case 'f' => Fortran
    case _ => throw new IllegalArgumentException("NDimensional Ordering accepts only 'c' or 'f'.")
  }
}
