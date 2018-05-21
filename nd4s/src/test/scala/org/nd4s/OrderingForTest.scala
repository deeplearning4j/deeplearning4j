package org.nd4s

import org.scalatest.{Outcome, Suite, SuiteMixin}

trait OrderingForTest extends SuiteMixin{this:Suite =>
  val ordering:NDOrdering
}
trait COrderingForTest extends OrderingForTest{this:Suite =>
  override val ordering: NDOrdering = NDOrdering.C
}
trait FortranOrderingForTest extends OrderingForTest{this:Suite =>
  override val ordering: NDOrdering = NDOrdering.Fortran
}

