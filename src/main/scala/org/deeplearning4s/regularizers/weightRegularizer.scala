package org.deeplearning4s.regularizers

/**
  * Weight regularizers.
  */

sealed class WeightRegularizer(val l1: Double = Double.NaN, val l2: Double = Double.NaN)

case class NoRegularizer() extends WeightRegularizer()
case class l1(l: Double = 0.01) extends WeightRegularizer(l1 = l)
case class l2(l: Double = 0.01) extends WeightRegularizer(l2 = l)
case class l1l2(override val l1: Double = 0.01, override val l2: Double = 0.01) extends WeightRegularizer
