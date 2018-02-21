package org.deeplearning4j.scalnet.regularizers

/**
  * Weight regularizers.
  *
  * @author David Kale
  */
sealed class WeightRegularizer(val l1: Double = Double.NaN, val l2: Double = Double.NaN)

case class NoRegularizer() extends WeightRegularizer()
case class L1(l: Double = 0.01) extends WeightRegularizer(l1 = l)
case class L2(l: Double = 0.01) extends WeightRegularizer(l2 = l)
case class L1L2(override val l1: Double = 0.01, override val l2: Double = 0.01)
    extends WeightRegularizer
