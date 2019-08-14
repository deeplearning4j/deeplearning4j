package org.nd4s.samediff

import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j

class SameDiff[A <: INDArray] {

  var function2: Function2[A, A, A] = {null}

  def this(op: ((A, A) => A)) {
    this
    function2 = op
  }

  var places = List[A]()

  def bind(place: A): Unit = {
    places = place :: places
  }

  def exec(): A = {
    if (places.length == 2)
       function2(places.head, places.tail.head)
    else
      Nd4j.empty().asInstanceOf[A]
  }
}