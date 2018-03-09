package org.deeplearning4j.scalnet.layers

import org.deeplearning4j.nn.conf.layers
import org.deeplearning4j.nn.conf.layers.recurrent.Bidirectional.Mode

class Bidirectional(layer: Layer, mode: Mode, override val name: String = "") extends Layer {

  override def compile: layers.Layer = new layers.recurrent.Bidirectional(mode, layer.compile)

  override def inputShape: List[Int] = List.empty

  override def outputShape: List[Int] = List.empty

}

object Bidirectional {
  def apply(layer: Layer, mode: Mode = Mode.CONCAT): Bidirectional = new Bidirectional(layer, mode)
}
