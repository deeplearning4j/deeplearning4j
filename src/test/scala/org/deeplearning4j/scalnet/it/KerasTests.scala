package org.deeplearning4j.scalnet.it

import org.deeplearning4j.scalnet.examples.keras.feedforward.MLPMnistTwoLayerExample
import org.deeplearning4j.scalnet.examples.keras.convolution.LeNetMnistExample
import org.scalatest.FlatSpec
import scala.PartialFunction._

class KerasTests extends FlatSpec {
  val stream = new java.io.ByteArrayOutputStream()
  val nullStream = new java.io.OutputStream() { override def write(b: Int) = { } }

  def statRegex(label:String) = s""".*?($label): *([0-9.]+)""".r
  def statPrinter(lbl: String): PartialFunction[String, (String, Double)] = {
    val rx = statRegex(lbl)
    s => s match { case rx(l, n) => (l, n.toDouble) }
  }
  def statsPrinter(lbls: String*): PartialFunction[String, (String, Double)] = {
    lbls.toList.map(statPrinter).reduce(_ orElse _)
  }

  "MLPMnistTwoLayerExample" should "reach 0.9814 accuracy" in {
    var s: Array[String] = Array()
    Console.withOut(stream) {
      Console.withErr(nullStream) {
        MLPMnistTwoLayerExample.main(Array())
        s = stream.toString.split("\n").takeRight(6)
      }
    }
    s.collect(statsPrinter("F1 Score", "Accuracy", "Precision", "Recall")).foreach{
      case (l, n) => assert(n > 0.9813, s"$l is not large enough! found: $n")
    }
  }

  "LeNetMnistExample" should "reach 0.9905 accuracy" in {
    var s: Array[String] = Array()
    Console.withOut(stream) {
      Console.withErr(nullStream) {
        LeNetMnistExample.main(Array())
        s = stream.toString.split("\n").takeRight(6)
      }
    }
    s.collect(statsPrinter("F1 Score", "Accuracy", "Precision", "Recall")).foreach{
      case (l, n) => assert(n > 0.9905, s"$l is not large enough! found: $n")
    }
  }
}
