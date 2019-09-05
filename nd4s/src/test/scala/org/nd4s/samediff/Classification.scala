package org.nd4s.samediff

import org.nd4j.autodiff.samediff.{SDIndex, SameDiff, TrainingConfig}
import org.nd4j.linalg.api.buffer.DataType
import org.nd4j.linalg.dataset.DataSet
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.learning.config.Sgd
import org.nd4s.samediff.implicits.Implicits._
import shapeless.ops.record.Updater

object Classification extends App {

  override def main(args: Array[String]): Unit = {
    val seed = 1000
    val learning_rate = 0.1
    val x1_label1 = Nd4j.rand(3, 1, seed)
    val x2_label1 = Nd4j.rand(2, 1, seed)
    val x1_label2 = Nd4j.rand(7, 1, seed)
    val x2_label2 = Nd4j.rand(6, 1, seed)

    // np.append
    val x1s = x1_label1.add(x1_label2)
    val x2s = x2_label1.add(x2_label2)

    // Must have implicit sd here for some ops
    implicit val sd = SameDiff.create
    val ys = Nd4j.scalar(0.0).mul( x1_label1.length()).add(Nd4j.scalar(1.0).mul(x1_label2.length()))

    // empty sequence
    val X1 = sd.placeHolder("x1", DataType.FLOAT, Seq.empty[Long]:_*)
    val X2 = sd.placeHolder("x2", DataType.FLOAT, Seq.empty[Long]:_*)
    val y = sd.placeHolder("y", DataType.FLOAT, Seq.empty[Long]:_*)
    // trainable == True
    val w = sd.`var`("w", DataType.FLOAT, 0,0,0)

    // tf.math.sigmoid -> sd.nn or Transform
    val tmp = w.get(SDIndex.point(1)) * (X1 + w.get(SDIndex.point(0)))
    val tmp2 = sd.math.neg(w.get(SDIndex.point(2)) * (X2 + tmp))
    val y_model = sd.nn.sigmoid(tmp2)

    // who is target for reduce_mean
    val cost_fun = sd.math.neg(sd.math.log(y_model)) * y - sd.math.log(1-y_model) * (1-y)
    val cost = cost_fun.mean()

    val updater = new Sgd(learning_rate)
    val conf = new TrainingConfig.Builder().updater(updater).build

    sd.setTrainingConfig(conf)
    sd.fit()
  }
}
