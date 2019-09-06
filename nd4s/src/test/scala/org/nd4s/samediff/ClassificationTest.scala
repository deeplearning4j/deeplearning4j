package org.nd4s.samediff

import org.nd4j.autodiff.samediff.{SDIndex, SameDiff, TrainingConfig}
import org.nd4j.linalg.api.buffer.DataType
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.dataset.{DataSet, MultiDataSet}
import org.nd4j.linalg.dataset.adapter.SingletonMultiDataSetIterator
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.learning.config.Sgd
import org.nd4s.samediff.implicits.Implicits._
import org.scalatest.FlatSpec
import shapeless.ops.record.Updater

object ClassificationTest extends FlatSpec {

  "classification example" should "work" in {
    Nd4j.setDefaultDataTypes(DataType.DOUBLE, DataType.DOUBLE)
    val seed = 1000
    val learning_rate = 0.1
    val x1_label1 = Nd4j.rand(3, 1)
    val x2_label1 = Nd4j.rand(2, 1)
    val x1_label2 = Nd4j.rand(7, 1)
    val x2_label2 = Nd4j.rand(6, 1)

    // np.append, was not able to guess proper method
    val x1s = Nd4j.concat(0, x1_label1, x1_label2)
    val x2s = Nd4j.concat(0, x2_label1, x2_label2)

    // Must have implicit sd here for some ops, inobvious
    implicit val sd = SameDiff.create
    val ys = Nd4j.scalar(0.0).mul(x1_label1.length()).add(Nd4j.scalar(1.0).mul(x1_label2.length()))

    // empty sequence
    val X1 = sd.placeHolder("x1", DataType.FLOAT, 10, 1)
    val X2 = sd.placeHolder("x2", DataType.FLOAT, 8, 1)
    val y = sd.placeHolder("y", DataType.FLOAT)
    // There's no option to pass Trainable=True
    val w = sd.`var`("w", DataType.FLOAT, 0, 0, 0)

    // tf.math.sigmoid -> where can I get sigmoid ? sd.nn : Transform
    val tmp = w.get(SDIndex.point(1)) * (X1 + w.get(SDIndex.point(0)))
    val tmp2 = sd.math.neg(w.get(SDIndex.point(2)) * (X2 + tmp))
    val y_model = sd.nn.sigmoid(tmp2)

    // what is target for reduce_mean? What is proper replacement for np.reduce_mean?
    // 1 - SDVariable - doesn't work as is
    // lost in long formula!!!
    // java.lang.IllegalStateException: Only floating point types are supported for strict tranform ops - got INT - log
    val cost_fun = -sd.math.log(y_model) * y - sd.math.log(sd.constant(1.0) - y_model) * (sd.constant(1.0) - y)
    val loss = cost_fun.mean("loss")

    val updater = new Sgd(learning_rate)
    // mapping between values and ph

    sd.setLossVariables("loss")
    val conf = new TrainingConfig.Builder()
      .updater(updater)
      //.minimize("loss")
      .dataSetFeatureMapping("x1", "x2", "y")
      .markLabelsUnused()
      .build()

    val mds = new MultiDataSet(Array[INDArray](x1s, x2s, ys), new Array[INDArray](0))

    sd.setTrainingConfig(conf)
    sd.fit(new SingletonMultiDataSetIterator(mds), 1)

    import org.nd4j.linalg.api.ndarray.INDArray
    Console.print(sd.outputs())
  }
}
