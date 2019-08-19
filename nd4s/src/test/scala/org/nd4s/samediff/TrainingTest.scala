package org.nd4s.samediff

import org.nd4j.autodiff.samediff.{ SDVariable, SameDiff, TrainingConfig }
import org.nd4j.linalg.api.buffer.DataType
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.dataset.{ DataSet, MultiDataSet }
import org.nd4j.linalg.dataset.adapter.SingletonMultiDataSetIterator
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.learning.config.Adam
import org.nd4s.Implicits._
import org.nd4s.samediff.implicits.Implicits._
import org.scalatest.{ FlatSpec, Matchers }

class TrainingTest extends FlatSpec with Matchers {

  "SameDiff" should "allow loss calculation" in {
    for (i <- 0 until 2) {
      implicit val sd = SameDiff.create
      val ph = sd.placeHolder("ph", DataType.FLOAT, 3, 4)
      val w = sd.bind("w", Nd4j.rand(DataType.FLOAT, 4, 5))
      val b = sd.bind("b", Nd4j.rand(DataType.FLOAT, 5))
      val mmul = ph.mmul(w)
      val badd = mmul + b
      val add = badd + 1
      val shape = add.shape
      val unused1 = ph.mul(2)
      val unused2 = ph.sub(4)
      val unused3 = unused1.div(unused2)
      val loss1 = add.std("l1", true)
      val loss2 = mmul.mean("l2")
      Console.println(sd.summary)
      if (i == 0) {
        sd.setLossVariables("l1", "l2")
        sd.createGradFunction()
      } else {
        val tc = TrainingConfig.builder
          .updater(new Adam(0.01))
          .minimize("l1", "l2")
          .dataSetFeatureMapping("ph")
          .markLabelsUnused
          .build
        sd.setTrainingConfig(tc)
        val ds = new DataSet(Nd4j.create(3, 4), null)
        sd.fit(ds)
        sd.fit(ds)
      }
      for (s <- Array[String]("w", "b", badd.getVarName, add.getVarName, "l1", "l2")) {
        val gradVar = sd.getVariable(s).gradient
        assert(gradVar != null)
      }
      //Unused:
      assert(!shape.hasGradient)
      try assert(shape.gradient == null)
      catch {
        case e: IllegalStateException =>
          assert(e.getMessage.contains("only floating point variables"))
      }
      for (s <- Array[String](unused1.getVarName, unused2.getVarName, unused3.getVarName)) {
        assert(sd.getVariable(s).gradient == null)
      }
    }
  }

  "SameDiff" should "allow creating and running model with 2 losses: train on the first one, then change losses" in {
    // TODO: try to get rid of implicit here
    implicit val sd = SameDiff.create
    val ph1 = sd.placeHolder("ph1", DataType.FLOAT, 3, 4)
    val w1 = sd.bind("w1", Nd4j.rand(DataType.FLOAT, 4, 5))
    val b1 = sd.bind("b1", Nd4j.rand(DataType.FLOAT, 5))
    val mmul1 = ph1.mmul(w1)
    val badd1 = mmul1 + b1

    val ph2 = sd.placeHolder("ph2", DataType.FLOAT, 3, 2)
    val w2 = sd.bind("w2", Nd4j.rand(DataType.FLOAT, 2, 6))
    val b2 = sd.bind("b2", Nd4j.rand(DataType.FLOAT, 6))
    val mmul2 = ph2.mmul(w2)
    val badd2 = mmul2 + b2
    val loss1 = badd1.std("loss1", true)
    val loss2 = badd2.std("loss2", true)
    //First: create grad function for optimizing loss 1 only
    sd.setLossVariables("loss1")
    sd.createGradFunction()
    for (v <- Array[SDVariable](ph1, w1, b1, mmul1, badd1, loss1)) {
      assert(v.gradient != null)
    }
    for (v <- Array[SDVariable](ph2, w2, b2, mmul2, badd2, loss2)) {
      assert(v.gradient == null)
    }
    //Now, set to other loss function
    sd.setLossVariables("loss2")
    sd.createGradFunction()
    for (v <- Array[SDVariable](ph1, w1, b1, mmul1, badd1, loss1)) {
      assert(v.gradient == null)
    }
    for (v <- Array[SDVariable](ph2, w2, b2, mmul2, badd2, loss2)) {
      assert(v.gradient != null)
    }
    //Train the first side of the graph. The other side should remain unmodified!
    sd.setLossVariables("loss1")
    var w1Before = w1.getArr.dup
    var b1Before = b1.getArr.dup
    var w2Before = w2.getArr.dup
    var b2Before = b2.getArr.dup
    val tc = TrainingConfig.builder.updater(new Adam(1e-2)).dataSetFeatureMapping("ph1", "ph2").markLabelsUnused.build
    sd.setTrainingConfig(tc)
    val mds = new MultiDataSet(Array[INDArray](Nd4j.rand(DataType.FLOAT, 3, 4), Nd4j.rand(DataType.FLOAT, 3, 2)),
                               new Array[INDArray](0))
    sd.fit(new SingletonMultiDataSetIterator(mds), 3)
    assert(w1Before != w1.getArr)
    assert(b1Before != b1.getArr)
    assert(w2Before == w2.getArr)
    assert(b2Before == b2.getArr)
    //Train second side of graph; first side should be unmodified
    sd.setLossVariables("loss2")
    w1Before = w1.getArr.dup
    b1Before = b1.getArr.dup
    w2Before = w2.getArr.dup
    b2Before = b2.getArr.dup
    sd.fit(new SingletonMultiDataSetIterator(mds), 3)
    assert(w1Before == w1.getArr)
    assert(b1Before == b1.getArr)
    assert(w2Before != w2.getArr)
    assert(b2Before != b2.getArr)
  }
}
