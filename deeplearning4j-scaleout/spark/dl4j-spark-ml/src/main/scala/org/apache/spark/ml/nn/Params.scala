package org.apache.spark.ml.nn

import org.apache.spark.annotation.DeveloperApi
import org.apache.spark.ml.param._

import org.deeplearning4j.nn.conf.MultiLayerConfiguration;

@DeveloperApi
private[spark] trait HasMultiLayerConfiguration extends Params {

  /**
   * Param for multiLayer configuration.
   * @group param
   */
  val confParam = new Param[MultiLayerConfiguration](this, "conf", "internal configuration", 
      Some(new MultiLayerConfiguration()))

  /** @group getParam */
  def getConf: MultiLayerConfiguration = get(confParam)
}

/**
 * The reconstruction column is similar to 'prediction' but its type is not likely Double
 */
@DeveloperApi
private[spark] trait HasReconstructionCol extends Params {

  /**
   * Param for reconstruction column name.
   * @group param
   */
  val reconstructionCol: Param[String] = 
    new Param[String](this, "reconstructionCol", "reconstruction column name", Some("reconstruction"))

  /** @group getParam */
  def getProbabilityCol: String = get(reconstructionCol)
}

@DeveloperApi
private[spark] trait HasBatchSize extends Params {

  /**
   * Param for batch size
   * @group param
   */
  val batchSizeParam: IntParam = new IntParam(this, "batchSize", "batch size", Some(10))
  
  /** @group getParam */
  def getBatchSize: Int = get(batchSizeParam)
}

@DeveloperApi
private[spark] trait HasWindowSize extends Params {

  /**
   * Param for window size
   * @group param
   */
  val windowSizeParam: IntParam = new IntParam(this, "windowSize", "window size", Some(1))
  
  /** @group getParam */
  def getWindowSize: Int = get(windowSizeParam)
}

