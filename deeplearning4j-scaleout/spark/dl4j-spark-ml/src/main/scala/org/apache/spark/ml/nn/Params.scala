/*
 * Copyright 2015 Skymind,Inc.
 *
 *    Licensed under the Apache License, Version 2.0 (the "License");
 *    you may not use this file except in compliance with the License.
 *    You may obtain a copy of the License at
 *
 *        http://www.apache.org/licenses/LICENSE-2.0
 *
 *    Unless required by applicable law or agreed to in writing, software
 *    distributed under the License is distributed on an "AS IS" BASIS,
 *    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *    See the License for the specific language governing permissions and
 *    limitations under the License.
 */

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
  val conf = new Param[String](this, "conf", "multilayer configuration")

  /** @group getParam */
  def getConf: String = $(conf)
}

/**
 * The reconstruction column is similar to 'prediction' but is typically a Vector, not Double
 */
@DeveloperApi
private[spark] trait HasReconstructionCol extends Params {

  /**
   * Param for reconstruction column name.
   * @group param
   */
  val reconstructionCol: Param[String] = 
    new Param[String](this, "reconstructionCol", "reconstruction column name")

  /** @group getParam */
  def getReconstructionCol: String = $(reconstructionCol)
}

@DeveloperApi
private[spark] trait HasBatchSize extends Params {

  /**
   * Param for batch size
   * @group param
   */
  val batchSize: IntParam = new IntParam(this, "batchSize", "batch size")

  /** @group getParam */
  def getBatchSize: Int = $(batchSize)
}

@DeveloperApi
private[spark] trait HasEpochs extends Params {

  /**
   * Param for epochs
   * @group param
   */
  val epochs: IntParam = new IntParam(this, "epochs", "number of epochs")

  /** @group getParam */
  def getEpochs: Int = $(epochs)
}

@DeveloperApi
private[spark] trait HasLayerIndex extends Params {

  /**
   * Param for layer index
   * @group param
   */
  val layerIndex: IntParam = new IntParam(this, "layerIndex", "layer index (one-based)")

  /** @group getParam */
  def getLayerIndex: Int = $(layerIndex)
}

