/*******************************************************************************
 * Copyright (c) 2019 Konduit K.K.
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

//================== GENERATED CODE - DO NOT MODIFY THIS FILE ==================

package org.nd4j.linalg.factory.ops;

import static org.nd4j.linalg.factory.NDValidation.isSameType;

import org.nd4j.autodiff.loss.LossReduce;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.NDValidation;
import org.nd4j.linalg.factory.Nd4j;

public class NDLoss {
  public NDLoss() {
  }

  /**
   * Absolute difference loss: {@code sum_i abs( label[i] - predictions[i] )<br>
   *
   * @param label Label array (NUMERIC type)
   * @param predictions Predictions array (NUMERIC type)
   * @param weights Weights array. May be null. If null, a weight of 1.0 is used (NUMERIC type)
   * @param lossReduce Reduction type for the loss. See {@link LossReduce} for more details. Default: {@link LossReduce#MEAN_BY_NONZERO_WEIGHT_COUNT}
   * @return output loss variable (NUMERIC type)
   */
  public INDArray absoluteDifference(INDArray label, INDArray predictions, INDArray weights,
      LossReduce lossReduce) {
    NDValidation.validateNumerical("absoluteDifference", "label", label);
    NDValidation.validateNumerical("absoluteDifference", "predictions", predictions);
    NDValidation.validateNumerical("absoluteDifference", "weights", weights);
    return Nd4j.exec(new org.nd4j.linalg.api.ops.impl.loss.AbsoluteDifferenceLoss(label, predictions, weights, lossReduce))[0];
  }

  /**
   * Cosine distance loss: {@code 1 - cosineSimilarity(x,y)} or {@code 1 - sum_i label[i] * prediction[i]}, which is<br>
   * equivalent to cosine distance when both the predictions and labels are normalized.<br>
   * <b>Note</b>: This loss function assumes that both the predictions and labels are normalized to have unit l2 norm.<br>
   * If this is not the case, you should normalize them first by dividing by norm2(String, SDVariable, boolean, int...)<br>
   * along the cosine distance dimension (with keepDims=true).<br>
   *
   * @param label Label array (NUMERIC type)
   * @param predictions Predictions array (NUMERIC type)
   * @param weights Weights array. May be null. If null, a weight of 1.0 is use (NUMERIC type)
   * @param lossReduce Reduction type for the loss. See {@link LossReduce} for more details. Default: {@link LossReduce#MEAN_BY_NONZERO_WEIGHT_COUNT}
   * @param dimension Dimension to perform the cosine distance over
   * @return output Cosine distance loss  (NUMERIC type)
   */
  public INDArray cosineDistance(INDArray label, INDArray predictions, INDArray weights,
      LossReduce lossReduce, int dimension) {
    NDValidation.validateNumerical("cosineDistance", "label", label);
    NDValidation.validateNumerical("cosineDistance", "predictions", predictions);
    NDValidation.validateNumerical("cosineDistance", "weights", weights);
    return Nd4j.exec(new org.nd4j.linalg.api.ops.impl.loss.CosineDistanceLoss(label, predictions, weights, lossReduce, dimension))[0];
  }

  /**
   * Hinge loss: a loss function used for training classifiers.<br>
   * Implements {@code L = max(0, 1 - t * predictions)} where t is the label values after internally converting to {-1,1}<br>
   * from the user specified {0,1}. Note that Labels should be provided with values {0,1}.<br>
   *
   * @param label Label array. Each value should be 0.0 or 1.0 (internally -1 to 1 is used) (NUMERIC type)
   * @param predictions Predictions array (NUMERIC type)
   * @param weights Weights array. May be null. If null, a weight of 1.0 is used (NUMERIC type)
   * @param lossReduce Reduction type for the loss. See {@link LossReduce} for more details. Default: {@link LossReduce#MEAN_BY_NONZERO_WEIGHT_COUNT}
   * @return output Loss variable (NUMERIC type)
   */
  public INDArray hingeLoss(INDArray label, INDArray predictions, INDArray weights,
      LossReduce lossReduce) {
    NDValidation.validateNumerical("hingeLoss", "label", label);
    NDValidation.validateNumerical("hingeLoss", "predictions", predictions);
    NDValidation.validateNumerical("hingeLoss", "weights", weights);
    return Nd4j.exec(new org.nd4j.linalg.api.ops.impl.loss.HingeLoss(label, predictions, weights, lossReduce))[0];
  }

  /**
   * Huber loss function, used for robust regression. It is similar both squared error loss and absolute difference loss,<br>
   * though is less sensitive to outliers than squared error.<br>
   * Huber loss implements:<br>
   * <pre><br>
   * {@code L = 0.5 * (label[i] - predictions[i])^2 if abs(label[i] - predictions[i]) < delta<br>
   * L = delta * abs(label[i] - predictions[i]) - 0.5 * delta^2 otherwise<br>
   * }<br>
   * </pre><br>
   *
   * @param label Label array (NUMERIC type)
   * @param predictions Predictions array (NUMERIC type)
   * @param weights Weights array. May be null. If null, a weight of 1.0 is used (NUMERIC type)
   * @param lossReduce Reduction type for the loss. See {@link LossReduce} for more details. Default: {@link LossReduce#MEAN_BY_NONZERO_WEIGHT_COUNT}
   * @param delta Loss function delta value
   * @return output Huber loss (NUMERIC type)
   */
  public INDArray huberLoss(INDArray label, INDArray predictions, INDArray weights,
      LossReduce lossReduce, double delta) {
    NDValidation.validateNumerical("huberLoss", "label", label);
    NDValidation.validateNumerical("huberLoss", "predictions", predictions);
    NDValidation.validateNumerical("huberLoss", "weights", weights);
    return Nd4j.exec(new org.nd4j.linalg.api.ops.impl.loss.HuberLoss(label, predictions, weights, lossReduce, delta))[0];
  }

  /**
   * L2 loss: 1/2 * sum(x^2)<br>
   *
   * @param var Variable to calculate L2 loss of (NUMERIC type)
   * @return output L2 loss (NUMERIC type)
   */
  public INDArray l2Loss(INDArray var) {
    NDValidation.validateNumerical("l2Loss", "var", var);
    return Nd4j.exec(new org.nd4j.linalg.api.ops.impl.loss.L2Loss(var))[0];
  }
}
