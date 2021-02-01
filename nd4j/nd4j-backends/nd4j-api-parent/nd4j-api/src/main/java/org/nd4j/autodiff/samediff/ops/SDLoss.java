/*
 *  ******************************************************************************
 *  *
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
 *  *  See the NOTICE file distributed with this work for additional
 *  *  information regarding copyright ownership.
 *  * Unless required by applicable law or agreed to in writing, software
 *  * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 *  * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 *  * License for the specific language governing permissions and limitations
 *  * under the License.
 *  *
 *  * SPDX-License-Identifier: Apache-2.0
 *  *****************************************************************************
 */

//================== GENERATED CODE - DO NOT MODIFY THIS FILE ==================

package org.nd4j.autodiff.samediff.ops;

import java.lang.String;
import org.nd4j.autodiff.loss.LossReduce;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;

public class SDLoss extends SDOps {
  public SDLoss(SameDiff sameDiff) {
    super(sameDiff);
  }

  /**
   * Absolute difference loss: {@code sum_i abs( label[i] - predictions[i] )}<br>
   *
   * @param label Label array (NUMERIC type)
   * @param predictions Predictions array (NUMERIC type)
   * @param weights Weights array. May be null. If null, a weight of 1.0 is used (NUMERIC type)
   * @param lossReduce Reduction type for the loss. See {@link LossReduce} for more details. Default: {@link LossReduce#MEAN_BY_NONZERO_WEIGHT_COUNT}
   * @return output loss variable (NUMERIC type)
   */
  public SDVariable absoluteDifference(SDVariable label, SDVariable predictions, SDVariable weights,
      LossReduce lossReduce) {
    SDValidation.validateNumerical("absoluteDifference", "label", label);
    SDValidation.validateNumerical("absoluteDifference", "predictions", predictions);
    SDValidation.validateNumerical("absoluteDifference", "weights", weights);
    SDVariable out = new org.nd4j.linalg.api.ops.impl.loss.AbsoluteDifferenceLoss(sd,label, predictions, weights, lossReduce).outputVariable();
    out.markAsLoss();
    return out;
  }

  /**
   * Absolute difference loss: {@code sum_i abs( label[i] - predictions[i] )}<br>
   *
   * @param name name May be null. Name for the output variable
   * @param label Label array (NUMERIC type)
   * @param predictions Predictions array (NUMERIC type)
   * @param weights Weights array. May be null. If null, a weight of 1.0 is used (NUMERIC type)
   * @param lossReduce Reduction type for the loss. See {@link LossReduce} for more details. Default: {@link LossReduce#MEAN_BY_NONZERO_WEIGHT_COUNT}
   * @return output loss variable (NUMERIC type)
   */
  public SDVariable absoluteDifference(String name, SDVariable label, SDVariable predictions,
      SDVariable weights, LossReduce lossReduce) {
    SDValidation.validateNumerical("absoluteDifference", "label", label);
    SDValidation.validateNumerical("absoluteDifference", "predictions", predictions);
    SDValidation.validateNumerical("absoluteDifference", "weights", weights);
    SDVariable out =  new org.nd4j.linalg.api.ops.impl.loss.AbsoluteDifferenceLoss(sd,label, predictions, weights, lossReduce).outputVariable();
    out.markAsLoss();
    return sd.updateVariableNameAndReference(out, name);
  }

  /**
   * Absolute difference loss: {@code sum_i abs( label[i] - predictions[i] )}<br>
   *
   * @param label Label array (NUMERIC type)
   * @param predictions Predictions array (NUMERIC type)
   * @param weights Weights array. May be null. If null, a weight of 1.0 is used (NUMERIC type)
   * @return output loss variable (NUMERIC type)
   */
  public SDVariable absoluteDifference(SDVariable label, SDVariable predictions,
      SDVariable weights) {
    SDValidation.validateNumerical("absoluteDifference", "label", label);
    SDValidation.validateNumerical("absoluteDifference", "predictions", predictions);
    SDValidation.validateNumerical("absoluteDifference", "weights", weights);
    SDVariable out = new org.nd4j.linalg.api.ops.impl.loss.AbsoluteDifferenceLoss(sd,label, predictions, weights, org.nd4j.autodiff.loss.LossReduce.MEAN_BY_NONZERO_WEIGHT_COUNT).outputVariable();
    out.markAsLoss();
    return out;
  }

  /**
   * Absolute difference loss: {@code sum_i abs( label[i] - predictions[i] )}<br>
   *
   * @param name name May be null. Name for the output variable
   * @param label Label array (NUMERIC type)
   * @param predictions Predictions array (NUMERIC type)
   * @param weights Weights array. May be null. If null, a weight of 1.0 is used (NUMERIC type)
   * @return output loss variable (NUMERIC type)
   */
  public SDVariable absoluteDifference(String name, SDVariable label, SDVariable predictions,
      SDVariable weights) {
    SDValidation.validateNumerical("absoluteDifference", "label", label);
    SDValidation.validateNumerical("absoluteDifference", "predictions", predictions);
    SDValidation.validateNumerical("absoluteDifference", "weights", weights);
    SDVariable out =  new org.nd4j.linalg.api.ops.impl.loss.AbsoluteDifferenceLoss(sd,label, predictions, weights, org.nd4j.autodiff.loss.LossReduce.MEAN_BY_NONZERO_WEIGHT_COUNT).outputVariable();
    out.markAsLoss();
    return sd.updateVariableNameAndReference(out, name);
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
  public SDVariable cosineDistance(SDVariable label, SDVariable predictions, SDVariable weights,
      LossReduce lossReduce, int dimension) {
    SDValidation.validateNumerical("cosineDistance", "label", label);
    SDValidation.validateNumerical("cosineDistance", "predictions", predictions);
    SDValidation.validateNumerical("cosineDistance", "weights", weights);
    SDVariable out = new org.nd4j.linalg.api.ops.impl.loss.CosineDistanceLoss(sd,label, predictions, weights, lossReduce, dimension).outputVariable();
    out.markAsLoss();
    return out;
  }

  /**
   * Cosine distance loss: {@code 1 - cosineSimilarity(x,y)} or {@code 1 - sum_i label[i] * prediction[i]}, which is<br>
   * equivalent to cosine distance when both the predictions and labels are normalized.<br>
   * <b>Note</b>: This loss function assumes that both the predictions and labels are normalized to have unit l2 norm.<br>
   * If this is not the case, you should normalize them first by dividing by norm2(String, SDVariable, boolean, int...)<br>
   * along the cosine distance dimension (with keepDims=true).<br>
   *
   * @param name name May be null. Name for the output variable
   * @param label Label array (NUMERIC type)
   * @param predictions Predictions array (NUMERIC type)
   * @param weights Weights array. May be null. If null, a weight of 1.0 is use (NUMERIC type)
   * @param lossReduce Reduction type for the loss. See {@link LossReduce} for more details. Default: {@link LossReduce#MEAN_BY_NONZERO_WEIGHT_COUNT}
   * @param dimension Dimension to perform the cosine distance over
   * @return output Cosine distance loss  (NUMERIC type)
   */
  public SDVariable cosineDistance(String name, SDVariable label, SDVariable predictions,
      SDVariable weights, LossReduce lossReduce, int dimension) {
    SDValidation.validateNumerical("cosineDistance", "label", label);
    SDValidation.validateNumerical("cosineDistance", "predictions", predictions);
    SDValidation.validateNumerical("cosineDistance", "weights", weights);
    SDVariable out =  new org.nd4j.linalg.api.ops.impl.loss.CosineDistanceLoss(sd,label, predictions, weights, lossReduce, dimension).outputVariable();
    out.markAsLoss();
    return sd.updateVariableNameAndReference(out, name);
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
   * @param dimension Dimension to perform the cosine distance over
   * @return output Cosine distance loss  (NUMERIC type)
   */
  public SDVariable cosineDistance(SDVariable label, SDVariable predictions, SDVariable weights,
      int dimension) {
    SDValidation.validateNumerical("cosineDistance", "label", label);
    SDValidation.validateNumerical("cosineDistance", "predictions", predictions);
    SDValidation.validateNumerical("cosineDistance", "weights", weights);
    SDVariable out = new org.nd4j.linalg.api.ops.impl.loss.CosineDistanceLoss(sd,label, predictions, weights, org.nd4j.autodiff.loss.LossReduce.MEAN_BY_NONZERO_WEIGHT_COUNT, dimension).outputVariable();
    out.markAsLoss();
    return out;
  }

  /**
   * Cosine distance loss: {@code 1 - cosineSimilarity(x,y)} or {@code 1 - sum_i label[i] * prediction[i]}, which is<br>
   * equivalent to cosine distance when both the predictions and labels are normalized.<br>
   * <b>Note</b>: This loss function assumes that both the predictions and labels are normalized to have unit l2 norm.<br>
   * If this is not the case, you should normalize them first by dividing by norm2(String, SDVariable, boolean, int...)<br>
   * along the cosine distance dimension (with keepDims=true).<br>
   *
   * @param name name May be null. Name for the output variable
   * @param label Label array (NUMERIC type)
   * @param predictions Predictions array (NUMERIC type)
   * @param weights Weights array. May be null. If null, a weight of 1.0 is use (NUMERIC type)
   * @param dimension Dimension to perform the cosine distance over
   * @return output Cosine distance loss  (NUMERIC type)
   */
  public SDVariable cosineDistance(String name, SDVariable label, SDVariable predictions,
      SDVariable weights, int dimension) {
    SDValidation.validateNumerical("cosineDistance", "label", label);
    SDValidation.validateNumerical("cosineDistance", "predictions", predictions);
    SDValidation.validateNumerical("cosineDistance", "weights", weights);
    SDVariable out =  new org.nd4j.linalg.api.ops.impl.loss.CosineDistanceLoss(sd,label, predictions, weights, org.nd4j.autodiff.loss.LossReduce.MEAN_BY_NONZERO_WEIGHT_COUNT, dimension).outputVariable();
    out.markAsLoss();
    return sd.updateVariableNameAndReference(out, name);
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
  public SDVariable hingeLoss(SDVariable label, SDVariable predictions, SDVariable weights,
      LossReduce lossReduce) {
    SDValidation.validateNumerical("hingeLoss", "label", label);
    SDValidation.validateNumerical("hingeLoss", "predictions", predictions);
    SDValidation.validateNumerical("hingeLoss", "weights", weights);
    SDVariable out = new org.nd4j.linalg.api.ops.impl.loss.HingeLoss(sd,label, predictions, weights, lossReduce).outputVariable();
    out.markAsLoss();
    return out;
  }

  /**
   * Hinge loss: a loss function used for training classifiers.<br>
   * Implements {@code L = max(0, 1 - t * predictions)} where t is the label values after internally converting to {-1,1}<br>
   * from the user specified {0,1}. Note that Labels should be provided with values {0,1}.<br>
   *
   * @param name name May be null. Name for the output variable
   * @param label Label array. Each value should be 0.0 or 1.0 (internally -1 to 1 is used) (NUMERIC type)
   * @param predictions Predictions array (NUMERIC type)
   * @param weights Weights array. May be null. If null, a weight of 1.0 is used (NUMERIC type)
   * @param lossReduce Reduction type for the loss. See {@link LossReduce} for more details. Default: {@link LossReduce#MEAN_BY_NONZERO_WEIGHT_COUNT}
   * @return output Loss variable (NUMERIC type)
   */
  public SDVariable hingeLoss(String name, SDVariable label, SDVariable predictions,
      SDVariable weights, LossReduce lossReduce) {
    SDValidation.validateNumerical("hingeLoss", "label", label);
    SDValidation.validateNumerical("hingeLoss", "predictions", predictions);
    SDValidation.validateNumerical("hingeLoss", "weights", weights);
    SDVariable out =  new org.nd4j.linalg.api.ops.impl.loss.HingeLoss(sd,label, predictions, weights, lossReduce).outputVariable();
    out.markAsLoss();
    return sd.updateVariableNameAndReference(out, name);
  }

  /**
   * Hinge loss: a loss function used for training classifiers.<br>
   * Implements {@code L = max(0, 1 - t * predictions)} where t is the label values after internally converting to {-1,1}<br>
   * from the user specified {0,1}. Note that Labels should be provided with values {0,1}.<br>
   *
   * @param label Label array. Each value should be 0.0 or 1.0 (internally -1 to 1 is used) (NUMERIC type)
   * @param predictions Predictions array (NUMERIC type)
   * @param weights Weights array. May be null. If null, a weight of 1.0 is used (NUMERIC type)
   * @return output Loss variable (NUMERIC type)
   */
  public SDVariable hingeLoss(SDVariable label, SDVariable predictions, SDVariable weights) {
    SDValidation.validateNumerical("hingeLoss", "label", label);
    SDValidation.validateNumerical("hingeLoss", "predictions", predictions);
    SDValidation.validateNumerical("hingeLoss", "weights", weights);
    SDVariable out = new org.nd4j.linalg.api.ops.impl.loss.HingeLoss(sd,label, predictions, weights, org.nd4j.autodiff.loss.LossReduce.MEAN_BY_NONZERO_WEIGHT_COUNT).outputVariable();
    out.markAsLoss();
    return out;
  }

  /**
   * Hinge loss: a loss function used for training classifiers.<br>
   * Implements {@code L = max(0, 1 - t * predictions)} where t is the label values after internally converting to {-1,1}<br>
   * from the user specified {0,1}. Note that Labels should be provided with values {0,1}.<br>
   *
   * @param name name May be null. Name for the output variable
   * @param label Label array. Each value should be 0.0 or 1.0 (internally -1 to 1 is used) (NUMERIC type)
   * @param predictions Predictions array (NUMERIC type)
   * @param weights Weights array. May be null. If null, a weight of 1.0 is used (NUMERIC type)
   * @return output Loss variable (NUMERIC type)
   */
  public SDVariable hingeLoss(String name, SDVariable label, SDVariable predictions,
      SDVariable weights) {
    SDValidation.validateNumerical("hingeLoss", "label", label);
    SDValidation.validateNumerical("hingeLoss", "predictions", predictions);
    SDValidation.validateNumerical("hingeLoss", "weights", weights);
    SDVariable out =  new org.nd4j.linalg.api.ops.impl.loss.HingeLoss(sd,label, predictions, weights, org.nd4j.autodiff.loss.LossReduce.MEAN_BY_NONZERO_WEIGHT_COUNT).outputVariable();
    out.markAsLoss();
    return sd.updateVariableNameAndReference(out, name);
  }

  /**
   * Huber loss function, used for robust regression. It is similar both squared error loss and absolute difference loss,<br>
   * though is less sensitive to outliers than squared error.<br>
   * Huber loss implements:<br>
   * <pre><br>
   * {@code L = 0.5 * (label[i] - predictions[i])^2 if abs(label[i] - predictions[i]) < delta}<br>
   * {@code L = delta * abs(label[i] - predictions[i]) - 0.5 * delta^2 otherwise}<br>
   * </pre><br>
   *
   * @param label Label array (NUMERIC type)
   * @param predictions Predictions array (NUMERIC type)
   * @param weights Weights array. May be null. If null, a weight of 1.0 is used (NUMERIC type)
   * @param lossReduce Reduction type for the loss. See {@link LossReduce} for more details. Default: {@link LossReduce#MEAN_BY_NONZERO_WEIGHT_COUNT}
   * @param delta Loss function delta value
   * @return output Huber loss (NUMERIC type)
   */
  public SDVariable huberLoss(SDVariable label, SDVariable predictions, SDVariable weights,
      LossReduce lossReduce, double delta) {
    SDValidation.validateNumerical("huberLoss", "label", label);
    SDValidation.validateNumerical("huberLoss", "predictions", predictions);
    SDValidation.validateNumerical("huberLoss", "weights", weights);
    SDVariable out = new org.nd4j.linalg.api.ops.impl.loss.HuberLoss(sd,label, predictions, weights, lossReduce, delta).outputVariable();
    out.markAsLoss();
    return out;
  }

  /**
   * Huber loss function, used for robust regression. It is similar both squared error loss and absolute difference loss,<br>
   * though is less sensitive to outliers than squared error.<br>
   * Huber loss implements:<br>
   * <pre><br>
   * {@code L = 0.5 * (label[i] - predictions[i])^2 if abs(label[i] - predictions[i]) < delta}<br>
   * {@code L = delta * abs(label[i] - predictions[i]) - 0.5 * delta^2 otherwise}<br>
   * </pre><br>
   *
   * @param name name May be null. Name for the output variable
   * @param label Label array (NUMERIC type)
   * @param predictions Predictions array (NUMERIC type)
   * @param weights Weights array. May be null. If null, a weight of 1.0 is used (NUMERIC type)
   * @param lossReduce Reduction type for the loss. See {@link LossReduce} for more details. Default: {@link LossReduce#MEAN_BY_NONZERO_WEIGHT_COUNT}
   * @param delta Loss function delta value
   * @return output Huber loss (NUMERIC type)
   */
  public SDVariable huberLoss(String name, SDVariable label, SDVariable predictions,
      SDVariable weights, LossReduce lossReduce, double delta) {
    SDValidation.validateNumerical("huberLoss", "label", label);
    SDValidation.validateNumerical("huberLoss", "predictions", predictions);
    SDValidation.validateNumerical("huberLoss", "weights", weights);
    SDVariable out =  new org.nd4j.linalg.api.ops.impl.loss.HuberLoss(sd,label, predictions, weights, lossReduce, delta).outputVariable();
    out.markAsLoss();
    return sd.updateVariableNameAndReference(out, name);
  }

  /**
   * Huber loss function, used for robust regression. It is similar both squared error loss and absolute difference loss,<br>
   * though is less sensitive to outliers than squared error.<br>
   * Huber loss implements:<br>
   * <pre><br>
   * {@code L = 0.5 * (label[i] - predictions[i])^2 if abs(label[i] - predictions[i]) < delta}<br>
   * {@code L = delta * abs(label[i] - predictions[i]) - 0.5 * delta^2 otherwise}<br>
   * </pre><br>
   *
   * @param label Label array (NUMERIC type)
   * @param predictions Predictions array (NUMERIC type)
   * @param weights Weights array. May be null. If null, a weight of 1.0 is used (NUMERIC type)
   * @param delta Loss function delta value
   * @return output Huber loss (NUMERIC type)
   */
  public SDVariable huberLoss(SDVariable label, SDVariable predictions, SDVariable weights,
      double delta) {
    SDValidation.validateNumerical("huberLoss", "label", label);
    SDValidation.validateNumerical("huberLoss", "predictions", predictions);
    SDValidation.validateNumerical("huberLoss", "weights", weights);
    SDVariable out = new org.nd4j.linalg.api.ops.impl.loss.HuberLoss(sd,label, predictions, weights, org.nd4j.autodiff.loss.LossReduce.MEAN_BY_NONZERO_WEIGHT_COUNT, delta).outputVariable();
    out.markAsLoss();
    return out;
  }

  /**
   * Huber loss function, used for robust regression. It is similar both squared error loss and absolute difference loss,<br>
   * though is less sensitive to outliers than squared error.<br>
   * Huber loss implements:<br>
   * <pre><br>
   * {@code L = 0.5 * (label[i] - predictions[i])^2 if abs(label[i] - predictions[i]) < delta}<br>
   * {@code L = delta * abs(label[i] - predictions[i]) - 0.5 * delta^2 otherwise}<br>
   * </pre><br>
   *
   * @param name name May be null. Name for the output variable
   * @param label Label array (NUMERIC type)
   * @param predictions Predictions array (NUMERIC type)
   * @param weights Weights array. May be null. If null, a weight of 1.0 is used (NUMERIC type)
   * @param delta Loss function delta value
   * @return output Huber loss (NUMERIC type)
   */
  public SDVariable huberLoss(String name, SDVariable label, SDVariable predictions,
      SDVariable weights, double delta) {
    SDValidation.validateNumerical("huberLoss", "label", label);
    SDValidation.validateNumerical("huberLoss", "predictions", predictions);
    SDValidation.validateNumerical("huberLoss", "weights", weights);
    SDVariable out =  new org.nd4j.linalg.api.ops.impl.loss.HuberLoss(sd,label, predictions, weights, org.nd4j.autodiff.loss.LossReduce.MEAN_BY_NONZERO_WEIGHT_COUNT, delta).outputVariable();
    out.markAsLoss();
    return sd.updateVariableNameAndReference(out, name);
  }

  /**
   * L2 loss: 1/2 * sum(x^2)<br>
   *
   * @param var Variable to calculate L2 loss of (NUMERIC type)
   * @return output L2 loss (NUMERIC type)
   */
  public SDVariable l2Loss(SDVariable var) {
    SDValidation.validateNumerical("l2Loss", "var", var);
    SDVariable out = new org.nd4j.linalg.api.ops.impl.loss.L2Loss(sd,var).outputVariable();
    out.markAsLoss();
    return out;
  }

  /**
   * L2 loss: 1/2 * sum(x^2)<br>
   *
   * @param name name May be null. Name for the output variable
   * @param var Variable to calculate L2 loss of (NUMERIC type)
   * @return output L2 loss (NUMERIC type)
   */
  public SDVariable l2Loss(String name, SDVariable var) {
    SDValidation.validateNumerical("l2Loss", "var", var);
    SDVariable out =  new org.nd4j.linalg.api.ops.impl.loss.L2Loss(sd,var).outputVariable();
    out.markAsLoss();
    return sd.updateVariableNameAndReference(out, name);
  }

  /**
   * Log loss, i.e., binary cross entropy loss, usually used for binary multi-label classification. Implements:<br>
   * {@code -1/numExamples * sum_i (labels[i] * log(predictions[i] + epsilon) + (1-labels[i]) * log(1-predictions[i] + epsilon))}<br>
   *
   * @param label Label array (NUMERIC type)
   * @param predictions Predictions array (NUMERIC type)
   * @param weights Weights array. May be null. If null, a weight of 1.0 is used (NUMERIC type)
   * @param lossReduce Reduction type for the loss. See {@link LossReduce} for more details. Default: {@link LossReduce#MEAN_BY_NONZERO_WEIGHT_COUNT}
   * @param epsilon epsilon
   * @return output Log loss  (NUMERIC type)
   */
  public SDVariable logLoss(SDVariable label, SDVariable predictions, SDVariable weights,
      LossReduce lossReduce, double epsilon) {
    SDValidation.validateNumerical("logLoss", "label", label);
    SDValidation.validateNumerical("logLoss", "predictions", predictions);
    SDValidation.validateNumerical("logLoss", "weights", weights);
    SDVariable out = new org.nd4j.linalg.api.ops.impl.loss.LogLoss(sd,label, predictions, weights, lossReduce, epsilon).outputVariable();
    out.markAsLoss();
    return out;
  }

  /**
   * Log loss, i.e., binary cross entropy loss, usually used for binary multi-label classification. Implements:<br>
   * {@code -1/numExamples * sum_i (labels[i] * log(predictions[i] + epsilon) + (1-labels[i]) * log(1-predictions[i] + epsilon))}<br>
   *
   * @param name name May be null. Name for the output variable
   * @param label Label array (NUMERIC type)
   * @param predictions Predictions array (NUMERIC type)
   * @param weights Weights array. May be null. If null, a weight of 1.0 is used (NUMERIC type)
   * @param lossReduce Reduction type for the loss. See {@link LossReduce} for more details. Default: {@link LossReduce#MEAN_BY_NONZERO_WEIGHT_COUNT}
   * @param epsilon epsilon
   * @return output Log loss  (NUMERIC type)
   */
  public SDVariable logLoss(String name, SDVariable label, SDVariable predictions,
      SDVariable weights, LossReduce lossReduce, double epsilon) {
    SDValidation.validateNumerical("logLoss", "label", label);
    SDValidation.validateNumerical("logLoss", "predictions", predictions);
    SDValidation.validateNumerical("logLoss", "weights", weights);
    SDVariable out =  new org.nd4j.linalg.api.ops.impl.loss.LogLoss(sd,label, predictions, weights, lossReduce, epsilon).outputVariable();
    out.markAsLoss();
    return sd.updateVariableNameAndReference(out, name);
  }

  /**
   * Log loss, i.e., binary cross entropy loss, usually used for binary multi-label classification. Implements:<br>
   * {@code -1/numExamples * sum_i (labels[i] * log(predictions[i] + epsilon) + (1-labels[i]) * log(1-predictions[i] + epsilon))}<br>
   *
   * @param label Label array (NUMERIC type)
   * @param predictions Predictions array (NUMERIC type)
   * @return output Log loss  (NUMERIC type)
   */
  public SDVariable logLoss(SDVariable label, SDVariable predictions) {
    SDValidation.validateNumerical("logLoss", "label", label);
    SDValidation.validateNumerical("logLoss", "predictions", predictions);
    SDVariable out = new org.nd4j.linalg.api.ops.impl.loss.LogLoss(sd,label, predictions, null, org.nd4j.autodiff.loss.LossReduce.MEAN_BY_NONZERO_WEIGHT_COUNT, 0.0).outputVariable();
    out.markAsLoss();
    return out;
  }

  /**
   * Log loss, i.e., binary cross entropy loss, usually used for binary multi-label classification. Implements:<br>
   * {@code -1/numExamples * sum_i (labels[i] * log(predictions[i] + epsilon) + (1-labels[i]) * log(1-predictions[i] + epsilon))}<br>
   *
   * @param name name May be null. Name for the output variable
   * @param label Label array (NUMERIC type)
   * @param predictions Predictions array (NUMERIC type)
   * @return output Log loss  (NUMERIC type)
   */
  public SDVariable logLoss(String name, SDVariable label, SDVariable predictions) {
    SDValidation.validateNumerical("logLoss", "label", label);
    SDValidation.validateNumerical("logLoss", "predictions", predictions);
    SDVariable out =  new org.nd4j.linalg.api.ops.impl.loss.LogLoss(sd,label, predictions, null, org.nd4j.autodiff.loss.LossReduce.MEAN_BY_NONZERO_WEIGHT_COUNT, 0.0).outputVariable();
    out.markAsLoss();
    return sd.updateVariableNameAndReference(out, name);
  }

  /**
   * Log poisson loss: a loss function used for training classifiers.<br>
   * Implements {@code L = exp(c) - z * c} where c is log(predictions) and z is labels.<br>
   *
   * @param label Label array. Each value should be 0.0 or 1.0 (NUMERIC type)
   * @param predictions Predictions array (has to be log(x) of actual predictions) (NUMERIC type)
   * @param weights Weights array. May be null. If null, a weight of 1.0 is used (NUMERIC type)
   * @param lossReduce Reduction type for the loss. See {@link LossReduce} for more details. Default: {@link LossReduce#MEAN_BY_NONZERO_WEIGHT_COUNT}
   * @param full Boolean flag. true for logPoissonFull, false for logPoisson
   * @return output Loss variable (NUMERIC type)
   */
  public SDVariable logPoisson(SDVariable label, SDVariable predictions, SDVariable weights,
      LossReduce lossReduce, boolean full) {
    SDValidation.validateNumerical("logPoisson", "label", label);
    SDValidation.validateNumerical("logPoisson", "predictions", predictions);
    SDValidation.validateNumerical("logPoisson", "weights", weights);
    SDVariable out = new org.nd4j.linalg.api.ops.impl.loss.LogPoissonLoss(sd,label, predictions, weights, lossReduce, full).outputVariable();
    out.markAsLoss();
    return out;
  }

  /**
   * Log poisson loss: a loss function used for training classifiers.<br>
   * Implements {@code L = exp(c) - z * c} where c is log(predictions) and z is labels.<br>
   *
   * @param name name May be null. Name for the output variable
   * @param label Label array. Each value should be 0.0 or 1.0 (NUMERIC type)
   * @param predictions Predictions array (has to be log(x) of actual predictions) (NUMERIC type)
   * @param weights Weights array. May be null. If null, a weight of 1.0 is used (NUMERIC type)
   * @param lossReduce Reduction type for the loss. See {@link LossReduce} for more details. Default: {@link LossReduce#MEAN_BY_NONZERO_WEIGHT_COUNT}
   * @param full Boolean flag. true for logPoissonFull, false for logPoisson
   * @return output Loss variable (NUMERIC type)
   */
  public SDVariable logPoisson(String name, SDVariable label, SDVariable predictions,
      SDVariable weights, LossReduce lossReduce, boolean full) {
    SDValidation.validateNumerical("logPoisson", "label", label);
    SDValidation.validateNumerical("logPoisson", "predictions", predictions);
    SDValidation.validateNumerical("logPoisson", "weights", weights);
    SDVariable out =  new org.nd4j.linalg.api.ops.impl.loss.LogPoissonLoss(sd,label, predictions, weights, lossReduce, full).outputVariable();
    out.markAsLoss();
    return sd.updateVariableNameAndReference(out, name);
  }

  /**
   * Log poisson loss: a loss function used for training classifiers.<br>
   * Implements {@code L = exp(c) - z * c} where c is log(predictions) and z is labels.<br>
   *
   * @param label Label array. Each value should be 0.0 or 1.0 (NUMERIC type)
   * @param predictions Predictions array (has to be log(x) of actual predictions) (NUMERIC type)
   * @param weights Weights array. May be null. If null, a weight of 1.0 is used (NUMERIC type)
   * @param full Boolean flag. true for logPoissonFull, false for logPoisson
   * @return output Loss variable (NUMERIC type)
   */
  public SDVariable logPoisson(SDVariable label, SDVariable predictions, SDVariable weights,
      boolean full) {
    SDValidation.validateNumerical("logPoisson", "label", label);
    SDValidation.validateNumerical("logPoisson", "predictions", predictions);
    SDValidation.validateNumerical("logPoisson", "weights", weights);
    SDVariable out = new org.nd4j.linalg.api.ops.impl.loss.LogPoissonLoss(sd,label, predictions, weights, org.nd4j.autodiff.loss.LossReduce.MEAN_BY_NONZERO_WEIGHT_COUNT, full).outputVariable();
    out.markAsLoss();
    return out;
  }

  /**
   * Log poisson loss: a loss function used for training classifiers.<br>
   * Implements {@code L = exp(c) - z * c} where c is log(predictions) and z is labels.<br>
   *
   * @param name name May be null. Name for the output variable
   * @param label Label array. Each value should be 0.0 or 1.0 (NUMERIC type)
   * @param predictions Predictions array (has to be log(x) of actual predictions) (NUMERIC type)
   * @param weights Weights array. May be null. If null, a weight of 1.0 is used (NUMERIC type)
   * @param full Boolean flag. true for logPoissonFull, false for logPoisson
   * @return output Loss variable (NUMERIC type)
   */
  public SDVariable logPoisson(String name, SDVariable label, SDVariable predictions,
      SDVariable weights, boolean full) {
    SDValidation.validateNumerical("logPoisson", "label", label);
    SDValidation.validateNumerical("logPoisson", "predictions", predictions);
    SDValidation.validateNumerical("logPoisson", "weights", weights);
    SDVariable out =  new org.nd4j.linalg.api.ops.impl.loss.LogPoissonLoss(sd,label, predictions, weights, org.nd4j.autodiff.loss.LossReduce.MEAN_BY_NONZERO_WEIGHT_COUNT, full).outputVariable();
    out.markAsLoss();
    return sd.updateVariableNameAndReference(out, name);
  }

  /**
   * Mean pairwise squared error.<br>
   * MPWSE loss calculates the difference between pairs of consecutive elements in the predictions and labels arrays.<br>
   * For example, if predictions = [p0, p1, p2] and labels are [l0, l1, l2] then MPWSE is:<br>
   * {@code [((p0-p1) - (l0-l1))^2 + ((p0-p2) - (l0-l2))^2 + ((p1-p2) - (l1-l2))^2] / 3}<br>
   *
   * @param label Label array (NUMERIC type)
   * @param predictions Predictions array (NUMERIC type)
   * @param weights Weights array. May be null. If null, a weight of 1.0 is used. Must be either null, scalar, or have shape [batchSize] (NUMERIC type)
   * @param lossReduce Reduction type for the loss. See {@link LossReduce} for more details. Default: {@link LossReduce#MEAN_BY_NONZERO_WEIGHT_COUNT}
   * @return output Loss variable, scalar output (NUMERIC type)
   */
  public SDVariable meanPairwiseSquaredError(SDVariable label, SDVariable predictions,
      SDVariable weights, LossReduce lossReduce) {
    SDValidation.validateNumerical("meanPairwiseSquaredError", "label", label);
    SDValidation.validateNumerical("meanPairwiseSquaredError", "predictions", predictions);
    SDValidation.validateNumerical("meanPairwiseSquaredError", "weights", weights);
    SDVariable out = new org.nd4j.linalg.api.ops.impl.loss.MeanPairwiseSquaredErrorLoss(sd,label, predictions, weights, lossReduce).outputVariable();
    out.markAsLoss();
    return out;
  }

  /**
   * Mean pairwise squared error.<br>
   * MPWSE loss calculates the difference between pairs of consecutive elements in the predictions and labels arrays.<br>
   * For example, if predictions = [p0, p1, p2] and labels are [l0, l1, l2] then MPWSE is:<br>
   * {@code [((p0-p1) - (l0-l1))^2 + ((p0-p2) - (l0-l2))^2 + ((p1-p2) - (l1-l2))^2] / 3}<br>
   *
   * @param name name May be null. Name for the output variable
   * @param label Label array (NUMERIC type)
   * @param predictions Predictions array (NUMERIC type)
   * @param weights Weights array. May be null. If null, a weight of 1.0 is used. Must be either null, scalar, or have shape [batchSize] (NUMERIC type)
   * @param lossReduce Reduction type for the loss. See {@link LossReduce} for more details. Default: {@link LossReduce#MEAN_BY_NONZERO_WEIGHT_COUNT}
   * @return output Loss variable, scalar output (NUMERIC type)
   */
  public SDVariable meanPairwiseSquaredError(String name, SDVariable label, SDVariable predictions,
      SDVariable weights, LossReduce lossReduce) {
    SDValidation.validateNumerical("meanPairwiseSquaredError", "label", label);
    SDValidation.validateNumerical("meanPairwiseSquaredError", "predictions", predictions);
    SDValidation.validateNumerical("meanPairwiseSquaredError", "weights", weights);
    SDVariable out =  new org.nd4j.linalg.api.ops.impl.loss.MeanPairwiseSquaredErrorLoss(sd,label, predictions, weights, lossReduce).outputVariable();
    out.markAsLoss();
    return sd.updateVariableNameAndReference(out, name);
  }

  /**
   * Mean pairwise squared error.<br>
   * MPWSE loss calculates the difference between pairs of consecutive elements in the predictions and labels arrays.<br>
   * For example, if predictions = [p0, p1, p2] and labels are [l0, l1, l2] then MPWSE is:<br>
   * {@code [((p0-p1) - (l0-l1))^2 + ((p0-p2) - (l0-l2))^2 + ((p1-p2) - (l1-l2))^2] / 3}<br>
   *
   * @param label Label array (NUMERIC type)
   * @param predictions Predictions array (NUMERIC type)
   * @param weights Weights array. May be null. If null, a weight of 1.0 is used. Must be either null, scalar, or have shape [batchSize] (NUMERIC type)
   * @return output Loss variable, scalar output (NUMERIC type)
   */
  public SDVariable meanPairwiseSquaredError(SDVariable label, SDVariable predictions,
      SDVariable weights) {
    SDValidation.validateNumerical("meanPairwiseSquaredError", "label", label);
    SDValidation.validateNumerical("meanPairwiseSquaredError", "predictions", predictions);
    SDValidation.validateNumerical("meanPairwiseSquaredError", "weights", weights);
    SDVariable out = new org.nd4j.linalg.api.ops.impl.loss.MeanPairwiseSquaredErrorLoss(sd,label, predictions, weights, org.nd4j.autodiff.loss.LossReduce.MEAN_BY_NONZERO_WEIGHT_COUNT).outputVariable();
    out.markAsLoss();
    return out;
  }

  /**
   * Mean pairwise squared error.<br>
   * MPWSE loss calculates the difference between pairs of consecutive elements in the predictions and labels arrays.<br>
   * For example, if predictions = [p0, p1, p2] and labels are [l0, l1, l2] then MPWSE is:<br>
   * {@code [((p0-p1) - (l0-l1))^2 + ((p0-p2) - (l0-l2))^2 + ((p1-p2) - (l1-l2))^2] / 3}<br>
   *
   * @param name name May be null. Name for the output variable
   * @param label Label array (NUMERIC type)
   * @param predictions Predictions array (NUMERIC type)
   * @param weights Weights array. May be null. If null, a weight of 1.0 is used. Must be either null, scalar, or have shape [batchSize] (NUMERIC type)
   * @return output Loss variable, scalar output (NUMERIC type)
   */
  public SDVariable meanPairwiseSquaredError(String name, SDVariable label, SDVariable predictions,
      SDVariable weights) {
    SDValidation.validateNumerical("meanPairwiseSquaredError", "label", label);
    SDValidation.validateNumerical("meanPairwiseSquaredError", "predictions", predictions);
    SDValidation.validateNumerical("meanPairwiseSquaredError", "weights", weights);
    SDVariable out =  new org.nd4j.linalg.api.ops.impl.loss.MeanPairwiseSquaredErrorLoss(sd,label, predictions, weights, org.nd4j.autodiff.loss.LossReduce.MEAN_BY_NONZERO_WEIGHT_COUNT).outputVariable();
    out.markAsLoss();
    return sd.updateVariableNameAndReference(out, name);
  }

  /**
   * Mean squared error loss function. Implements {@code (label[i] - prediction[i])^2} - i.e., squared error on a per-element basis.<br>
   * When averaged (using {@link LossReduce#MEAN_BY_WEIGHT} or {@link LossReduce#MEAN_BY_NONZERO_WEIGHT_COUNT} (the default))<br>
   * this is the mean squared error loss function.<br>
   *
   * @param label Label array (NUMERIC type)
   * @param predictions Predictions array (NUMERIC type)
   * @param weights Weights array. May be null. If null, a weight of 1.0 is used (NUMERIC type)
   * @param lossReduce Reduction type for the loss. See {@link LossReduce} for more details. Default: {@link LossReduce#MEAN_BY_NONZERO_WEIGHT_COUNT}
   * @return output Loss variable (NUMERIC type)
   */
  public SDVariable meanSquaredError(SDVariable label, SDVariable predictions, SDVariable weights,
      LossReduce lossReduce) {
    SDValidation.validateNumerical("meanSquaredError", "label", label);
    SDValidation.validateNumerical("meanSquaredError", "predictions", predictions);
    SDValidation.validateNumerical("meanSquaredError", "weights", weights);
    SDVariable out = new org.nd4j.linalg.api.ops.impl.loss.MeanSquaredErrorLoss(sd,label, predictions, weights, lossReduce).outputVariable();
    out.markAsLoss();
    return out;
  }

  /**
   * Mean squared error loss function. Implements {@code (label[i] - prediction[i])^2} - i.e., squared error on a per-element basis.<br>
   * When averaged (using {@link LossReduce#MEAN_BY_WEIGHT} or {@link LossReduce#MEAN_BY_NONZERO_WEIGHT_COUNT} (the default))<br>
   * this is the mean squared error loss function.<br>
   *
   * @param name name May be null. Name for the output variable
   * @param label Label array (NUMERIC type)
   * @param predictions Predictions array (NUMERIC type)
   * @param weights Weights array. May be null. If null, a weight of 1.0 is used (NUMERIC type)
   * @param lossReduce Reduction type for the loss. See {@link LossReduce} for more details. Default: {@link LossReduce#MEAN_BY_NONZERO_WEIGHT_COUNT}
   * @return output Loss variable (NUMERIC type)
   */
  public SDVariable meanSquaredError(String name, SDVariable label, SDVariable predictions,
      SDVariable weights, LossReduce lossReduce) {
    SDValidation.validateNumerical("meanSquaredError", "label", label);
    SDValidation.validateNumerical("meanSquaredError", "predictions", predictions);
    SDValidation.validateNumerical("meanSquaredError", "weights", weights);
    SDVariable out =  new org.nd4j.linalg.api.ops.impl.loss.MeanSquaredErrorLoss(sd,label, predictions, weights, lossReduce).outputVariable();
    out.markAsLoss();
    return sd.updateVariableNameAndReference(out, name);
  }

  /**
   * Mean squared error loss function. Implements {@code (label[i] - prediction[i])^2} - i.e., squared error on a per-element basis.<br>
   * When averaged (using {@link LossReduce#MEAN_BY_WEIGHT} or {@link LossReduce#MEAN_BY_NONZERO_WEIGHT_COUNT} (the default))<br>
   * this is the mean squared error loss function.<br>
   *
   * @param label Label array (NUMERIC type)
   * @param predictions Predictions array (NUMERIC type)
   * @param weights Weights array. May be null. If null, a weight of 1.0 is used (NUMERIC type)
   * @return output Loss variable (NUMERIC type)
   */
  public SDVariable meanSquaredError(SDVariable label, SDVariable predictions, SDVariable weights) {
    SDValidation.validateNumerical("meanSquaredError", "label", label);
    SDValidation.validateNumerical("meanSquaredError", "predictions", predictions);
    SDValidation.validateNumerical("meanSquaredError", "weights", weights);
    SDVariable out = new org.nd4j.linalg.api.ops.impl.loss.MeanSquaredErrorLoss(sd,label, predictions, weights, org.nd4j.autodiff.loss.LossReduce.MEAN_BY_NONZERO_WEIGHT_COUNT).outputVariable();
    out.markAsLoss();
    return out;
  }

  /**
   * Mean squared error loss function. Implements {@code (label[i] - prediction[i])^2} - i.e., squared error on a per-element basis.<br>
   * When averaged (using {@link LossReduce#MEAN_BY_WEIGHT} or {@link LossReduce#MEAN_BY_NONZERO_WEIGHT_COUNT} (the default))<br>
   * this is the mean squared error loss function.<br>
   *
   * @param name name May be null. Name for the output variable
   * @param label Label array (NUMERIC type)
   * @param predictions Predictions array (NUMERIC type)
   * @param weights Weights array. May be null. If null, a weight of 1.0 is used (NUMERIC type)
   * @return output Loss variable (NUMERIC type)
   */
  public SDVariable meanSquaredError(String name, SDVariable label, SDVariable predictions,
      SDVariable weights) {
    SDValidation.validateNumerical("meanSquaredError", "label", label);
    SDValidation.validateNumerical("meanSquaredError", "predictions", predictions);
    SDValidation.validateNumerical("meanSquaredError", "weights", weights);
    SDVariable out =  new org.nd4j.linalg.api.ops.impl.loss.MeanSquaredErrorLoss(sd,label, predictions, weights, org.nd4j.autodiff.loss.LossReduce.MEAN_BY_NONZERO_WEIGHT_COUNT).outputVariable();
    out.markAsLoss();
    return sd.updateVariableNameAndReference(out, name);
  }

  /**
   * Sigmoid cross entropy: applies the sigmoid activation function on the input logits (input "pre-sigmoid preductions")<br>
   * and implements the binary cross entropy loss function. This implementation is numerically more stable than using<br>
   * standard (but separate) sigmoid activation function and log loss (binary cross entropy) loss function.<br>
   * Implements:<br>
   * {@code -1/numExamples * sum_i (labels[i] * log(sigmoid(logits[i])) + (1-labels[i]) * log(1-sigmoid(logits[i])))}<br>
   * though this is done in a mathematically equivalent but more numerical stable form.<br>
   * <br>
   * When label smoothing is > 0, the following label smoothing is used:<br>
   * <pre><br>
   * {@code numClasses = labels.size(1);<br>
   * label = (1.0 - labelSmoothing) * label + 0.5 * labelSmoothing}<br>
   * </pre><br>
   *
   * @param label Label array (NUMERIC type)
   * @param predictionLogits Predictions array (NUMERIC type)
   * @param weights Weights array. May be null. If null, a weight of 1.0 is used (NUMERIC type)
   * @param lossReduce Reduction type for the loss. See {@link LossReduce} for more details. Default: {@link LossReduce#MEAN_BY_NONZERO_WEIGHT_COUNT}
   * @param labelSmoothing Label smoothing value. Default value: 0
   * @return output Loss variable (NUMERIC type)
   */
  public SDVariable sigmoidCrossEntropy(SDVariable label, SDVariable predictionLogits,
      SDVariable weights, LossReduce lossReduce, double labelSmoothing) {
    SDValidation.validateNumerical("sigmoidCrossEntropy", "label", label);
    SDValidation.validateNumerical("sigmoidCrossEntropy", "predictionLogits", predictionLogits);
    SDValidation.validateNumerical("sigmoidCrossEntropy", "weights", weights);
    SDVariable out = new org.nd4j.linalg.api.ops.impl.loss.SigmoidCrossEntropyLoss(sd,label, predictionLogits, weights, lossReduce, labelSmoothing).outputVariable();
    out.markAsLoss();
    return out;
  }

  /**
   * Sigmoid cross entropy: applies the sigmoid activation function on the input logits (input "pre-sigmoid preductions")<br>
   * and implements the binary cross entropy loss function. This implementation is numerically more stable than using<br>
   * standard (but separate) sigmoid activation function and log loss (binary cross entropy) loss function.<br>
   * Implements:<br>
   * {@code -1/numExamples * sum_i (labels[i] * log(sigmoid(logits[i])) + (1-labels[i]) * log(1-sigmoid(logits[i])))}<br>
   * though this is done in a mathematically equivalent but more numerical stable form.<br>
   * <br>
   * When label smoothing is > 0, the following label smoothing is used:<br>
   * <pre><br>
   * {@code numClasses = labels.size(1);<br>
   * label = (1.0 - labelSmoothing) * label + 0.5 * labelSmoothing}<br>
   * </pre><br>
   *
   * @param name name May be null. Name for the output variable
   * @param label Label array (NUMERIC type)
   * @param predictionLogits Predictions array (NUMERIC type)
   * @param weights Weights array. May be null. If null, a weight of 1.0 is used (NUMERIC type)
   * @param lossReduce Reduction type for the loss. See {@link LossReduce} for more details. Default: {@link LossReduce#MEAN_BY_NONZERO_WEIGHT_COUNT}
   * @param labelSmoothing Label smoothing value. Default value: 0
   * @return output Loss variable (NUMERIC type)
   */
  public SDVariable sigmoidCrossEntropy(String name, SDVariable label, SDVariable predictionLogits,
      SDVariable weights, LossReduce lossReduce, double labelSmoothing) {
    SDValidation.validateNumerical("sigmoidCrossEntropy", "label", label);
    SDValidation.validateNumerical("sigmoidCrossEntropy", "predictionLogits", predictionLogits);
    SDValidation.validateNumerical("sigmoidCrossEntropy", "weights", weights);
    SDVariable out =  new org.nd4j.linalg.api.ops.impl.loss.SigmoidCrossEntropyLoss(sd,label, predictionLogits, weights, lossReduce, labelSmoothing).outputVariable();
    out.markAsLoss();
    return sd.updateVariableNameAndReference(out, name);
  }

  /**
   * Sigmoid cross entropy: applies the sigmoid activation function on the input logits (input "pre-sigmoid preductions")<br>
   * and implements the binary cross entropy loss function. This implementation is numerically more stable than using<br>
   * standard (but separate) sigmoid activation function and log loss (binary cross entropy) loss function.<br>
   * Implements:<br>
   * {@code -1/numExamples * sum_i (labels[i] * log(sigmoid(logits[i])) + (1-labels[i]) * log(1-sigmoid(logits[i])))}<br>
   * though this is done in a mathematically equivalent but more numerical stable form.<br>
   * <br>
   * When label smoothing is > 0, the following label smoothing is used:<br>
   * <pre><br>
   * {@code numClasses = labels.size(1);<br>
   * label = (1.0 - labelSmoothing) * label + 0.5 * labelSmoothing}<br>
   * </pre><br>
   *
   * @param label Label array (NUMERIC type)
   * @param predictionLogits Predictions array (NUMERIC type)
   * @param weights Weights array. May be null. If null, a weight of 1.0 is used (NUMERIC type)
   * @return output Loss variable (NUMERIC type)
   */
  public SDVariable sigmoidCrossEntropy(SDVariable label, SDVariable predictionLogits,
      SDVariable weights) {
    SDValidation.validateNumerical("sigmoidCrossEntropy", "label", label);
    SDValidation.validateNumerical("sigmoidCrossEntropy", "predictionLogits", predictionLogits);
    SDValidation.validateNumerical("sigmoidCrossEntropy", "weights", weights);
    SDVariable out = new org.nd4j.linalg.api.ops.impl.loss.SigmoidCrossEntropyLoss(sd,label, predictionLogits, weights, org.nd4j.autodiff.loss.LossReduce.MEAN_BY_NONZERO_WEIGHT_COUNT, 0.0).outputVariable();
    out.markAsLoss();
    return out;
  }

  /**
   * Sigmoid cross entropy: applies the sigmoid activation function on the input logits (input "pre-sigmoid preductions")<br>
   * and implements the binary cross entropy loss function. This implementation is numerically more stable than using<br>
   * standard (but separate) sigmoid activation function and log loss (binary cross entropy) loss function.<br>
   * Implements:<br>
   * {@code -1/numExamples * sum_i (labels[i] * log(sigmoid(logits[i])) + (1-labels[i]) * log(1-sigmoid(logits[i])))}<br>
   * though this is done in a mathematically equivalent but more numerical stable form.<br>
   * <br>
   * When label smoothing is > 0, the following label smoothing is used:<br>
   * <pre><br>
   * {@code numClasses = labels.size(1);<br>
   * label = (1.0 - labelSmoothing) * label + 0.5 * labelSmoothing}<br>
   * </pre><br>
   *
   * @param name name May be null. Name for the output variable
   * @param label Label array (NUMERIC type)
   * @param predictionLogits Predictions array (NUMERIC type)
   * @param weights Weights array. May be null. If null, a weight of 1.0 is used (NUMERIC type)
   * @return output Loss variable (NUMERIC type)
   */
  public SDVariable sigmoidCrossEntropy(String name, SDVariable label, SDVariable predictionLogits,
      SDVariable weights) {
    SDValidation.validateNumerical("sigmoidCrossEntropy", "label", label);
    SDValidation.validateNumerical("sigmoidCrossEntropy", "predictionLogits", predictionLogits);
    SDValidation.validateNumerical("sigmoidCrossEntropy", "weights", weights);
    SDVariable out =  new org.nd4j.linalg.api.ops.impl.loss.SigmoidCrossEntropyLoss(sd,label, predictionLogits, weights, org.nd4j.autodiff.loss.LossReduce.MEAN_BY_NONZERO_WEIGHT_COUNT, 0.0).outputVariable();
    out.markAsLoss();
    return sd.updateVariableNameAndReference(out, name);
  }

  /**
   * Applies the softmax activation function to the input, then implement multi-class cross entropy:<br>
   * {@code -sum_classes label[i] * log(p[c])} where {@code p = softmax(logits)}<br>
   * If {@link LossReduce#NONE} is used, returned shape is [numExamples] out for [numExamples, numClasses] predicitons/labels;<br>
   * otherwise, the output is a scalar.<br>
   * <p><br>
   * When label smoothing is > 0, the following label smoothing is used:<br>
   * <pre><br>
   * {@code numClasses = labels.size(1);<br>
   * oneHotLabel = (1.0 - labelSmoothing) * oneHotLabels + labelSmoothing/numClasses}<br>
   * </pre><br>
   *
   * @param oneHotLabels Label array. Should be one-hot per example and same shape as predictions (for example, [mb, nOut]) (NUMERIC type)
   * @param logitPredictions Predictions array (pre-softmax) (NUMERIC type)
   * @param weights Weights array. May be null. If null, a weight of 1.0 is used (NUMERIC type)
   * @param lossReduce Reduction type for the loss. See {@link LossReduce} for more details. Default: {@link LossReduce#MEAN_BY_NONZERO_WEIGHT_COUNT}
   * @param labelSmoothing Label smoothing value. Default value: 0
   * @return output Loss variable (NUMERIC type)
   */
  public SDVariable softmaxCrossEntropy(SDVariable oneHotLabels, SDVariable logitPredictions,
      SDVariable weights, LossReduce lossReduce, double labelSmoothing) {
    SDValidation.validateNumerical("softmaxCrossEntropy", "oneHotLabels", oneHotLabels);
    SDValidation.validateNumerical("softmaxCrossEntropy", "logitPredictions", logitPredictions);
    SDValidation.validateNumerical("softmaxCrossEntropy", "weights", weights);
    SDVariable out = new org.nd4j.linalg.api.ops.impl.loss.SoftmaxCrossEntropyLoss(sd,oneHotLabels, logitPredictions, weights, lossReduce, labelSmoothing).outputVariable();
    out.markAsLoss();
    return out;
  }

  /**
   * Applies the softmax activation function to the input, then implement multi-class cross entropy:<br>
   * {@code -sum_classes label[i] * log(p[c])} where {@code p = softmax(logits)}<br>
   * If {@link LossReduce#NONE} is used, returned shape is [numExamples] out for [numExamples, numClasses] predicitons/labels;<br>
   * otherwise, the output is a scalar.<br>
   * <p><br>
   * When label smoothing is > 0, the following label smoothing is used:<br>
   * <pre><br>
   * {@code numClasses = labels.size(1);<br>
   * oneHotLabel = (1.0 - labelSmoothing) * oneHotLabels + labelSmoothing/numClasses}<br>
   * </pre><br>
   *
   * @param name name May be null. Name for the output variable
   * @param oneHotLabels Label array. Should be one-hot per example and same shape as predictions (for example, [mb, nOut]) (NUMERIC type)
   * @param logitPredictions Predictions array (pre-softmax) (NUMERIC type)
   * @param weights Weights array. May be null. If null, a weight of 1.0 is used (NUMERIC type)
   * @param lossReduce Reduction type for the loss. See {@link LossReduce} for more details. Default: {@link LossReduce#MEAN_BY_NONZERO_WEIGHT_COUNT}
   * @param labelSmoothing Label smoothing value. Default value: 0
   * @return output Loss variable (NUMERIC type)
   */
  public SDVariable softmaxCrossEntropy(String name, SDVariable oneHotLabels,
      SDVariable logitPredictions, SDVariable weights, LossReduce lossReduce,
      double labelSmoothing) {
    SDValidation.validateNumerical("softmaxCrossEntropy", "oneHotLabels", oneHotLabels);
    SDValidation.validateNumerical("softmaxCrossEntropy", "logitPredictions", logitPredictions);
    SDValidation.validateNumerical("softmaxCrossEntropy", "weights", weights);
    SDVariable out =  new org.nd4j.linalg.api.ops.impl.loss.SoftmaxCrossEntropyLoss(sd,oneHotLabels, logitPredictions, weights, lossReduce, labelSmoothing).outputVariable();
    out.markAsLoss();
    return sd.updateVariableNameAndReference(out, name);
  }

  /**
   * Applies the softmax activation function to the input, then implement multi-class cross entropy:<br>
   * {@code -sum_classes label[i] * log(p[c])} where {@code p = softmax(logits)}<br>
   * If {@link LossReduce#NONE} is used, returned shape is [numExamples] out for [numExamples, numClasses] predicitons/labels;<br>
   * otherwise, the output is a scalar.<br>
   * <p><br>
   * When label smoothing is > 0, the following label smoothing is used:<br>
   * <pre><br>
   * {@code numClasses = labels.size(1);<br>
   * oneHotLabel = (1.0 - labelSmoothing) * oneHotLabels + labelSmoothing/numClasses}<br>
   * </pre><br>
   *
   * @param oneHotLabels Label array. Should be one-hot per example and same shape as predictions (for example, [mb, nOut]) (NUMERIC type)
   * @param logitPredictions Predictions array (pre-softmax) (NUMERIC type)
   * @param weights Weights array. May be null. If null, a weight of 1.0 is used (NUMERIC type)
   * @return output Loss variable (NUMERIC type)
   */
  public SDVariable softmaxCrossEntropy(SDVariable oneHotLabels, SDVariable logitPredictions,
      SDVariable weights) {
    SDValidation.validateNumerical("softmaxCrossEntropy", "oneHotLabels", oneHotLabels);
    SDValidation.validateNumerical("softmaxCrossEntropy", "logitPredictions", logitPredictions);
    SDValidation.validateNumerical("softmaxCrossEntropy", "weights", weights);
    SDVariable out = new org.nd4j.linalg.api.ops.impl.loss.SoftmaxCrossEntropyLoss(sd,oneHotLabels, logitPredictions, weights, org.nd4j.autodiff.loss.LossReduce.MEAN_BY_NONZERO_WEIGHT_COUNT, 0.0).outputVariable();
    out.markAsLoss();
    return out;
  }

  /**
   * Applies the softmax activation function to the input, then implement multi-class cross entropy:<br>
   * {@code -sum_classes label[i] * log(p[c])} where {@code p = softmax(logits)}<br>
   * If {@link LossReduce#NONE} is used, returned shape is [numExamples] out for [numExamples, numClasses] predicitons/labels;<br>
   * otherwise, the output is a scalar.<br>
   * <p><br>
   * When label smoothing is > 0, the following label smoothing is used:<br>
   * <pre><br>
   * {@code numClasses = labels.size(1);<br>
   * oneHotLabel = (1.0 - labelSmoothing) * oneHotLabels + labelSmoothing/numClasses}<br>
   * </pre><br>
   *
   * @param name name May be null. Name for the output variable
   * @param oneHotLabels Label array. Should be one-hot per example and same shape as predictions (for example, [mb, nOut]) (NUMERIC type)
   * @param logitPredictions Predictions array (pre-softmax) (NUMERIC type)
   * @param weights Weights array. May be null. If null, a weight of 1.0 is used (NUMERIC type)
   * @return output Loss variable (NUMERIC type)
   */
  public SDVariable softmaxCrossEntropy(String name, SDVariable oneHotLabels,
      SDVariable logitPredictions, SDVariable weights) {
    SDValidation.validateNumerical("softmaxCrossEntropy", "oneHotLabels", oneHotLabels);
    SDValidation.validateNumerical("softmaxCrossEntropy", "logitPredictions", logitPredictions);
    SDValidation.validateNumerical("softmaxCrossEntropy", "weights", weights);
    SDVariable out =  new org.nd4j.linalg.api.ops.impl.loss.SoftmaxCrossEntropyLoss(sd,oneHotLabels, logitPredictions, weights, org.nd4j.autodiff.loss.LossReduce.MEAN_BY_NONZERO_WEIGHT_COUNT, 0.0).outputVariable();
    out.markAsLoss();
    return sd.updateVariableNameAndReference(out, name);
  }

  /**
   * As per softmaxCrossEntropy(String, SDVariable, SDVariable, LossReduce) but the labels variable<br>
   * is represented as an integer array instead of the equivalent one-hot array.<br>
   * i.e., if logits are rank N, then labels have rank N-1<br>
   *
   * @param logits Logits array ("pre-softmax activations") (NUMERIC type)
   * @param labels Labels array. Must be an integer type. (INT type)
   * @return output Softmax cross entropy (NUMERIC type)
   */
  public SDVariable sparseSoftmaxCrossEntropy(SDVariable logits, SDVariable labels) {
    SDValidation.validateNumerical("sparseSoftmaxCrossEntropy", "logits", logits);
    SDValidation.validateInteger("sparseSoftmaxCrossEntropy", "labels", labels);
    SDVariable out = new org.nd4j.linalg.api.ops.impl.loss.SparseSoftmaxCrossEntropyLossWithLogits(sd,logits, labels).outputVariable();
    out.markAsLoss();
    return out;
  }

  /**
   * As per softmaxCrossEntropy(String, SDVariable, SDVariable, LossReduce) but the labels variable<br>
   * is represented as an integer array instead of the equivalent one-hot array.<br>
   * i.e., if logits are rank N, then labels have rank N-1<br>
   *
   * @param name name May be null. Name for the output variable
   * @param logits Logits array ("pre-softmax activations") (NUMERIC type)
   * @param labels Labels array. Must be an integer type. (INT type)
   * @return output Softmax cross entropy (NUMERIC type)
   */
  public SDVariable sparseSoftmaxCrossEntropy(String name, SDVariable logits, SDVariable labels) {
    SDValidation.validateNumerical("sparseSoftmaxCrossEntropy", "logits", logits);
    SDValidation.validateInteger("sparseSoftmaxCrossEntropy", "labels", labels);
    SDVariable out =  new org.nd4j.linalg.api.ops.impl.loss.SparseSoftmaxCrossEntropyLossWithLogits(sd,logits, labels).outputVariable();
    out.markAsLoss();
    return sd.updateVariableNameAndReference(out, name);
  }

  /**
   * Weighted cross entropy loss with logits<br>
   *
   * @param targets targets array (NUMERIC type)
   * @param inputs input array (NUMERIC type)
   * @param weights eights array. May be null. If null, a weight of 1.0 is used (NUMERIC type)
   * @return output Loss variable (NUMERIC type)
   */
  public SDVariable weightedCrossEntropyWithLogits(SDVariable targets, SDVariable inputs,
      SDVariable weights) {
    SDValidation.validateNumerical("weightedCrossEntropyWithLogits", "targets", targets);
    SDValidation.validateNumerical("weightedCrossEntropyWithLogits", "inputs", inputs);
    SDValidation.validateNumerical("weightedCrossEntropyWithLogits", "weights", weights);
    SDVariable out = new org.nd4j.linalg.api.ops.impl.loss.WeightedCrossEntropyLoss(sd,targets, inputs, weights).outputVariable();
    out.markAsLoss();
    return out;
  }

  /**
   * Weighted cross entropy loss with logits<br>
   *
   * @param name name May be null. Name for the output variable
   * @param targets targets array (NUMERIC type)
   * @param inputs input array (NUMERIC type)
   * @param weights eights array. May be null. If null, a weight of 1.0 is used (NUMERIC type)
   * @return output Loss variable (NUMERIC type)
   */
  public SDVariable weightedCrossEntropyWithLogits(String name, SDVariable targets,
      SDVariable inputs, SDVariable weights) {
    SDValidation.validateNumerical("weightedCrossEntropyWithLogits", "targets", targets);
    SDValidation.validateNumerical("weightedCrossEntropyWithLogits", "inputs", inputs);
    SDValidation.validateNumerical("weightedCrossEntropyWithLogits", "weights", weights);
    SDVariable out =  new org.nd4j.linalg.api.ops.impl.loss.WeightedCrossEntropyLoss(sd,targets, inputs, weights).outputVariable();
    out.markAsLoss();
    return sd.updateVariableNameAndReference(out, name);
  }
}
