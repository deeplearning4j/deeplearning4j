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
   * Absolute difference loss: {@code sum_i abs( label[i] - predictions[i] )<br>
   *
   * @param label Label array (NUMERIC type)
   * @param predictions Predictions array (NUMERIC type)
   * @param weights Weights array. May be null. If null, a weight of 1.0 is used (NUMERIC type)
   * @return output loss variable (NUMERIC type)
   */
  public INDArray absoluteDifference(INDArray label, INDArray predictions, INDArray weights) {
    NDValidation.validateNumerical("absoluteDifference", "label", label);
    NDValidation.validateNumerical("absoluteDifference", "predictions", predictions);
    NDValidation.validateNumerical("absoluteDifference", "weights", weights);
    return Nd4j.exec(new org.nd4j.linalg.api.ops.impl.loss.AbsoluteDifferenceLoss(label, predictions, weights, org.nd4j.autodiff.loss.LossReduce.MEAN_BY_NONZERO_WEIGHT_COUNT))[0];
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
  public INDArray cosineDistance(INDArray label, INDArray predictions, INDArray weights,
      int dimension) {
    NDValidation.validateNumerical("cosineDistance", "label", label);
    NDValidation.validateNumerical("cosineDistance", "predictions", predictions);
    NDValidation.validateNumerical("cosineDistance", "weights", weights);
    return Nd4j.exec(new org.nd4j.linalg.api.ops.impl.loss.CosineDistanceLoss(label, predictions, weights, org.nd4j.autodiff.loss.LossReduce.MEAN_BY_NONZERO_WEIGHT_COUNT, dimension))[0];
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
   * Hinge loss: a loss function used for training classifiers.<br>
   * Implements {@code L = max(0, 1 - t * predictions)} where t is the label values after internally converting to {-1,1}<br>
   * from the user specified {0,1}. Note that Labels should be provided with values {0,1}.<br>
   *
   * @param label Label array. Each value should be 0.0 or 1.0 (internally -1 to 1 is used) (NUMERIC type)
   * @param predictions Predictions array (NUMERIC type)
   * @param weights Weights array. May be null. If null, a weight of 1.0 is used (NUMERIC type)
   * @return output Loss variable (NUMERIC type)
   */
  public INDArray hingeLoss(INDArray label, INDArray predictions, INDArray weights) {
    NDValidation.validateNumerical("hingeLoss", "label", label);
    NDValidation.validateNumerical("hingeLoss", "predictions", predictions);
    NDValidation.validateNumerical("hingeLoss", "weights", weights);
    return Nd4j.exec(new org.nd4j.linalg.api.ops.impl.loss.HingeLoss(label, predictions, weights, org.nd4j.autodiff.loss.LossReduce.MEAN_BY_NONZERO_WEIGHT_COUNT))[0];
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
   * @param delta Loss function delta value
   * @return output Huber loss (NUMERIC type)
   */
  public INDArray huberLoss(INDArray label, INDArray predictions, INDArray weights, double delta) {
    NDValidation.validateNumerical("huberLoss", "label", label);
    NDValidation.validateNumerical("huberLoss", "predictions", predictions);
    NDValidation.validateNumerical("huberLoss", "weights", weights);
    return Nd4j.exec(new org.nd4j.linalg.api.ops.impl.loss.HuberLoss(label, predictions, weights, org.nd4j.autodiff.loss.LossReduce.MEAN_BY_NONZERO_WEIGHT_COUNT, delta))[0];
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
  public INDArray logLoss(INDArray label, INDArray predictions, INDArray weights,
      LossReduce lossReduce, double epsilon) {
    NDValidation.validateNumerical("logLoss", "label", label);
    NDValidation.validateNumerical("logLoss", "predictions", predictions);
    NDValidation.validateNumerical("logLoss", "weights", weights);
    return Nd4j.exec(new org.nd4j.linalg.api.ops.impl.loss.LogLoss(label, predictions, weights, lossReduce, epsilon))[0];
  }

  /**
   * Log loss, i.e., binary cross entropy loss, usually used for binary multi-label classification. Implements:<br>
   * {@code -1/numExamples * sum_i (labels[i] * log(predictions[i] + epsilon) + (1-labels[i]) * log(1-predictions[i] + epsilon))}<br>
   *
   * @param label Label array (NUMERIC type)
   * @param predictions Predictions array (NUMERIC type)
   * @param weights Weights array. May be null. If null, a weight of 1.0 is used (NUMERIC type)
   * @param epsilon epsilon
   * @return output Log loss  (NUMERIC type)
   */
  public INDArray logLoss(INDArray label, INDArray predictions, INDArray weights, double epsilon) {
    NDValidation.validateNumerical("logLoss", "label", label);
    NDValidation.validateNumerical("logLoss", "predictions", predictions);
    NDValidation.validateNumerical("logLoss", "weights", weights);
    return Nd4j.exec(new org.nd4j.linalg.api.ops.impl.loss.LogLoss(label, predictions, weights, org.nd4j.autodiff.loss.LossReduce.MEAN_BY_NONZERO_WEIGHT_COUNT, epsilon))[0];
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
  public INDArray logPoisson(INDArray label, INDArray predictions, INDArray weights,
      LossReduce lossReduce, boolean full) {
    NDValidation.validateNumerical("logPoisson", "label", label);
    NDValidation.validateNumerical("logPoisson", "predictions", predictions);
    NDValidation.validateNumerical("logPoisson", "weights", weights);
    return Nd4j.exec(new org.nd4j.linalg.api.ops.impl.loss.LogPoissonLoss(label, predictions, weights, lossReduce, full))[0];
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
  public INDArray logPoisson(INDArray label, INDArray predictions, INDArray weights, boolean full) {
    NDValidation.validateNumerical("logPoisson", "label", label);
    NDValidation.validateNumerical("logPoisson", "predictions", predictions);
    NDValidation.validateNumerical("logPoisson", "weights", weights);
    return Nd4j.exec(new org.nd4j.linalg.api.ops.impl.loss.LogPoissonLoss(label, predictions, weights, org.nd4j.autodiff.loss.LossReduce.MEAN_BY_NONZERO_WEIGHT_COUNT, full))[0];
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
  public INDArray meanPairwiseSquaredError(INDArray label, INDArray predictions, INDArray weights,
      LossReduce lossReduce) {
    NDValidation.validateNumerical("meanPairwiseSquaredError", "label", label);
    NDValidation.validateNumerical("meanPairwiseSquaredError", "predictions", predictions);
    NDValidation.validateNumerical("meanPairwiseSquaredError", "weights", weights);
    return Nd4j.exec(new org.nd4j.linalg.api.ops.impl.loss.MeanPairwiseSquaredErrorLoss(label, predictions, weights, lossReduce))[0];
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
  public INDArray meanPairwiseSquaredError(INDArray label, INDArray predictions, INDArray weights) {
    NDValidation.validateNumerical("meanPairwiseSquaredError", "label", label);
    NDValidation.validateNumerical("meanPairwiseSquaredError", "predictions", predictions);
    NDValidation.validateNumerical("meanPairwiseSquaredError", "weights", weights);
    return Nd4j.exec(new org.nd4j.linalg.api.ops.impl.loss.MeanPairwiseSquaredErrorLoss(label, predictions, weights, org.nd4j.autodiff.loss.LossReduce.MEAN_BY_NONZERO_WEIGHT_COUNT))[0];
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
  public INDArray meanSquaredError(INDArray label, INDArray predictions, INDArray weights,
      LossReduce lossReduce) {
    NDValidation.validateNumerical("meanSquaredError", "label", label);
    NDValidation.validateNumerical("meanSquaredError", "predictions", predictions);
    NDValidation.validateNumerical("meanSquaredError", "weights", weights);
    return Nd4j.exec(new org.nd4j.linalg.api.ops.impl.loss.MeanSquaredErrorLoss(label, predictions, weights, lossReduce))[0];
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
  public INDArray meanSquaredError(INDArray label, INDArray predictions, INDArray weights) {
    NDValidation.validateNumerical("meanSquaredError", "label", label);
    NDValidation.validateNumerical("meanSquaredError", "predictions", predictions);
    NDValidation.validateNumerical("meanSquaredError", "weights", weights);
    return Nd4j.exec(new org.nd4j.linalg.api.ops.impl.loss.MeanSquaredErrorLoss(label, predictions, weights, org.nd4j.autodiff.loss.LossReduce.MEAN_BY_NONZERO_WEIGHT_COUNT))[0];
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
  public INDArray sigmoidCrossEntropy(INDArray label, INDArray predictionLogits, INDArray weights,
      LossReduce lossReduce, double labelSmoothing) {
    NDValidation.validateNumerical("sigmoidCrossEntropy", "label", label);
    NDValidation.validateNumerical("sigmoidCrossEntropy", "predictionLogits", predictionLogits);
    NDValidation.validateNumerical("sigmoidCrossEntropy", "weights", weights);
    return Nd4j.exec(new org.nd4j.linalg.api.ops.impl.loss.SigmoidCrossEntropyLoss(label, predictionLogits, weights, lossReduce, labelSmoothing))[0];
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
  public INDArray sigmoidCrossEntropy(INDArray label, INDArray predictionLogits, INDArray weights) {
    NDValidation.validateNumerical("sigmoidCrossEntropy", "label", label);
    NDValidation.validateNumerical("sigmoidCrossEntropy", "predictionLogits", predictionLogits);
    NDValidation.validateNumerical("sigmoidCrossEntropy", "weights", weights);
    return Nd4j.exec(new org.nd4j.linalg.api.ops.impl.loss.SigmoidCrossEntropyLoss(label, predictionLogits, weights, org.nd4j.autodiff.loss.LossReduce.MEAN_BY_NONZERO_WEIGHT_COUNT, 0.0))[0];
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
  public INDArray softmaxCrossEntropy(INDArray oneHotLabels, INDArray logitPredictions,
      INDArray weights, LossReduce lossReduce, double labelSmoothing) {
    NDValidation.validateNumerical("softmaxCrossEntropy", "oneHotLabels", oneHotLabels);
    NDValidation.validateNumerical("softmaxCrossEntropy", "logitPredictions", logitPredictions);
    NDValidation.validateNumerical("softmaxCrossEntropy", "weights", weights);
    return Nd4j.exec(new org.nd4j.linalg.api.ops.impl.loss.SoftmaxCrossEntropyLoss(oneHotLabels, logitPredictions, weights, lossReduce, labelSmoothing))[0];
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
   * @param labelSmoothing Label smoothing value. Default value: 0
   * @return output Loss variable (NUMERIC type)
   */
  public INDArray softmaxCrossEntropy(INDArray oneHotLabels, INDArray logitPredictions,
      INDArray weights, double labelSmoothing) {
    NDValidation.validateNumerical("softmaxCrossEntropy", "oneHotLabels", oneHotLabels);
    NDValidation.validateNumerical("softmaxCrossEntropy", "logitPredictions", logitPredictions);
    NDValidation.validateNumerical("softmaxCrossEntropy", "weights", weights);
    return Nd4j.exec(new org.nd4j.linalg.api.ops.impl.loss.SoftmaxCrossEntropyLoss(oneHotLabels, logitPredictions, weights, org.nd4j.autodiff.loss.LossReduce.MEAN_BY_NONZERO_WEIGHT_COUNT, labelSmoothing))[0];
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
  public INDArray sparseSoftmaxCrossEntropy(INDArray logits, INDArray labels) {
    NDValidation.validateNumerical("sparseSoftmaxCrossEntropy", "logits", logits);
    NDValidation.validateInteger("sparseSoftmaxCrossEntropy", "labels", labels);
    return Nd4j.exec(new org.nd4j.linalg.api.ops.impl.loss.SparseSoftmaxCrossEntropyLossWithLogits(logits, labels))[0];
  }

  /**
   * Weighted cross entropy loss with logits<br>
   *
   * @param targets targets array (NUMERIC type)
   * @param inputs input array (NUMERIC type)
   * @param weights eights array. May be null. If null, a weight of 1.0 is used (NUMERIC type)
   * @return output Loss variable (NUMERIC type)
   */
  public INDArray weightedCrossEntropyWithLogits(INDArray targets, INDArray inputs,
      INDArray weights) {
    NDValidation.validateNumerical("weightedCrossEntropyWithLogits", "targets", targets);
    NDValidation.validateNumerical("weightedCrossEntropyWithLogits", "inputs", inputs);
    NDValidation.validateNumerical("weightedCrossEntropyWithLogits", "weights", weights);
    return Nd4j.exec(new org.nd4j.linalg.api.ops.impl.loss.WeightedCrossEntropyLoss(targets, inputs, weights))[0];
  }
}
