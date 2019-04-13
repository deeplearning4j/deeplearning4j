package org.nd4j.autodiff.samediff.ops;

import lombok.NonNull;
import org.nd4j.autodiff.loss.LossReduce;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.base.Preconditions;
import org.nd4j.linalg.api.ops.impl.loss.LogLoss;
import org.nd4j.linalg.api.ops.impl.loss.SigmoidCrossEntropyLoss;
import org.nd4j.linalg.api.ops.impl.loss.SoftmaxCrossEntropyLoss;

/**
 * SameDiff loss functions<br>
 * Accessible via {@link SameDiff#loss()}
 *
 * @author Alex Black
 */
public class SDLoss extends SDOps {
    public SDLoss(SameDiff sameDiff) {
        super(sameDiff);
    }

    /**
     * See {@link #absoluteDifference(String, SDVariable, SDVariable, SDVariable, LossReduce)}.
     */
    public SDVariable absoluteDifference(String name, @NonNull SDVariable label, @NonNull SDVariable predictions) {
        return absoluteDifference(name, label, predictions, null, LossReduce.MEAN_BY_NONZERO_WEIGHT_COUNT);
    }

    /**
     * Absolute difference loss: {@code sum_i abs( label[i] - predictions[i] )
     *
     * @param name        Name of the operation
     * @param label       Label array
     * @param predictions Predictions array
     * @param weights     Weights array. May be null. If null, a weight of 1.0 is used
     * @param lossReduce  Reduction type for the loss. See {@link LossReduce} for more details. Default: {@link LossReduce#MEAN_BY_NONZERO_WEIGHT_COUNT}
     * @return Loss variable
     */
    public SDVariable absoluteDifference(String name, @NonNull SDVariable label, @NonNull SDVariable predictions,
                                         SDVariable weights, @NonNull LossReduce lossReduce) {
        if (weights == null)
            weights = sd.scalar(null, predictions.dataType(), 1.0);
        SDVariable result = f().lossAbsoluteDifference(label, predictions, weights, lossReduce);
        result = updateVariableNameAndReference(result, name);
        result.markAsLoss();
        return result;
    }

    /**
     * See {@link #absoluteDifference(String, SDVariable, SDVariable, SDVariable, LossReduce)}.
     */
    public SDVariable absoluteDifference(String name, @NonNull SDVariable label, @NonNull SDVariable predictions, @NonNull LossReduce lossReduce) {
        return absoluteDifference(name, label, predictions, null, lossReduce);
    }

    /**
     * See {@link #cosineDistance(String, SDVariable, SDVariable, SDVariable, LossReduce, int)}.
     */
    public SDVariable cosineDistance(String name, @NonNull SDVariable label, @NonNull SDVariable predictions, int dimension) {
        return cosineDistance(name, label, predictions, null, LossReduce.MEAN_BY_NONZERO_WEIGHT_COUNT, dimension);
    }

    /**
     * Cosine distance loss: {@code 1 - cosineSimilarity(x,y)} or {@code 1 - sum_i label[i] * prediction[i]}, which is
     * equivalent to cosine distance when both the predictions and labels are normalized.<br>
     * <b>Note</b>: This loss function assumes that both the predictions and labels are normalized to have unit l2 norm.
     * If this is not the case, you should normalize them first by dividing by {@link SameDiff#norm2(String, SDVariable, boolean, int...)}
     * along the cosine distance dimension (with keepDims=true).
     *
     * @param name        Name of the operation
     * @param label       Label array
     * @param predictions Predictions array
     * @param weights     Weights array. May be null. If null, a weight of 1.0 is used
     * @param lossReduce  Reduction type for the loss. See {@link LossReduce} for more details. Default: {@link LossReduce#MEAN_BY_NONZERO_WEIGHT_COUNT}
     * @param dimension   Dimension to perform the cosine distance over
     * @return Cosine distance loss variable
     */
    public SDVariable cosineDistance(String name, @NonNull SDVariable label, @NonNull SDVariable predictions,
                                     SDVariable weights, @NonNull LossReduce lossReduce, int dimension) {
        if (weights == null)
            weights = sd.scalar(null, predictions.dataType(), 1.0);
        SDVariable result = f().lossCosineDistance(label, predictions, weights, lossReduce, dimension);
        result = updateVariableNameAndReference(result, name);
        result.markAsLoss();
        return result;
    }

    /**
     * See {@link #cosineDistance(String, SDVariable, SDVariable, SDVariable, LossReduce, int)}.
     */
    public SDVariable cosineDistance(String name, @NonNull SDVariable label, @NonNull SDVariable predictions,
                                     @NonNull LossReduce lossReduce, int dimension) {
        return cosineDistance(name, label, predictions, null, lossReduce, dimension);
    }

    /**
     * See {@link #hingeLoss(String, SDVariable, SDVariable, SDVariable, LossReduce)}.
     */
    public SDVariable hingeLoss(String name, @NonNull SDVariable label, @NonNull SDVariable predictions) {
        return hingeLoss(name, label, predictions, null, LossReduce.MEAN_BY_NONZERO_WEIGHT_COUNT);
    }

    /**
     * Hinge loss: a loss function used for training classifiers.
     * Implements {@code L = max(0, 1 - t * predictions)} where t is the label values after internally converting to {-1,1}
     * from the user specified {0,1}. Note that Labels should be provided with values {0,1}.
     *
     * @param name        Name of the operation
     * @param label       Label array. Each value should be 0.0 or 1.0 (internally -1 to 1 is used)
     * @param predictions Predictions array
     * @param weights     Weights array. May be null. If null, a weight of 1.0 is used
     * @param lossReduce  Reduction type for the loss. See {@link LossReduce} for more details. Default: {@link LossReduce#MEAN_BY_NONZERO_WEIGHT_COUNT}
     * @return Loss variable
     */
    public SDVariable hingeLoss(String name, @NonNull SDVariable label, @NonNull SDVariable predictions,
                                SDVariable weights, @NonNull LossReduce lossReduce) {
        if (weights == null)
            weights = sd.scalar(null, predictions.dataType(), 1.0);
        SDVariable result = f().lossHinge(label, predictions, weights, lossReduce);
        result = updateVariableNameAndReference(result, name);
        result.markAsLoss();
        return result;
    }

    /**
     * See {@link #hingeLoss(String, SDVariable, SDVariable, SDVariable, LossReduce)}.
     */
    public SDVariable hingeLoss(String name, @NonNull SDVariable label, @NonNull SDVariable predictions, @NonNull LossReduce lossReduce) {
        return hingeLoss(name, label, predictions, null, lossReduce);
    }

    /**
     * See {@link #huberLoss(String, SDVariable, SDVariable, SDVariable, LossReduce, double)}.
     */
    public SDVariable huberLoss(String name, @NonNull SDVariable label, @NonNull SDVariable predictions, double delta) {
        return huberLoss(name, label, predictions, null, LossReduce.MEAN_BY_NONZERO_WEIGHT_COUNT, delta);
    }

    /**
     * Huber loss function, used for robust regression. It is similar both squared error loss and absolute difference loss,
     * though is less sensitive to outliers than squared error.<br>
     * Huber loss implements:
     * <pre>
     * {@code L = 0.5 * (label[i] - predictions[i])^2 if abs(label[i] - predictions[i]) < delta
     *  L = delta * abs(label[i] - predictions[i]) - 0.5 * delta^2 otherwise
     *     }
     * </pre>
     *
     * @param name        Name of the operation
     * @param label       Label array
     * @param predictions Predictions array
     * @param weights     Weights array. May be null. If null, a weight of 1.0 is used
     * @param lossReduce  Reduction type for the loss. See {@link LossReduce} for more details. Default: {@link LossReduce#MEAN_BY_NONZERO_WEIGHT_COUNT}
     * @param delta       Loss function delta value
     * @return Huber loss variable
     */
    public SDVariable huberLoss(String name, @NonNull SDVariable label, @NonNull SDVariable predictions,
                                SDVariable weights, @NonNull LossReduce lossReduce, double delta) {
        if (weights == null)
            weights = sd.scalar(null, predictions.dataType(), 1.0);
        SDVariable result = f().lossHuber(label, predictions, weights, lossReduce, delta);
        result = updateVariableNameAndReference(result, name);
        result.markAsLoss();
        return result;
    }

    /**
     * See {@link #huberLoss(String, SDVariable, SDVariable, SDVariable, LossReduce, double)}.
     */
    public SDVariable huberLoss(String name, @NonNull SDVariable label, @NonNull SDVariable predictions, @NonNull LossReduce lossReduce, double delta) {
        return huberLoss(name, label, predictions, null, lossReduce, delta);
    }

    /**
     * L2 loss: 1/2 * sum(x^2)
     *
     * @param var Variable to calculate L2 loss of
     * @return L2 loss
     */
    public SDVariable l2Loss(@NonNull SDVariable var) {
        return l2Loss(null, var);
    }

    /**
     * L2 loss: 1/2 * sum(x^2)
     *
     * @param name Name of the output variable
     * @param var  Variable to calculate L2 loss of
     * @return L2 loss
     */
    public SDVariable l2Loss(String name, @NonNull SDVariable var) {
        SDVariable result = f().lossL2(var);
        result = updateVariableNameAndReference(result, name);
        result.markAsLoss();
        return result;
    }

    /**
     * See {@link #logLoss(String, SDVariable, SDVariable, SDVariable, LossReduce, double)}.
     */
    public SDVariable logLoss(String name, @NonNull SDVariable label, @NonNull SDVariable predictions) {
        return logLoss(name, label, predictions, null, LossReduce.MEAN_BY_NONZERO_WEIGHT_COUNT, LogLoss.DEFAULT_EPSILON);
    }

    /**
     * Log loss, i.e., binary cross entropy loss, usually used for binary multi-label classification. Implements:
     * {@code -1/numExamples * sum_i (labels[i] * log(predictions[i] + epsilon) + (1-labels[i]) * log(1-predictions[i] + epsilon))}
     *
     * @param name        Name of the operation
     * @param label       Label array
     * @param predictions Predictions array
     * @param weights     Weights array. May be null. If null, a weight of 1.0 is used
     * @param lossReduce  Reduction type for the loss. See {@link LossReduce} for more details. Default: {@link LossReduce#MEAN_BY_NONZERO_WEIGHT_COUNT}
     * @return Log loss variable
     */
    public SDVariable logLoss(String name, @NonNull SDVariable label, @NonNull SDVariable predictions,
                              SDVariable weights, @NonNull LossReduce lossReduce, double epsilon) {
        if (weights == null)
            weights = sd.scalar(null, predictions.dataType(), 1.0);
        SDVariable result = f().lossLog(label, predictions, weights, lossReduce, epsilon);
        result = updateVariableNameAndReference(result, name);
        result.markAsLoss();
        return result;
    }

    /**
     * See {@link #logLoss(String, SDVariable, SDVariable, SDVariable, LossReduce, double)}.
     */
    public SDVariable logLoss(String name, @NonNull SDVariable label, @NonNull SDVariable predictions, @NonNull LossReduce lossReduce) {
        return logLoss(name, label, predictions, null, lossReduce, LogLoss.DEFAULT_EPSILON);
    }

    /**
     * See {@link #logPoisson(String, SDVariable, SDVariable, SDVariable, LossReduce)}.
     */
    public SDVariable logPoisson(String name, @NonNull SDVariable label, @NonNull SDVariable predictions) {
        return logPoisson(name, label, predictions, null, LossReduce.MEAN_BY_NONZERO_WEIGHT_COUNT);
    }

    /**
     * Log poisson loss: a loss function used for training classifiers.
     * Implements {@code L = exp(c) - z * c} where c is log(predictions) and z is labels.
     *
     * @param name        Name of the operation
     * @param label       Label array. Each value should be 0.0 or 1.0
     * @param predictions Predictions array (has to be log(x) of actual predictions)
     * @param weights     Weights array. May be null. If null, a weight of 1.0 is used
     * @param lossReduce  Reduction type for the loss. See {@link LossReduce} for more details. Default: {@link LossReduce#MEAN_BY_NONZERO_WEIGHT_COUNT}
     * @return Loss variable
     */
    public SDVariable logPoisson(String name, @NonNull SDVariable label, @NonNull SDVariable predictions,
                                 SDVariable weights, @NonNull LossReduce lossReduce) {
        if (weights == null)
            weights = sd.scalar(null, predictions.dataType(), 1.0);
        SDVariable result = f().lossLogPoisson(label, predictions, weights, lossReduce);
        result = updateVariableNameAndReference(result, name);
        result.markAsLoss();
        return result;
    }

    /**
     * See {@link #logPoisson(String, SDVariable, SDVariable, SDVariable, LossReduce)}.
     */
    public SDVariable logPoisson(String name, @NonNull SDVariable label, @NonNull SDVariable predictions, @NonNull LossReduce lossReduce) {
        return logPoisson(name, label, predictions, null, lossReduce);
    }

    /**
     * See {@link #logPoissonFull(String, SDVariable, SDVariable, SDVariable, LossReduce)}.
     */
    public SDVariable logPoissonFull(String name, @NonNull SDVariable label, @NonNull SDVariable predictions) {
        return logPoissonFull(name, label, predictions, null, LossReduce.MEAN_BY_NONZERO_WEIGHT_COUNT);
    }

    /**
     * Log poisson loss: a loss function used for training classifiers.
     * Implements {@code L = exp(c) - z * c + z * log(z) - z + 0.5 * log(2 * pi * z)}
     * where c is log(predictions) and z is labels.
     *
     * @param name        Name of the operation
     * @param label       Label array. Each value should be 0.0 or 1.0
     * @param predictions Predictions array (has to be log(x) of actual predictions)
     * @param weights     Weights array. May be null. If null, a weight of 1.0 is used
     * @param lossReduce  Reduction type for the loss. See {@link LossReduce} for more details. Default: {@link LossReduce#MEAN_BY_NONZERO_WEIGHT_COUNT}
     * @return Loss variable
     */
    public SDVariable logPoissonFull(String name, @NonNull SDVariable label, @NonNull SDVariable predictions,
                                     SDVariable weights, @NonNull LossReduce lossReduce) {
        if (weights == null)
            weights = sd.scalar(null, predictions.dataType(), 1.0);
        SDVariable result = f().lossLogPoissonFull(label, predictions, weights, lossReduce);
        result = updateVariableNameAndReference(result, name);
        result.markAsLoss();
        return result;
    }

    /**
     * See {@link #logPoissonFull(String, SDVariable, SDVariable, SDVariable, LossReduce)}.
     */
    public SDVariable logPoissonFull(String name, @NonNull SDVariable label, @NonNull SDVariable predictions, @NonNull LossReduce lossReduce) {
        return logPoissonFull(name, label, predictions, null, lossReduce);
    }

    /**
     * See {@link #meanPairwiseSquaredError(String, SDVariable, SDVariable, SDVariable, LossReduce)}.
     */
    public SDVariable meanPairwiseSquaredError(String name, @NonNull SDVariable label, @NonNull SDVariable predictions, @NonNull LossReduce lossReduce) {
        return meanPairwiseSquaredError(name, label, predictions, null, lossReduce);
    }

    /**
     * Mean pairwise squared error.<br>
     * MPWSE loss calculates the difference between pairs of consecutive elements in the predictions and labels arrays.
     * For example, if predictions = [p0, p1, p2] and labels are [l0, l1, l2] then MPWSE is:
     * {@code [((p0-p1) - (l0-l1))^2 + ((p0-p2) - (l0-l2))^2 + ((p1-p2) - (l1-l2))^2] / 3}<br>
     *
     * @param name        Name of the operation
     * @param label       Label array
     * @param predictions Predictions array
     * @param weights     Weights array. May be null. If null, a weight of 1.0 is used. Must be either null, scalar, or have shape [batchSize]
     * @return Loss variable, scalar output
     */
    public SDVariable meanPairwiseSquaredError(String name, @NonNull SDVariable label, @NonNull SDVariable predictions, SDVariable weights, @NonNull LossReduce lossReduce) {
        if (weights == null)
            weights = sd.scalar(null, predictions.dataType(), 1.0);
        SDVariable result = f().lossMeanPairwiseSquaredError(label, predictions, weights, lossReduce);
        result = updateVariableNameAndReference(result, name);
        result.markAsLoss();
        return result;
    }

    /**
     * See {@link #meanSquaredError(String, SDVariable, SDVariable, SDVariable, LossReduce)}.
     */
    public SDVariable meanSquaredError(String name, @NonNull SDVariable label, @NonNull SDVariable predictions) {
        return meanSquaredError(name, label, predictions, null, LossReduce.MEAN_BY_NONZERO_WEIGHT_COUNT);
    }

    /**
     * Mean squared error loss function. Implements {@code (label[i] - prediction[i])^2} - i.e., squared error on a per-element basis.
     * When averaged (using {@link LossReduce#MEAN_BY_WEIGHT} or {@link LossReduce#MEAN_BY_NONZERO_WEIGHT_COUNT} (the default))
     * this is the mean squared error loss function.
     *
     * @param name        Name of the operation
     * @param label       Label array
     * @param predictions Predictions array
     * @param weights     Weights array. May be null. If null, a weight of 1.0 is used
     * @param lossReduce  Reduction type for the loss. See {@link LossReduce} for more details. Default: {@link LossReduce#MEAN_BY_NONZERO_WEIGHT_COUNT}
     * @return Loss variable
     */
    public SDVariable meanSquaredError(String name, @NonNull SDVariable label, @NonNull SDVariable predictions,
                                       SDVariable weights, @NonNull LossReduce lossReduce) {
        if (weights == null)
            weights = sd.scalar(null, predictions.dataType(), 1.0);
        SDVariable result = f().lossMeanSquaredError(label, predictions, weights, lossReduce);
        result = updateVariableNameAndReference(result, name);
        result.markAsLoss();
        return result;
    }

    /**
     * See {@link #meanSquaredError(String, SDVariable, SDVariable, SDVariable, LossReduce)}.
     */
    public SDVariable meanSquaredError(String name, @NonNull SDVariable label, @NonNull SDVariable predictions, @NonNull LossReduce lossReduce) {
        return meanSquaredError(name, label, predictions, null, lossReduce);
    }

    /**
     * See {@link #sigmoidCrossEntropy(String, SDVariable, SDVariable, SDVariable, LossReduce, double)}.
     */
    public SDVariable sigmoidCrossEntropy(String name, @NonNull SDVariable label, @NonNull SDVariable predictions) {
        return sigmoidCrossEntropy(name, label, predictions, null, LossReduce.MEAN_BY_NONZERO_WEIGHT_COUNT, SigmoidCrossEntropyLoss.DEFAULT_LABEL_SMOOTHING);
    }

    /**
     * Sigmoid cross entropy: applies the sigmoid activation function on the input logits (input "pre-sigmoid preductions")
     * and implements the binary cross entropy loss function. This implementation is numerically more stable than using
     * standard (but separate) sigmoid activation function and log loss (binary cross entropy) loss function.<br>
     * Implements:
     * {@code -1/numExamples * sum_i (labels[i] * log(sigmoid(logits[i])) + (1-labels[i]) * log(1-sigmoid(logits[i])))}
     * though this is done in a mathematically equivalent but more numerical stable form.<br>
     * <br>
     * When label smoothing is > 0, the following label smoothing is used:<br>
     * <pre>
     * {@code numClasses = labels.size(1);
     * label = (1.0 - labelSmoothing) * label + 0.5 * labelSmoothing}
     * </pre>
     *
     * @param name             Name of the operation
     * @param label            Label array
     * @param predictionLogits Predictions array
     * @param weights          Weights array. May be null. If null, a weight of 1.0 is used
     * @param lossReduce       Reduction type for the loss. See {@link LossReduce} for more details. Default: {@link LossReduce#MEAN_BY_NONZERO_WEIGHT_COUNT}
     * @return Loss variable
     */
    public SDVariable sigmoidCrossEntropy(String name, @NonNull SDVariable label, @NonNull SDVariable predictionLogits,
                                          SDVariable weights, @NonNull LossReduce lossReduce, double labelSmoothing) {
        if (weights == null)
            weights = sd.scalar(null, predictionLogits.dataType(), 1.0);
        SDVariable result = f().lossSigmoidCrossEntropy(label, predictionLogits, weights, lossReduce, labelSmoothing);
        result = updateVariableNameAndReference(result, name);
        result.markAsLoss();
        return result;
    }

    /**
     * See {@link #sigmoidCrossEntropy(String, SDVariable, SDVariable, SDVariable, LossReduce, double)}.
     */
    public SDVariable sigmoidCrossEntropy(String name, @NonNull SDVariable label, @NonNull SDVariable predictions, @NonNull LossReduce lossReduce) {
        return sigmoidCrossEntropy(name, label, predictions, null, lossReduce, SigmoidCrossEntropyLoss.DEFAULT_LABEL_SMOOTHING);
    }

    /**
     * See {@link #softmaxCrossEntropy(String, SDVariable, SDVariable, SDVariable, LossReduce, double)}.
     */
    public SDVariable softmaxCrossEntropy(String name, @NonNull SDVariable label, @NonNull SDVariable predictions) {
        return softmaxCrossEntropy(name, label, predictions, null, LossReduce.MEAN_BY_NONZERO_WEIGHT_COUNT, SoftmaxCrossEntropyLoss.DEFAULT_LABEL_SMOOTHING);
    }

    /**
     * Applies the softmax activation function to the input, then implement multi-class cross entropy:<br>
     * {@code -sum_classes label[i] * log(p[c])} where {@code p = softmax(logits)}<br>
     * If {@link LossReduce#NONE} is used, returned shape is [numExamples] out for [numExamples, numClasses] predicitons/labels;
     * otherwise, the output is a scalar.<br>
     * <p>
     * When label smoothing is > 0, the following label smoothing is used:<br>
     * <pre>
     * {@code numClasses = labels.size(1);
     * oneHotLabel = (1.0 - labelSmoothing) * oneHotLabels + labelSmoothing/numClasses}
     * </pre>
     *
     * @param name             Name of the operation
     * @param oneHotLabels     Label array. Should be one-hot per example and same shape as predictions (for example, [mb, nOut])
     * @param logitPredictions Predictions array (pre-softmax)
     * @param weights          Weights array. May be null. If null, a weight of 1.0 is used
     * @param lossReduce       Reduction type for the loss. See {@link LossReduce} for more details. Default: {@link LossReduce#MEAN_BY_NONZERO_WEIGHT_COUNT}
     * @param labelSmoothing   Label smoothing value. Default value: 0
     * @return Loss variable
     */
    public SDVariable softmaxCrossEntropy(String name, @NonNull SDVariable oneHotLabels, @NonNull SDVariable logitPredictions,
                                          SDVariable weights, @NonNull LossReduce lossReduce, double labelSmoothing) {
        if (weights == null)
            weights = sd.scalar(null, logitPredictions.dataType(), 1.0);
        SDVariable result = f().lossSoftmaxCrossEntropy(oneHotLabels, logitPredictions, weights, lossReduce, labelSmoothing);
        result = updateVariableNameAndReference(result, name);
        result.markAsLoss();
        return result;
    }

    /**
     * See {@link #softmaxCrossEntropy(String, SDVariable, SDVariable, SDVariable, LossReduce, double)}.
     */
    public SDVariable softmaxCrossEntropy(String name, @NonNull SDVariable label, @NonNull SDVariable predictions, @NonNull LossReduce lossReduce) {
        return softmaxCrossEntropy(name, label, predictions, null, lossReduce, SoftmaxCrossEntropyLoss.DEFAULT_LABEL_SMOOTHING);
    }

    /**
     * See {@link #sparseSoftmaxCrossEntropy(String, SDVariable, SDVariable)}
     */
    public SDVariable sparseSoftmaxCrossEntropy(@NonNull SDVariable logits, @NonNull SDVariable labels) {
        return sparseSoftmaxCrossEntropy(null, logits, labels);
    }

    /**
     * As per {@link #softmaxCrossEntropy(String, SDVariable, SDVariable, LossReduce)} but the labels variable
     * is represented as an integer array instead of the equivalent one-hot array.<br>
     * i.e., if logits are rank N, then labels have rank N-1
     *
     * @param name   Name of the output variable. May be null
     * @param logits Logits array ("pre-softmax activations")
     * @param labels Labels array. Must be an integer type.
     * @return Softmax cross entropy
     */
    public SDVariable sparseSoftmaxCrossEntropy(String name, @NonNull SDVariable logits, @NonNull SDVariable labels) {
        Preconditions.checkState(labels.dataType().isIntType(), "Labels variable must be an integer type: got %s", logits);
        SDVariable result = f().lossSparseSoftmaxCrossEntropy(logits, labels);
        result = updateVariableNameAndReference(result, name);
        result.markAsLoss();
        return result;
    }

    /**
     * TODO
     *
     * @param targets
     * @param inputs
     * @param weights
     * @return
     */
    public SDVariable weightedCrossEntropyWithLogits(SDVariable targets, SDVariable inputs,
                                                     SDVariable weights) {
        return weightedCrossEntropyWithLogits(null, targets, inputs, weights);
    }

    /**
     * TODO
     *
     * @param name
     * @param targets
     * @param inputs
     * @param weights
     * @return
     */
    public SDVariable weightedCrossEntropyWithLogits(String name, SDVariable targets, SDVariable inputs,
                                                     SDVariable weights) {
        SDVariable result = f().weightedCrossEntropyWithLogits(targets, inputs, weights);
        result = updateVariableNameAndReference(result, name);
        result.markAsLoss();
        return result;
    }


}
