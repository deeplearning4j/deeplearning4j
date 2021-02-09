/*
 *  ******************************************************************************
 *  *
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
 *  * See the NOTICE file distributed with this work for additional
 *  * information regarding copyright ownership.
 *  * Unless required by applicable law or agreed to in writing, software
 *  * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 *  * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 *  * License for the specific language governing permissions and limitations
 *  * under the License.
 *  *
 *  * SPDX-License-Identifier: Apache-2.0
 *  *****************************************************************************
 */

/**
 * Generated using ExtractFromExisting.kt
 */

package org.nd4j.codegen.ops

import org.nd4j.codegen.api.Language
import org.nd4j.codegen.api.doc.DocScope
import org.nd4j.codegen.dsl.*
import org.nd4j.codegen.api.DataType.*
import org.nd4j.codegen.api.LossReduce

fun SDLoss() =  Namespace("Loss"){

    Op("absoluteDifference") {
        javaPackage = "org.nd4j.linalg.api.ops.impl.loss"
        javaOpClass = "AbsoluteDifferenceLoss"
        Input(NUMERIC, "label") { description = "Label array" }
        Input(NUMERIC, "predictions") { description = "Predictions array" }
        Input(NUMERIC, "weights") { description = "Weights array. May be null. If null, a weight of 1.0 is used" }
        Arg(LOSS_REDUCE, "lossReduce") { description = "Reduction type for the loss. See LossReduce for more details. Default: LossReduce#MEAN_BY_NONZERO_WEIGHT_COUNT"; defaultValue = LossReduce.MEAN_BY_NONZERO_WEIGHT_COUNT}
        Output(NUMERIC, "output"){ description = "loss variable" }
        Doc(Language.ANY, DocScope.ALL){
            """
                Absolute difference loss: {@code sum_i abs( label[i] - predictions[i] )}
            """.trimIndent()
        }
    }

    Op("cosineDistance") {
        javaPackage = "org.nd4j.linalg.api.ops.impl.loss"
        javaOpClass = "CosineDistanceLoss"
        Input(NUMERIC, "label") { description = "Label array" }
        Input(NUMERIC, "predictions") { description = "Predictions array" }
        Input(NUMERIC, "weights") { description = "Weights array. May be null. If null, a weight of 1.0 is use" }
        Arg(LOSS_REDUCE, "lossReduce") { description = "Reduction type for the loss. See LossReduce for more details. Default: LossReduce#MEAN_BY_NONZERO_WEIGHT_COUNT"; defaultValue = LossReduce.MEAN_BY_NONZERO_WEIGHT_COUNT}
        Arg(INT, "dimension") { description = "Dimension to perform the cosine distance over" }
        Output(NUMERIC, "output"){ description = "Cosine distance loss " }
        Doc(Language.ANY, DocScope.ALL){
            """
                Cosine distance loss: {@code 1 - cosineSimilarity(x,y)} or {@code 1 - sum_i label[i] * prediction[i]}, which is
                equivalent to cosine distance when both the predictions and labels are normalized.<br>
                <b>Note</b>: This loss function assumes that both the predictions and labels are normalized to have unit l2 norm.
                If this is not the case, you should normalize them first by dividing by norm2(String, SDVariable, boolean, int...)
                along the cosine distance dimension (with keepDims=true).
            """.trimIndent()
        }
    }

    Op("hingeLoss") {
        javaPackage = "org.nd4j.linalg.api.ops.impl.loss"
        javaOpClass = "HingeLoss"
        Input(NUMERIC, "label") { description = "Label array. Each value should be 0.0 or 1.0 (internally -1 to 1 is used)" }
        Input(NUMERIC, "predictions") { description = "Predictions array" }
        Input(NUMERIC, "weights") { description = "Weights array. May be null. If null, a weight of 1.0 is used" }
        Arg(LOSS_REDUCE, "lossReduce") { description = "Reduction type for the loss. See LossReduce for more details. Default: LossReduce#MEAN_BY_NONZERO_WEIGHT_COUNT"; defaultValue = LossReduce.MEAN_BY_NONZERO_WEIGHT_COUNT}
        Output(NUMERIC, "output"){ description = "Loss variable" }
        Doc(Language.ANY, DocScope.ALL){
            """
                Hinge loss: a loss function used for training classifiers.
                Implements {@code L = max(0, 1 - t * predictions)} where t is the label values after internally converting to {-1,1}
                from the user specified {0,1}. Note that Labels should be provided with values {0,1}.
            """.trimIndent()
        }
    }


    Op("huberLoss") {
        javaPackage = "org.nd4j.linalg.api.ops.impl.loss"
        javaOpClass = "HuberLoss"
        Input(NUMERIC, "label") { description = "Label array" }
        Input(NUMERIC, "predictions") { description = "Predictions array" }
        Input(NUMERIC, "weights") { description = "Weights array. May be null. If null, a weight of 1.0 is used" }
        Arg(LOSS_REDUCE, "lossReduce") { description = "Reduction type for the loss. See LossReduce for more details. Default: LossReduce#MEAN_BY_NONZERO_WEIGHT_COUNT"; defaultValue = LossReduce.MEAN_BY_NONZERO_WEIGHT_COUNT}
        Arg(FLOATING_POINT, "delta") { description = "Loss function delta value" }
        Output(NUMERIC, "output"){ description = "Huber loss" }
        Doc(Language.ANY, DocScope.ALL){
            """
                Huber loss function, used for robust regression. It is similar both squared error loss and absolute difference loss,
                though is less sensitive to outliers than squared error.<br>
                Huber loss implements:
                <pre>
                {@code L = 0.5 * (label[i] - predictions[i])^2 if abs(label[i] - predictions[i]) < delta}
                {@code L = delta * abs(label[i] - predictions[i]) - 0.5 * delta^2 otherwise}
                </pre>
            """.trimIndent()
        }
    }

    Op("l2Loss") {
        javaPackage = "org.nd4j.linalg.api.ops.impl.loss"
        javaOpClass = "L2Loss"
        Input(NUMERIC, "var") { description = "Variable to calculate L2 loss of" }
        Output(NUMERIC, "output"){ description = "L2 loss" }
        Doc(Language.ANY, DocScope.ALL){
            """
                L2 loss: 1/2 * sum(x^2)
            """.trimIndent()
        }
    }

    Op("logLoss") {
        javaPackage = "org.nd4j.linalg.api.ops.impl.loss"
        javaOpClass = "LogLoss"
        Input(NUMERIC, "label") { description = "Label array" }
        Input(NUMERIC, "predictions") { description = "Predictions array" }
        Input(NUMERIC, "weights") { description = "Weights array. May be null. If null, a weight of 1.0 is used"; defaultValue = null }
        Arg(LOSS_REDUCE, "lossReduce") { description = "Reduction type for the loss. See LossReduce for more details. Default: LossReduce#MEAN_BY_NONZERO_WEIGHT_COUNT"; defaultValue = LossReduce.MEAN_BY_NONZERO_WEIGHT_COUNT}
        Arg(FLOATING_POINT, "epsilon") { description = "epsilon"; defaultValue = 0.0 }
        Output(NUMERIC, "output"){ description = "Log loss " }
        Doc(Language.ANY, DocScope.ALL){
            """
                Log loss, i.e., binary cross entropy loss, usually used for binary multi-label classification. Implements:
                {@code -1/numExamples * sum_i (labels[i] * log(predictions[i] + epsilon) + (1-labels[i]) * log(1-predictions[i] + epsilon))}
            """.trimIndent()
        }
    }

    Op("logPoisson") {
        javaPackage = "org.nd4j.linalg.api.ops.impl.loss"
        javaOpClass = "LogPoissonLoss"
        Input(NUMERIC, "label") { description = "Label array. Each value should be 0.0 or 1.0" }
        Input(NUMERIC, "predictions") { description = "Predictions array (has to be log(x) of actual predictions)" }
        Input(NUMERIC, "weights") { description = "Weights array. May be null. If null, a weight of 1.0 is used" }
        Arg(LOSS_REDUCE, "lossReduce") { description = "Reduction type for the loss. See LossReduce for more details. Default: LossReduce#MEAN_BY_NONZERO_WEIGHT_COUNT"; defaultValue = LossReduce.MEAN_BY_NONZERO_WEIGHT_COUNT}
        Arg(BOOL, "full") {description = "Boolean flag. true for logPoissonFull, false for logPoisson"}
        Output(NUMERIC, "output"){ description = "Loss variable" }
        Doc(Language.ANY, DocScope.ALL){
            """
                Log poisson loss: a loss function used for training classifiers.
                Implements {@code L = exp(c) - z * c} where c is log(predictions) and z is labels.
            """.trimIndent()
        }
    }

    // logPoissonFull is not implemented. Simply a moniker for logPoisson with full = true

    Op("meanPairwiseSquaredError") {
        javaPackage = "org.nd4j.linalg.api.ops.impl.loss"
        javaOpClass = "MeanPairwiseSquaredErrorLoss"
        Input(NUMERIC, "label") { description = "Label array" }
        Input(NUMERIC, "predictions") { description = "Predictions array" }
        Input(NUMERIC, "weights") { description = "Weights array. May be null. If null, a weight of 1.0 is used. Must be either null, scalar, or have shape [batchSize]" }
        Arg(LOSS_REDUCE, "lossReduce") { description = "Reduction type for the loss. See LossReduce for more details. Default: LossReduce#MEAN_BY_NONZERO_WEIGHT_COUNT"; defaultValue = LossReduce.MEAN_BY_NONZERO_WEIGHT_COUNT}
        Output(NUMERIC, "output"){ description = "Loss variable, scalar output" }
        Doc(Language.ANY, DocScope.ALL){
            """
                Mean pairwise squared error.<br>
                MPWSE loss calculates the difference between pairs of consecutive elements in the predictions and labels arrays.
                For example, if predictions = [p0, p1, p2] and labels are [l0, l1, l2] then MPWSE is:
                {@code [((p0-p1) - (l0-l1))^2 + ((p0-p2) - (l0-l2))^2 + ((p1-p2) - (l1-l2))^2] / 3}<br>
            """.trimIndent()
        }
    }

    Op("meanSquaredError") {
        javaPackage = "org.nd4j.linalg.api.ops.impl.loss"
        javaOpClass = "MeanSquaredErrorLoss"
        Input(NUMERIC, "label") { description = "Label array" }
        Input(NUMERIC, "predictions") { description = "Predictions array" }
        Input(NUMERIC, "weights") { description = "Weights array. May be null. If null, a weight of 1.0 is used" }
        Arg(LOSS_REDUCE, "lossReduce") { description = "Reduction type for the loss. See LossReduce for more details. Default: LossReduce#MEAN_BY_NONZERO_WEIGHT_COUNT"; defaultValue = LossReduce.MEAN_BY_NONZERO_WEIGHT_COUNT}
        Output(NUMERIC, "output"){ description = "Loss variable" }
        Doc(Language.ANY, DocScope.ALL){
            """
                Mean squared error loss function. Implements {@code (label[i] - prediction[i])^2} - i.e., squared error on a per-element basis.
                When averaged (using LossReduce#MEAN_BY_WEIGHT or LossReduce#MEAN_BY_NONZERO_WEIGHT_COUNT (the default))
                this is the mean squared error loss function.
            """.trimIndent()
        }
    }

    Op("sigmoidCrossEntropy") {
        javaPackage = "org.nd4j.linalg.api.ops.impl.loss"
        javaOpClass = "SigmoidCrossEntropyLoss"
        Input(NUMERIC, "label") { description = "Label array" }
        Input(NUMERIC, "predictionLogits") { description = "Predictions array" }
        Input(NUMERIC, "weights") { description = "Weights array. May be null. If null, a weight of 1.0 is used" }
        Arg(LOSS_REDUCE, "lossReduce") { description = "Reduction type for the loss. See LossReduce for more details. Default: LossReduce#MEAN_BY_NONZERO_WEIGHT_COUNT"; defaultValue = LossReduce.MEAN_BY_NONZERO_WEIGHT_COUNT}
        Arg(FLOATING_POINT, "labelSmoothing") { description = "Label smoothing value. Default value: 0"; defaultValue = 0.0}
        Output(NUMERIC, "output"){ description = "Loss variable" }
        Doc(Language.ANY, DocScope.ALL){
            """
                Sigmoid cross entropy: applies the sigmoid activation function on the input logits (input "pre-sigmoid preductions")
                and implements the binary cross entropy loss function. This implementation is numerically more stable than using
                standard (but separate) sigmoid activation function and log loss (binary cross entropy) loss function.<br>
                Implements:
                {@code -1/numExamples * sum_i (labels[i] * log(sigmoid(logits[i])) + (1-labels[i]) * log(1-sigmoid(logits[i])))}
                though this is done in a mathematically equivalent but more numerical stable form.<br>
                <br>
                When label smoothing is > 0, the following label smoothing is used:<br>
                <pre>
                {@code numClasses = labels.size(1);
                label = (1.0 - labelSmoothing) * label + 0.5 * labelSmoothing}
                </pre>
            """.trimIndent()
        }
    }

    Op("softmaxCrossEntropy") {
        javaPackage = "org.nd4j.linalg.api.ops.impl.loss"
        javaOpClass = "SoftmaxCrossEntropyLoss"
        Input(NUMERIC, "oneHotLabels") { description = "Label array. Should be one-hot per example and same shape as predictions (for example, [mb, nOut])" }
        Input(NUMERIC, "logitPredictions") { description = "Predictions array (pre-softmax)" }
        Input(NUMERIC, "weights") { description = "Weights array. May be null. If null, a weight of 1.0 is used" }
        Arg(LOSS_REDUCE, "lossReduce") { description = "Reduction type for the loss. See LossReduce for more details. Default: LossReduce#MEAN_BY_NONZERO_WEIGHT_COUNT"; defaultValue = LossReduce.MEAN_BY_NONZERO_WEIGHT_COUNT}
        Arg(FLOATING_POINT, "labelSmoothing") { description = "Label smoothing value. Default value: 0"; defaultValue = 0.0}
        Output(NUMERIC, "output"){ description = "Loss variable" }
        Doc(Language.ANY, DocScope.ALL){
            """
                Applies the softmax activation function to the input, then implement multi-class cross entropy:<br>
                {@code -sum_classes label[i] * log(p[c])} where {@code p = softmax(logits)}<br>
                If LossReduce#NONE is used, returned shape is [numExamples] out for [numExamples, numClasses] predicitons/labels;
                otherwise, the output is a scalar.<br>
                <p>
                When label smoothing is > 0, the following label smoothing is used:<br>
                <pre>
                {@code numClasses = labels.size(1);
                oneHotLabel = (1.0 - labelSmoothing) * oneHotLabels + labelSmoothing/numClasses}
                </pre>
            """.trimIndent()
        }
    }

    Op("sparseSoftmaxCrossEntropy") {
        javaPackage = "org.nd4j.linalg.api.ops.impl.loss"
        javaOpClass = "SparseSoftmaxCrossEntropyLossWithLogits"
        Input(NUMERIC, "logits") { description = "Logits array (\"pre-softmax activations\")" }
        Input(INT, "labels") { description = "Labels array. Must be an integer type." }
        Output(NUMERIC, "output"){ description = "Softmax cross entropy" }
        Doc(Language.ANY, DocScope.ALL){
            """
                As per softmaxCrossEntropy(String, SDVariable, SDVariable, LossReduce) but the labels variable
                is represented as an integer array instead of the equivalent one-hot array.<br>
                i.e., if logits are rank N, then labels have rank N-1
            """.trimIndent()
        }
    }


    Op("weightedCrossEntropyWithLogits") {
        javaPackage = "org.nd4j.linalg.api.ops.impl.loss"
        javaOpClass = "WeightedCrossEntropyLoss"
        Input(NUMERIC, "targets") { description = "targets array" }
        Input(NUMERIC, "inputs") { description = "input array" }
        Input(NUMERIC, "weights") { description = "eights array. May be null. If null, a weight of 1.0 is used" }
        Output(NUMERIC, "output"){ description = "Loss variable" }

        Doc(Language.ANY, DocScope.ALL){
        """
            Weighted cross entropy loss with logits
        """.trimIndent()
        }
    }
}
