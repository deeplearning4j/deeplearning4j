/*******************************************************************************
 * Copyright (c) 2015-2019 Skymind, Inc.
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

package org.nd4j.autodiff.samediff.ops;

import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.base.Preconditions;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ops.impl.reduce3.CosineSimilarity;
import org.nd4j.linalg.api.ops.impl.reduce3.EuclideanDistance;
import org.nd4j.linalg.api.ops.impl.reduce3.ManhattanDistance;
import org.nd4j.linalg.api.ops.impl.shape.ConfusionMatrix;
import org.nd4j.linalg.api.ops.impl.shape.Eye;
import org.nd4j.linalg.indexing.conditions.Condition;

import java.util.List;

import static org.nd4j.autodiff.samediff.ops.SDValidation.*;

/**
 * SameDiff math operations<br>
 * Accessible via {@link SameDiff#math()}
 *
 * @author Alex Black
 */
public class SDMath extends SDOps {
    public SDMath(SameDiff sameDiff) {
        super(sameDiff);
    }

    /**
     * Elementwise absolute value operation: out = abs(x)
     *
     * @param x Input variable
     * @return Output variable
     */
    public SDVariable abs(SDVariable x) {
        return abs(null, x);
    }

    /**
     * Elementwise absolute value operation: out = abs(x)
     *
     * @param name Name of the output variable
     * @param x    Input variable
     * @return Output variable
     */
    public SDVariable abs(String name, SDVariable x) {
        validateNumerical("abs", x);
        SDVariable result = f().abs(x);
        return updateVariableNameAndReference(result, name);
    }

    /**
     * Elementwise acos (arccosine, inverse cosine) operation: out = arccos(x)
     *
     * @param x Input variable
     * @return Output variable
     */
    public SDVariable acos(SDVariable x) {
        return acos(null, x);
    }

    /**
     * Elementwise acos (arccosine, inverse cosine) operation: out = arccos(x)
     *
     * @param name Output variable name
     * @param x    Input variable
     * @return Output variable
     */
    public SDVariable acos(String name, SDVariable x) {
        validateNumerical("acos", x);
        SDVariable result = f().acos(x);
        return updateVariableNameAndReference(result, name);
    }

    /**
     * Elementwise acosh (inverse hyperbolic cosine) function: out = acosh(x)
     *
     * @param x Input variable
     * @return Output variable
     */
    public SDVariable acosh(SDVariable x) {
        return acosh(null, x);
    }

    /**
     * Elementwise acosh (inverse hyperbolic cosine) function: out = acosh(x)
     *
     * @param name Output variable name
     * @param x    Input variable
     * @return Output variable
     */
    public SDVariable acosh(String name, SDVariable x) {
        validateNumerical("acosh", x);
        SDVariable result = f().acosh(x);
        return updateVariableNameAndReference(result, name);
    }

    /**
     * Absolute max array reduction operation, optionally along specified dimensions: out = max(abs(x))
     *
     * @param in         Input variable
     * @param dimensions Dimensions to reduce over. If dimensions are not specified, full array reduction is performed
     * @return Reduced array of rank (input rank - num dimensions)
     */
    public SDVariable amax(SDVariable in, int... dimensions) {
        return amax(null, in, dimensions);
    }

    /**
     * Absolute max array reduction operation, optionally along specified dimensions: out = max(abs(x))
     *
     * @param name       Name of the output variable
     * @param in         Input variable
     * @param dimensions Dimensions to reduce over. If dimensions are not specified, full array reduction is performed
     * @return Reduced array of rank (input rank - num dimensions)
     */
    public SDVariable amax(String name, SDVariable in, int... dimensions) {
        validateNumerical("amax", in);
        SDVariable ret = f().amax(in, dimensions);
        return updateVariableNameAndReference(ret, name);
    }

    /**
     * Absolute mean array reduction operation, optionally along specified dimensions: out = mean(abs(x))
     *
     * @param in         Input variable
     * @param dimensions Dimensions to reduce over. If dimensions are not specified, full array reduction is performed
     * @return Reduced array of rank (input rank - num dimensions)
     */
    public SDVariable amean(SDVariable in, int... dimensions) {
        validateNumerical("amean", in);
        return amean(null, in, dimensions);
    }

    /**
     * Absolute mean array reduction operation, optionally along specified dimensions: out = mean(abs(x))
     *
     * @param name       Name of the output variable
     * @param in         Input variable
     * @param dimensions Dimensions to reduce over. If dimensions are not specified, full array reduction is performed
     * @return Reduced array of rank (input rank - num dimensions)
     */
    public SDVariable amean(String name, SDVariable in, int... dimensions) {
        validateNumerical("amean", in);
        SDVariable ret = f().amean(in, dimensions);
        return updateVariableNameAndReference(ret, name);
    }

    /**
     * Absolute min array reduction operation, optionally along specified dimensions: out = min(abs(x))
     *
     * @param in         Input variable
     * @param dimensions Dimensions to reduce over. If dimensions are not specified, full array reduction is performed
     * @return Reduced array of rank (input rank - num dimensions)
     */
    public SDVariable amin(SDVariable in, int... dimensions) {
        return amin(null, in, dimensions);
    }

    /**
     * Absolute min array reduction operation, optionally along specified dimensions: out = min(abs(x))
     *
     * @param name       Name of the output variable
     * @param in         Input variable
     * @param dimensions Dimensions to reduce over. If dimensions are not specified, full array reduction is performed
     * @return Reduced array of rank (input rank - num dimensions)
     */
    public SDVariable amin(String name, SDVariable in, int... dimensions) {
        validateNumerical("amin", in);
        SDVariable ret = f().amin(in, dimensions);
        return updateVariableNameAndReference(ret, name);
    }

    /**
     * Boolean AND operation: elementwise (x != 0) && (y != 0)<br>
     * If x and y arrays have equal shape, the output shape is the same as these inputs.<br>
     * Note: supports broadcasting if x and y have different shapes and are broadcastable.<br>
     * Returns an array with values 1 where condition is satisfied, or value 0 otherwise.
     *
     * @param x Input 1
     * @param y Input 2
     * @return Output SDVariable with values 0 and 1 based on where the condition is satisfied
     */
    public SDVariable and(SDVariable x, SDVariable y) {
        return and(null, x, y);
    }

    /**
     * Boolean AND operation: elementwise (x != 0) && (y != 0)<br>
     * If x and y arrays have equal shape, the output shape is the same as these inputs.<br>
     * Note: supports broadcasting if x and y have different shapes and are broadcastable.<br>
     * Returns an array with values 1 where condition is satisfied, or value 0 otherwise.
     *
     * @param name Name of the output variable
     * @param x    Input 1
     * @param y    Input 2
     * @return Output SDVariable with values 0 and 1 based on where the condition is satisfied
     */
    public SDVariable and(String name, SDVariable x, SDVariable y) {
        validateBool("boolean and", x, y);
        SDVariable result = f().and(x, y);
        return updateVariableNameAndReference(result, name);
    }

    /**
     * Elementwise asin (arcsin, inverse sine) operation: out = arcsin(x)
     *
     * @param x Input variable
     * @return Output variable
     */
    public SDVariable asin(SDVariable x) {
        return asin(null, x);
    }

    /**
     * Elementwise asin (arcsin, inverse sine) operation: out = arcsin(x)
     *
     * @param name Output variable name
     * @param x    Input variable
     * @return Output variable
     */
    public SDVariable asin(String name, SDVariable x) {
        validateNumerical("asin", x);
        SDVariable result = f().asin(x);
        return updateVariableNameAndReference(result, name);
    }

    /**
     * Elementwise asinh (inverse hyperbolic sine) function: out = asinh(x)
     *
     * @param x Input variable
     * @return Output variable
     */
    public SDVariable asinh(SDVariable x) {
        return asinh(null, x);
    }

    /**
     * Elementwise asinh (inverse hyperbolic sine) function: out = asinh(x)
     *
     * @param name Output variable name
     * @param x    Input variable
     * @return Output variable
     */
    public SDVariable asinh(String name, SDVariable x) {
        validateNumerical("asinh", x);
        SDVariable result = f().asinh(x);
        return updateVariableNameAndReference(result, name);
    }

    /**
     * Absolute sum array reduction operation, optionally along specified dimensions: out = sum(abs(x))
     *
     * @param in         Input variable
     * @param dimensions Dimensions to reduce over. If dimensions are not specified, full array reduction is performed
     * @return Reduced array of rank (input rank - num dimensions)
     */
    public SDVariable asum(SDVariable in, int... dimensions) {
        return asum(null, in, dimensions);
    }

    /**
     * Absolute sum array reduction operation, optionally along specified dimensions: out = sum(abs(x))
     *
     * @param name       Name of the output variable
     * @param in         Input variable
     * @param dimensions Dimensions to reduce over. If dimensions are not specified, full array reduction is performed
     * @return Reduced array of rank (input rank - num dimensions)
     */
    public SDVariable asum(String name, SDVariable in, int... dimensions) {
        validateNumerical("asum", in);
        SDVariable ret = f().asum(in, dimensions);
        return updateVariableNameAndReference(ret, name);
    }

    /**
     * Elementwise atan (arctangent, inverse tangent) operation: out = arctangent(x)
     *
     * @param x Input variable
     * @return Output variable
     */
    public SDVariable atan(SDVariable x) {
        return atan(null, x);
    }

    /**
     * Elementwise atan (arctangent, inverse tangent) operation: out = arctangent(x)
     *
     * @param name Output variable name
     * @param x    Input variable
     * @return Output variable
     */
    public SDVariable atan(String name, SDVariable x) {
        validateNumerical("atan", x);
        SDVariable result = f().atan(x);
        return updateVariableNameAndReference(result, name);
    }

    /**
     * Elementwise atan (arctangent, inverse tangent) operation: out = atan2(x,y).
     * Similar to atan(y/x) but sigts of x and y are used to determine the location of the result
     *
     * @param y Input Y variable
     * @param x Input X variable
     * @return Output variable
     */
    public SDVariable atan2(SDVariable y, SDVariable x) {
        return atan2(null, y, x);
    }

    /**
     * Elementwise atan (arctangent, inverse tangent) operation: out = atan2(x,y).
     * Similar to atan(y/x) but sigts of x and y are used to determine the location of the result
     *
     * @param name Name of the output variable
     * @param y    Input Y variable
     * @param x    Input X variable
     * @return Output variable
     */
    public SDVariable atan2(String name, SDVariable y, SDVariable x) {
        validateNumerical("atan2", y, x);
        SDVariable ret = f().atan2(y, x);
        return updateVariableNameAndReference(ret, name);
    }

    /**
     * Elementwise atanh (inverse hyperbolic tangent) function: out = atanh(x)
     *
     * @param x Input variable
     * @return Output variable
     */
    public SDVariable atanh(SDVariable x) {
        return atanh(null, x);
    }

    /**
     * Elementwise atanh (inverse hyperbolic tangent) function: out = atanh(x)
     *
     * @param name Output variable name
     * @param x    Input variable
     * @return Output variable
     */
    public SDVariable atanh(String name, SDVariable x) {
        validateNumerical("atanh", x);
        SDVariable result = f().atanh(x);
        return updateVariableNameAndReference(result, name);
    }

    /**
     * Element-wise ceiling function: out = ceil(x).
     * Rounds each value up to the nearest integer value (if not already an integer)
     *
     * @param x Input variable
     * @return Output variable
     */
    public SDVariable ceil(SDVariable x) {
        return ceil(null, x);
    }

    /**
     * Element-wise ceiling function: out = ceil(x).
     * Rounds each value up to the nearest integer value (if not already an integer)
     *
     * @param name Name of the output variable
     * @param x    Input variable
     * @return Output variable
     */
    public SDVariable ceil(String name, SDVariable x) {
        validateFloatingPoint("ceil", x);
        SDVariable ret = f().ceil(x);
        return updateVariableNameAndReference(ret, name);
    }

    /**
     * Clipping by L2 norm<br>
     * if l2Norm(x) < clipValue, then input is returned unmodifed<br>
     * Otherwise, out[i] = in[i] * clipValue / l2Norm(in)
     *
     * @param x         Input variable
     * @param clipValue Clipping value (maximum l2 norm)
     * @return Output variable
     */
    public SDVariable clipByNorm(SDVariable x, double clipValue) {
        return clipByNorm(null, x, clipValue);
    }

    /**
     * Clipping by L2 norm<br>
     * if l2Norm(x) < clipValue, then input is returned unmodifed<br>
     * Otherwise, out[i] = in[i] * clipValue / l2Norm(in)
     *
     * @param name      Name of the output variable
     * @param x         Input variable
     * @param clipValue Clipping value (maximum l2 norm)
     * @return Output variable
     */
    public SDVariable clipByNorm(String name, SDVariable x, double clipValue) {
        validateFloatingPoint("clip by norm", x);
        SDVariable ret = f().clipByNorm(x, clipValue);
        return updateVariableNameAndReference(ret, name);
    }

    /**
     * Clipping by L2 norm, optionally along dimension(s)<br>
     * if l2Norm(x,dimension) < clipValue, then input is returned unmodifed<br>
     * Otherwise, out[i] = in[i] * clipValue / l2Norm(in, dimensions) where each value is clipped according
     * to the corresponding l2Norm along the specified dimensions
     *
     * @param x          Input variable
     * @param clipValue  Clipping value (maximum l2 norm)
     * @param dimensions If not specified, all dimensions are used
     * @return Output variable
     */
    public SDVariable clipByNorm(SDVariable x, double clipValue, int... dimensions) {
        return clipByNorm(null, x, clipValue, dimensions);
    }

    /**
     * Clipping by L2 norm, optionally along dimension(s)<br>
     * if l2Norm(x,dimension) < clipValue, then input is returned unmodifed<br>
     * Otherwise, out[i] = in[i] * clipValue / l2Norm(in, dimensions) where each value is clipped according
     * to the corresponding l2Norm along the specified dimensions
     *
     * @param name       Output variable name
     * @param x          Input variable
     * @param clipValue  Clipping value (maximum l2 norm)
     * @param dimensions If not specified, all dimensions are used
     * @return Output variable
     */
    public SDVariable clipByNorm(String name, SDVariable x, double clipValue, int... dimensions) {
        validateFloatingPoint("clip by norm", x);
        SDVariable ret = f().clipByNorm(x, clipValue, dimensions);
        return updateVariableNameAndReference(ret, name);
    }

    /**
     * Element-wise clipping function:<br>
     * out[i] = in[i] if in[i] >= clipValueMin and in[i] <= clipValueMax<br>
     * out[i] = clipValueMin if in[i] < clipValueMin<br>
     * out[i] = clipValueMax if in[i] > clipValueMax<br>
     *
     * @param x            Input variable
     * @param clipValueMin Minimum value for clipping
     * @param clipValueMax Maximum value for clipping
     * @return Output variable
     */
    public SDVariable clipByValue(SDVariable x, double clipValueMin, double clipValueMax) {
        return clipByValue(null, x, clipValueMin, clipValueMax);
    }

    /**
     * Element-wise clipping function:<br>
     * out[i] = in[i] if in[i] >= clipValueMin and in[i] <= clipValueMax<br>
     * out[i] = clipValueMin if in[i] < clipValueMin<br>
     * out[i] = clipValueMax if in[i] > clipValueMax<br>
     *
     * @param name         Name of the output variable
     * @param x            Input variable
     * @param clipValueMin Minimum value for clipping
     * @param clipValueMax Maximum value for clipping
     * @return Output variable
     */
    public SDVariable clipByValue(String name, SDVariable x, double clipValueMin, double clipValueMax) {
        validateNumerical("clip by value", x);
        SDVariable ret = f().clipByValue(x, clipValueMin, clipValueMax);
        return updateVariableNameAndReference(ret, name);
    }

    /**
     * @see #confusionMatrix(String, SDVariable, SDVariable)
     */
    public SDVariable confusionMatrix(SDVariable labels, SDVariable predictions) {
        return confusionMatrix((String) null, labels, predictions);
    }

    public SDVariable confusionMatrix(String name, SDVariable labels, SDVariable pred) {
        return confusionMatrix(name, labels, pred, ConfusionMatrix.DEFAULT_DTYPE);
    }

    /**
     * Compute the 2d confusion matrix of size [numClasses, numClasses] from a pair of labels and predictions, both of
     * which are represented as integer values. This version assumes the number of classes is 1 + max(max(labels), max(pred))<br>
     * For example, if labels = [0, 1, 1] and predicted = [0, 2, 1] then output is:<br>
     * [1, 0, 0]<br>
     * [0, 1, 1]<br>
     * [0, 0, 0]<br>
     *
     * @param name   Name of the output variable
     * @param labels Labels - 1D array of integer values representing label values
     * @param pred   Predictions - 1D array of integer values representing predictions. Same length as labels
     * @return Output variable (2D, shape [numClasses, numClasses})
     */
    public SDVariable confusionMatrix(String name, SDVariable labels, SDVariable pred, DataType dataType) {
        validateInteger("confusionMatrix", "labels", labels);
        validateInteger("confusionMatrix", "prediction", pred);
        SDVariable result = f().confusionMatrix(labels, pred, dataType);
        return updateVariableNameAndReference(result, name);
    }

    /**
     * @see #confusionMatrix(String, SDVariable, SDVariable, Integer)
     */
    public SDVariable confusionMatrix(SDVariable labels, SDVariable pred, Integer numClasses) {
        return confusionMatrix(null, labels, pred, numClasses);
    }

    /**
     * Compute the 2d confusion matrix of size [numClasses, numClasses] from a pair of labels and predictions, both of
     * which are represented as integer values.<br>
     * For example, if labels = [0, 1, 1], predicted = [0, 2, 1], and numClasses=4 then output is:<br>
     * [1, 0, 0, 0]<br>
     * [0, 1, 1, 0]<br>
     * [0, 0, 0, 0]<br>
     * [0, 0, 0, 0]<br>
     *
     * @param name       Name of the output variable
     * @param labels     Labels - 1D array of integer values representing label values
     * @param pred       Predictions - 1D array of integer values representing predictions. Same length as labels
     * @param numClasses Number of classes
     * @return Output variable (2D, shape [numClasses, numClasses})
     */
    public SDVariable confusionMatrix(String name, SDVariable labels, SDVariable pred, Integer numClasses) {
        validateInteger("confusionMatrix", "labels", labels);
        validateInteger("confusionMatrix", "prediction", pred);
        SDVariable result = f().confusionMatrix(labels, pred, numClasses);
        return updateVariableNameAndReference(result, name);
    }

    /**
     * @see #confusionMatrix(String, SDVariable, SDVariable, SDVariable)
     */
    public SDVariable confusionMatrix(SDVariable labels, SDVariable pred, SDVariable weights) {
        return confusionMatrix(null, labels, pred, weights);
    }

    /**
     * Compute the 2d confusion matrix of size [numClasses, numClasses] from a pair of labels and predictions, both of
     * which are represented as integer values. This version assumes the number of classes is 1 + max(max(labels), max(pred))<br>
     * For example, if labels = [0, 1, 1], predicted = [0, 2, 1] and weights = [1, 2, 3]
     * [1, 0, 0]<br>
     * [0, 3, 2]<br>
     * [0, 0, 0]<br>
     *
     * @param name    Name of the output variable
     * @param labels  Labels - 1D array of integer values representing label values
     * @param pred    Predictions - 1D array of integer values representing predictions. Same length as labels
     * @param weights Weights - 1D array of values (may be real/decimal) representing the weight/contribution of
     *                each prediction. Must be same length as both labels and predictions arrays
     * @return Output variable (2D, shape [numClasses, numClasses})
     */
    public SDVariable confusionMatrix(String name, SDVariable labels, SDVariable pred, SDVariable weights) {
        validateInteger("confusionMatrix", "labels", labels);
        validateInteger("confusionMatrix", "prediction", pred);
        validateNumerical("confusionMatrix", "weights", weights);
        SDVariable result = f().confusionMatrix(labels, pred, weights);
        return updateVariableNameAndReference(result, name);
    }

    /**
     * @see #confusionMatrix(String, SDVariable, SDVariable, Integer, SDVariable)
     */
    public SDVariable confusionMatrix(SDVariable labels, SDVariable pred, Integer numClasses, SDVariable weights) {
        return confusionMatrix(null, labels, pred, numClasses, weights);
    }

    /**
     * Compute the 2d confusion matrix of size [numClasses, numClasses] from a pair of labels and predictions, both of
     * which are represented as integer values.<br>
     * For example, if labels = [0, 1, 1], predicted = [0, 2, 1], numClasses = 4, and weights = [1, 2, 3]
     * [1, 0, 0, 0]<br>
     * [0, 3, 2, 0]<br>
     * [0, 0, 0, 0]<br>
     * [0, 0, 0, 0]<br>
     *
     * @param name    Name of the output variable
     * @param labels  Labels - 1D array of integer values representing label values
     * @param pred    Predictions - 1D array of integer values representing predictions. Same length as labels
     * @param weights Weights - 1D array of values (may be real/decimal) representing the weight/contribution of
     *                each prediction. Must be same length as both labels and predictions arrays
     * @return Output variable (2D, shape [numClasses, numClasses})
     */
    public SDVariable confusionMatrix(String name, SDVariable labels, SDVariable pred, Integer numClasses, SDVariable weights) {
        validateInteger("confusionMatrix", "labels", labels);
        validateInteger("confusionMatrix", "prediction", pred);
        validateNumerical("confusionMatrix", "weights", weights);
        SDVariable result = f().confusionMatrix(labels, pred, numClasses, weights);
        return updateVariableNameAndReference(result, name);
    }

    /**
     * Elementwise cosine operation: out = cos(x)
     *
     * @param x Input variable
     * @return Output variable
     */
    public SDVariable cos(SDVariable x) {
        return cos(null, x);
    }

    /**
     * Elementwise cosine operation: out = cos(x)
     *
     * @param name Output variable name
     * @param x    Input variable
     * @return Output variable
     */
    public SDVariable cos(String name, SDVariable x) {
        validateNumerical("cos", x);
        SDVariable result = f().cos(x);
        return updateVariableNameAndReference(result, name);
    }

    /**
     * Elementwise cosh (hyperbolic cosine) operation: out = cosh(x)
     *
     * @param x Input variable
     * @return Output variable
     */
    public SDVariable cosh(SDVariable x) {
        return cosh(null, x);
    }

    /**
     * Elementwise cosh (hyperbolic cosine) operation: out = cosh(x)
     *
     * @param name Output variable name
     * @param x    Input variable
     * @return Output variable
     */
    public SDVariable cosh(String name, SDVariable x) {
        validateNumerical("cosh", x);
        SDVariable result = f().cosh(x);
        return updateVariableNameAndReference(result, name);
    }

    /**
     * @see #cosineDistance(String, SDVariable, SDVariable, int...)
     */
    public SDVariable cosineDistance(SDVariable x, SDVariable y, int... dimensions) {
        return cosineDistance(null, x, y, dimensions);
    }

    /**
     * Cosine distance reduction operation. The output contains the cosine distance for each
     * tensor/subset along the specified dimensions:<br>
     * out = 1.0 - cosineSimilarity(x,y)<br>
     * See {@link #cosineSimilarity(String, SDVariable, SDVariable, int...)}
     *
     * @param name       Name of the output variable
     * @param x          Input variable x
     * @param y          Input variable y
     * @param dimensions Dimensions to calculate cosine similarity over
     * @return Output variable
     */
    public SDVariable cosineDistance(String name, SDVariable x, SDVariable y, int... dimensions) {
        validateNumerical("cosine distance", x, y);
        SDVariable result = f().cosineDistance(x, y, dimensions);
        return updateVariableNameAndReference(result, name);
    }

    /**
     * @see #cosineSimilarity(String, SDVariable, SDVariable, int...)
     */
    public SDVariable cosineSimilarity(SDVariable x, SDVariable y, int... dimensions) {
        return cosineSimilarity(sd.generateNewVarName(CosineSimilarity.OP_NAME, 0), x, y, dimensions);
    }

    /**
     * Cosine similarity pairwise reduction operation. The output contains the cosine similarity for each tensor/subset
     * along the specified dimensions:<br>
     * out = (sum_i x[i] * y[i]) / ( sqrt(sum_i x[i]^2) * sqrt(sum_i y[i]^2)
     *
     * @param x          Input variable x
     * @param y          Input variable y
     * @param dimensions Dimensions to calculate cosine similarity over
     * @return Output variable
     */
    public SDVariable cosineSimilarity(String name, SDVariable x, SDVariable y, int... dimensions) {
        validateNumerical("cosine similarity", x, y);
        SDVariable cosim = f().cosineSimilarity(x, y, dimensions);
        return updateVariableNameAndReference(cosim, name);
    }

    /**
     * Count non zero array reduction operation, optionally along specified dimensions: out = count(x != 0)
     *
     * @param input      Input variable
     * @param dimensions Dimensions to reduce over. If dimensions are not specified, full array reduction is performed
     * @return Reduced array of rank (input rank - num dimensions)
     */
    public SDVariable countNonZero(SDVariable input, int... dimensions) {
        return countNonZero(null, input, dimensions);
    }

    /**
     * Count non zero array reduction operation, optionally along specified dimensions: out = count(x != 0)
     *
     * @param name       Name of the output variable
     * @param input      Input variable
     * @param dimensions Dimensions to reduce over. If dimensions are not specified, full array reduction is performed
     * @return Reduced array of rank (input rank - num dimensions)
     */
    public SDVariable countNonZero(String name, SDVariable input, int... dimensions) {
        validateNumerical("countNonZero", input);
        SDVariable res = f().countNonZero(input, dimensions);
        return updateVariableNameAndReference(res, name);
    }

    /**
     * Count zero array reduction operation, optionally along specified dimensions: out = count(x == 0)
     *
     * @param input      Input variable
     * @param dimensions Dimensions to reduce over. If dimensions are not specified, full array reduction is performed
     * @return Reduced array of rank (input rank - num dimensions)
     */
    public SDVariable countZero(SDVariable input, int... dimensions) {
        return countZero(null, input, dimensions);
    }

    /**
     * Count zero array reduction operation, optionally along specified dimensions: out = count(x == 0)
     *
     * @param name       Name of the output variable
     * @param input      Input variable
     * @param dimensions Dimensions to reduce over. If dimensions are not specified, full array reduction is performed
     * @return Reduced array of rank (input rank - num dimensions)
     */
    public SDVariable countZero(String name, SDVariable input, int... dimensions) {
        validateNumerical("countNonZero", input);
        SDVariable res = f().countZero(input, dimensions);
        return updateVariableNameAndReference(res, name);
    }

    /**
     * @see #cross(String, SDVariable, SDVariable)
     */
    public SDVariable cross(SDVariable a, SDVariable b) {
        return cross(null, a, b);
    }

    /**
     * Returns the pair-wise cross product of equal size arrays a and b: a x b = ||a||x||b|| sin(theta).
     * Can take rank 1 or above inputs (of equal shapes), but note that the last dimension must have dimension 3
     *
     * @param a First input
     * @param b Second input
     * @return Element-wise cross product
     */
    public SDVariable cross(String name, SDVariable a, SDVariable b) {
        validateNumerical("cross", a, b);
        SDVariable ret = f().cross(a, b);
        return updateVariableNameAndReference(ret, name);
    }

    /**
     * Element-wise cube function: out = x^3
     *
     * @param x Input variable
     * @return Output variable
     */
    public SDVariable cube(SDVariable x) {
        return cube(null, x);
    }

    /**
     * Element-wise cube function: out = x^3
     *
     * @param name Output variable name
     * @param x    Input variable
     * @return Output variable
     */
    public SDVariable cube(String name, SDVariable x) {
        validateNumerical("cube", x);
        SDVariable result = f().cube(x);
        return updateVariableNameAndReference(result, name);
    }

    /**
     * @see #diag(String, SDVariable)
     */
    public SDVariable diag(SDVariable x) {
        return diag(null, x);
    }

    /**
     * Returns an output variable with diagonal values equal to the specified values; off-diagonal values will be set to 0<br>
     * For example, if input = [1,2,3], then output is given by:<br>
     * [ 1, 0, 0]<br>
     * [ 0, 2, 0]<br>
     * [ 0, 0, 3]<br>
     * <br>
     * Higher input ranks are also supported: if input has shape [a,...,R-1] then output[i,...,k,i,...,k] = input[i,...,k].
     * i.e., for input rank R, output has rank 2R
     *
     * @param name Name of the output variable
     * @param x    Input variable
     * @return Output variable
     */
    public SDVariable diag(String name, SDVariable x) {
        SDVariable ret = f().diag(x);
        return updateVariableNameAndReference(ret, name);
    }

    /**
     * @see #diagPart(String, SDVariable)
     */
    public SDVariable diagPart(SDVariable x) {
        return diagPart(null, x);
    }

    /**
     * Extract the diagonal part from the input array.<br>
     * If input is<br>
     * [ 1, 0, 0]<br>
     * [ 0, 2, 0]<br>
     * [ 0, 0, 3]<br>
     * then output is [1, 2, 3].<br>
     * Supports higher dimensions: in general, out[i,...,k] = in[i,...,k,i,...,k]
     *
     * @param x Input variable
     * @return Diagonal part of the input
     * @see #diag(String, SDVariable)
     */
    public SDVariable diagPart(String name, SDVariable x) {
        SDVariable ret = f().diagPart(x);
        return updateVariableNameAndReference(ret, name);
    }

    /**
     * Entropy reduction: -sum(x * log(x))
     *
     * @param in         Input variable
     * @param dimensions Dimensions to reduce on (null/empty for full array)
     * @return Output variable
     */
    public SDVariable entropy(SDVariable in, int... dimensions) {
        return entropy(null, in, dimensions);
    }

    /**
     * Entropy reduction: -sum(x * log(x))
     *
     * @param name       Name of the output variable
     * @param in         Input variable
     * @param dimensions Dimensions to reduce on (null/empty for full array)
     * @return Output variable: reduced array of rank (input rank - num dimensions)
     */
    public SDVariable entropy(String name, SDVariable in, int... dimensions) {
        validateNumerical("entropy reduction", in);
        SDVariable ret = f().entropy(in, dimensions);
        return updateVariableNameAndReference(ret, name);
    }

    /**
     * Element-wise Gaussian error function - out = erf(in)
     *
     * @param x Input variable
     * @return Output variable
     */
    public SDVariable erf(SDVariable x) {
        return erf(null, x);
    }

    /**
     * Element-wise Gaussian error function - out = erf(in)
     *
     * @param name Output variable name
     * @param x    Input variable
     * @return Output variable
     */
    public SDVariable erf(String name, SDVariable x) {
        validateNumerical("erf (error function)", x);
        SDVariable ret = f().erf(x);
        return updateVariableNameAndReference(ret, name);
    }

    /**
     * Element-wise complementary Gaussian error function - out = erfc(in) = 1 - erf(in)
     *
     * @param x Input variable
     * @return Output variable
     */
    public SDVariable erfc(SDVariable x) {
        return erfc(null, x);
    }

    /**
     * Element-wise complementary Gaussian error function - out = erfc(in) = 1 - erf(in)
     *
     * @param name Name of the output variable
     * @param x    Input variable
     * @return Output variable
     */
    public SDVariable erfc(String name, SDVariable x) {
        validateNumerical("erfc", x);
        SDVariable ret = f().erfc(x);
        return updateVariableNameAndReference(ret, name);
    }

    /**
     * @see #euclideanDistance(String, SDVariable, SDVariable, int...)
     */
    public SDVariable euclideanDistance(SDVariable x, SDVariable y, int... dimensions) {
        return euclideanDistance(sd.generateNewVarName(EuclideanDistance.OP_NAME, 0), x, y, dimensions);
    }

    /**
     * Euclidean distance (l2 norm, l2 distance) reduction operation. The output contains the Euclidean distance for each
     * tensor/subset along the specified dimensions:<br>
     * out = sqrt( sum_i (x[i] - y[i])^2 )
     *
     * @param x          Input variable x
     * @param y          Input variable y
     * @param dimensions Dimensions to calculate cosine similarity over
     * @return Output variable
     */
    public SDVariable euclideanDistance(String name, SDVariable x, SDVariable y, int... dimensions) {
        validateNumerical("euclidean distance", x, y);
        SDVariable result = f().euclideanDistance(x, y, dimensions);
        return updateVariableNameAndReference(result, name);
    }

    /**
     * Elementwise exponent function: out = exp(x) = 2.71828...^x
     *
     * @param x Input variable
     * @return Output variable
     */
    public SDVariable exp(SDVariable x) {
        return exp(null, x);
    }

    /**
     * Elementwise exponent function: out = exp(x) = 2.71828...^x
     *
     * @param name Output variable name
     * @param x    Input variable
     * @return Output variable
     */
    public SDVariable exp(String name, SDVariable x) {
        validateNumerical("exp", x);
        SDVariable result = f().exp(x);
        return updateVariableNameAndReference(result, name);
    }

    /**
     * Elementwise 1.0 - exponent function: out = 1.0 - exp(x) = 1.0 - 2.71828...^x
     *
     * @param x Input variable
     * @return Output variable
     */
    public SDVariable expm1(SDVariable x) {
        return expm1(null, x);
    }

    /**
     * Elementwise 1.0 - exponent function: out = 1.0 - exp(x) = 1.0 - 2.71828...^x
     *
     * @param name Output variable name
     * @param x    Input variable
     * @return Output variable
     */
    public SDVariable expm1(String name, SDVariable x) {
        validateNumerical("expm1", x);
        SDVariable result = f().expm1(x);
        return updateVariableNameAndReference(result, name);
    }

    /**
     * Generate a square identity matrix with the specified number of rows.
     *
     * @param rows Number of rows (and columns)
     * @return SDVariable with an identity matrix array
     */
    public SDVariable eye(int rows) {
        return eye(rows, rows);
    }

    /**
     * Generate an identity matrix with the specified number of rows and columns.
     *
     * @param rows Number of rows
     */
    public SDVariable eye(String name, int rows) {
        return eye(name, rows, rows);
    }

    /**
     * @see #eye(String, int, int)
     */
    public SDVariable eye(int rows, int cols) {
        return eye(null, rows, cols);
    }

    /**
     * As per {@link #eye(String, int, int, DataType)} but with the default datatype, {@link Eye#DEFAULT_DTYPE}
     */
    public SDVariable eye(String name, int rows, int cols) {
        return eye(name, rows, cols, Eye.DEFAULT_DTYPE);
    }

    /**
     * Generate an identity matrix with the specified number of rows and columns
     * Example:<br>
     * <pre>
     * {@code SDVariable eye = eye(3,2)
     * eye:
     * [ 1, 0]
     * [ 0, 1]
     * [ 0, 0]}
     * </pre>
     *
     * @param name Name of the new SDVariable
     * @param rows Number of rows
     * @param cols Number of columns
     * @return SDVaribable identity matrix
     */
    public SDVariable eye(String name, int rows, int cols, DataType dataType) {
        return eye(name, rows, cols, dataType);
    }

    /**
     * see {@link #eye(String, int, int, DataType, int...)}
     */
    public SDVariable eye(int rows, int cols, DataType dataType, int... batchDimension) {
        return eye(null, rows, cols, dataType, batchDimension);
    }

    /**
     * Generate an identity matrix with the specified number of rows and columns, with optional leading dims<br>
     * Example:<br>
     * batchShape: [3,3]<br>
     * numRows: 2<br>
     * numCols: 4<br>
     * returns a tensor of shape (3, 3, 2, 4) that consists of 3 * 3 batches of (2,4)-shaped identity matrices:<br>
     * 1 0 0 0<br>
     * 0 1 0 0<br>
     *
     * @param rows           Number of rows
     * @param cols           Number of columns
     * @param batchDimension Batch dimensions. May be null
     */
    public SDVariable eye(String name, int rows, int cols, DataType dataType, int... batchDimension) {
        SDVariable eye = new Eye(sd, rows, cols, dataType, batchDimension).outputVariables()[0];
        return updateVariableNameAndReference(eye, name);
    }

    /**
     * As per {@link #eye(int, int, DataType, int...)} bit with the number of rows/columns specified as scalar SDVariables,
     * and the batch dimension specified as a 1D SDVariable
     */
    public SDVariable eye(SDVariable rows, SDVariable cols, SDVariable batchDimension) {
        return eye(null, rows, cols, batchDimension);
    }

    /**
     * As per {@link #eye(String, int, int, int...)} bit with the number of rows/columns specified as scalar SDVariables,
     * and the batch dimension specified as a 1D SDVariable
     */
    public SDVariable eye(String name, SDVariable rows, SDVariable cols, SDVariable batchDimension) {
        SDVariable eye = new Eye(sd, rows, cols, batchDimension).outputVariable();
        return updateVariableNameAndReference(eye, name);
    }

    /**
     * As per {@link #eye(String, int, int)} bit with the number of rows/columns specified as scalar SDVariables
     */
    public SDVariable eye(String name, SDVariable rows, SDVariable cols) {
        SDVariable eye = new Eye(sd, rows, cols).outputVariables()[0];
        return updateVariableNameAndReference(eye, name);
    }

    /**
     * As per {@link #eye(int, int)} bit with the number of rows/columns specified as scalar SDVariables
     */
    public SDVariable eye(SDVariable rows, SDVariable cols) {
        SDVariable eye = new Eye(sd, rows, cols).outputVariables()[0];
        return updateVariableNameAndReference(eye, null);
    }

    /**
     * As per {@link #eye(String, int)} but with the number of rows specified as a scalar SDVariable
     */
    public SDVariable eye(String name, SDVariable rows) {
        SDVariable eye = new Eye(sd, rows).outputVariables()[0];
        return updateVariableNameAndReference(eye, name);
    }

    /**
     * As per {@link #eye(int)} but with the number of rows specified as a scalar SDVariable
     */
    public SDVariable eye(SDVariable rows) {
        SDVariable eye = new Eye(sd, rows).outputVariables()[0];
        return updateVariableNameAndReference(eye, null);
    }

    /**
     * @see #firstIndex(String, SDVariable, Condition, int...)
     */
    public SDVariable firstIndex(SDVariable in, Condition condition, int... dimensions) {
        return firstIndex(null, in, condition, dimensions);
    }

    /**
     * First index reduction operation.<br>
     * Returns a variable that contains the index of the first element that matches the specified condition (for each
     * slice along the specified dimensions)
     *
     * @param name       Name of the output variable
     * @param in         Input variable
     * @param condition  Condition to check on input variable
     * @param dimensions Dimensions to reduce over. If dimensions are not specified, full array reduction is performed
     * @return Reduced array of rank (input rank - num dimensions)
     */
    public SDVariable firstIndex(String name, SDVariable in, Condition condition, int... dimensions) {
        return firstIndex(name, in, condition, false, dimensions);
    }

    /**
     * First index reduction operation.<br>
     * Returns a variable that contains the index of the first element that matches the specified condition (for each
     * slice along the specified dimensions)<br>
     * Note that if keepDims = true, the output variable has the same rank as the input variable,
     * with the reduced dimensions having size 1. This can be useful for later broadcast operations (such as subtracting
     * the mean along a dimension).<br>
     * Example: if input has shape [a,b,c] and dimensions=[1] then output has shape:
     * keepDims = true: [a,1,c]<br>
     * keepDims = false: [a,c]
     *
     * @param name       Name of the output variable
     * @param in         Input variable
     * @param condition  Condition to check on input variable
     * @param keepDims   If true: keep the dimensions that are reduced on (as length 1). False: remove the reduction dimensions
     * @param dimensions Dimensions to reduce over. If dimensions are not specified, full array reduction is performed
     * @return Reduced array of rank (input rank - num dimensions)
     */
    public SDVariable firstIndex(String name, SDVariable in, Condition condition, boolean keepDims, int... dimensions) {
        SDVariable ret = f().firstIndex(in, condition, keepDims, dimensions);
        return updateVariableNameAndReference(ret, name);
    }

    /**
     * @see #firstIndex(String, SDVariable, Condition, boolean, int...)
     */
    public SDVariable firstIndex(SDVariable in, Condition condition, boolean keepDims, int... dimensions) {
        return firstIndex(null, in, condition, keepDims, dimensions);
    }

    /**
     * Element-wise floor function: out = floor(x).
     * Rounds each value down to the nearest integer value (if not already an integer)
     *
     * @param x Input variable
     * @return Output variable
     */
    public SDVariable floor(SDVariable x) {
        return floor(null, x);
    }

    /**
     * Element-wise floor function: out = floor(x).
     * Rounds each value down to the nearest integer value (if not already an integer)
     *
     * @param name Output variable name
     * @param x    Input variable
     * @return Output variable
     */
    public SDVariable floor(String name, SDVariable x) {
        validateFloatingPoint("floor", x);
        SDVariable result = f().floor(x);
        return updateVariableNameAndReference(result, name);
    }

    /**
     * @see #hammingDistance(String, SDVariable, SDVariable, int...)
     */
    public SDVariable hammingDistance(SDVariable x, SDVariable y, int... dimensions) {
        return hammingDistance(null, x, y, dimensions);
    }

    /**
     * Hamming distance reduction operation. The output contains the cosine distance for each
     * tensor/subset along the specified dimensions:<br>
     * out = count( x[i] != y[i] )
     *
     * @param name       Name of the output variable
     * @param x          Input variable x
     * @param y          Input variable y
     * @param dimensions Dimensions to calculate cosine similarity over
     * @return Output variable
     */
    public SDVariable hammingDistance(String name, SDVariable x, SDVariable y, int... dimensions) {
        validateNumerical("hamming distance reduction", x, y);
        SDVariable result = f().hammingDistance(x, y, dimensions);
        return updateVariableNameAndReference(result, name);
    }

    /**
     * Index of the max absolute value: argmax(abs(in))
     *
     * @see SameDiff#argmax(SDVariable, int...)
     */
    public SDVariable iamax(SDVariable in, int... dimensions) {
        return iamax(null, in, dimensions);
    }

    /**
     * Index of the max absolute value: argmax(abs(in))
     *
     * @see SameDiff#argmax(String, SDVariable, boolean, int...)
     */
    public SDVariable iamax(String name, SDVariable in, int... dimensions) {
        return iamax(name, in, false, dimensions);
    }

    /**
     * Index of the max absolute value: argmax(abs(in))
     *
     * @see SameDiff#argmax(String, SDVariable, boolean, int...)
     */
    public SDVariable iamax(String name, SDVariable in, boolean keepDims, int... dimensions) {
        validateNumerical("iamax", in);
        SDVariable ret = f().iamax(in, keepDims, dimensions);
        return updateVariableNameAndReference(ret, name);
    }

    /**
     * Index of the max absolute value: argmax(abs(in))
     *
     * @see SameDiff#argmax(String, SDVariable, boolean, int...)
     */
    public SDVariable iamax(SDVariable in, boolean keepDims, int... dimensions) {
        return iamax(null, in, keepDims, dimensions);
    }

    /**
     * Index of the min absolute value: argmin(abs(in))
     *
     * @see SameDiff#argmin(String, SDVariable, boolean, int...)
     */
    public SDVariable iamin(SDVariable in, int... dimensions) {
        return iamin(null, in, dimensions);
    }

    /**
     * Index of the min absolute value: argmin(abs(in))
     *
     * @see SameDiff#argmin(String, SDVariable, boolean, int...)
     */
    public SDVariable iamin(String name, SDVariable in, int... dimensions) {
        return iamin(name, in, false, dimensions);
    }

    /**
     * Index of the min absolute value: argmin(abs(in))
     *
     * @see SameDiff#argmin(String, SDVariable, boolean, int...)
     */
    public SDVariable iamin(String name, SDVariable in, boolean keepDims, int... dimensions) {
        validateNumerical("iamin", in);
        SDVariable ret = f().iamin(in, keepDims, dimensions);
        return updateVariableNameAndReference(ret, name);
    }

    /**
     * Index of the min absolute value: argmin(abs(in))
     *
     * @see SameDiff#argmin(String, SDVariable, boolean, int...)
     */
    public SDVariable iamin(SDVariable in, boolean keepDims, int... dimensions) {
        return iamin(null, in, keepDims, dimensions);
    }

    /**
     * Is finite operation: elementwise isFinite(x)<br>
     * Returns an array with the same shape/size as the input, with values 1 where condition is satisfied, or
     * value 0 otherwise
     *
     * @param x Input array
     * @return Output SDVariable with values 0 and 1 based on where the condition is satisfied
     */
    public SDVariable isFinite(SDVariable x) {
        return isFinite(null, x);
    }

    /**
     * Is finite operation: elementwise isFinite(x)<br>
     * Returns an array with the same shape/size as the input, with values 1 where condition is satisfied, or
     * value 0 otherwise
     *
     * @param name Output variable name
     * @param x    Input array
     * @return Output SDVariable with values 0 and 1 based on where the condition is satisfied
     */
    public SDVariable isFinite(String name, SDVariable x) {
        validateFloatingPoint("isFinite", x);
        SDVariable result = f().isFinite(x);
        return updateVariableNameAndReference(result, name);
    }

    /**
     * Is infinite operation: elementwise isInfinite(x)<br>
     * Returns an array with the same shape/size as the input, with values 1 where condition is satisfied, or
     * value 0 otherwise
     *
     * @param x Input array
     * @return Output SDVariable with values 0 and 1 based on where the condition is satisfied
     */
    public SDVariable isInfinite(SDVariable x) {
        return isInfinite(null, x);
    }

    /**
     * Is infinite operation: elementwise isInfinite(x)<br>
     * Returns an array with the same shape/size as the input, with values 1 where condition is satisfied, or
     * value 0 otherwise
     *
     * @param name Output variable name
     * @param x    Input array
     * @return Output SDVariable with values 0 and 1 based on where the condition is satisfied
     */
    public SDVariable isInfinite(String name, SDVariable x) {
        validateFloatingPoint("isInfinite", x);
        SDVariable result = f().isInfinite(x);
        return updateVariableNameAndReference(result, name);
    }

    /**
     * Is maximum operation: elementwise x == max(x)<br>
     * Returns an array with the same shape/size as the input, with values 1 where condition is satisfied, or
     * value 0 otherwise
     *
     * @param x Input array
     * @return Output SDVariable with values 0 and 1 based on where the condition is satisfied
     */
    public SDVariable isMax(SDVariable x) {
        return isMax(null, x);
    }

    /**
     * Is maximum operation: elementwise x == max(x)<br>
     * Returns an array with the same shape/size as the input, with values 1 where condition is satisfied, or
     * value 0 otherwise
     *
     * @param name Name of the output variable
     * @param x    Input array
     * @return Output SDVariable with values 0 and 1 based on where the condition is satisfied
     */
    public SDVariable isMax(String name, SDVariable x) {
        validateNumerical("isMax", x);
        SDVariable ret = f().isMax(x);
        return updateVariableNameAndReference(ret, name);
    }

    /**
     * Is Not a Number operation: elementwise isNaN(x)<br>
     * Returns an array with the same shape/size as the input, with values 1 where condition is satisfied, or
     * value 0 otherwise
     *
     * @param x Input array
     * @return Output SDVariable with values 0 and 1 based on where the condition is satisfied
     */
    public SDVariable isNaN(SDVariable x) {
        return isNaN(null, x);
    }

    /**
     * Is Not a Number operation: elementwise isNaN(x)<br>
     * Returns an array with the same shape/size as the input, with values 1 where condition is satisfied, or
     * value 0 otherwise
     *
     * @param name Output variable name
     * @param x    Input array
     * @return Output SDVariable with values 0 and 1 based on where the condition is satisfied
     */
    public SDVariable isNaN(String name, SDVariable x) {
        validateFloatingPoint("isNaN", x);
        SDVariable result = f().isNaN(x);
        return updateVariableNameAndReference(result, name);
    }

    /**
     * Is the array non decreasing?<br>
     * An array is non-decreasing if for every valid i, x[i] <= x[i+1]. For Rank 2+ arrays, values are compared
     * in 'c' (row major) order
     *
     * @param x Input variable
     * @return Scalar variable with value 1 if non-decreasing, or 0 otherwise
     */
    public SDVariable isNonDecreasing(SDVariable x) {
        return isNonDecreasing(null, x);
    }

    /**
     * Is the array non decreasing?<br>
     * An array is non-decreasing if for every valid i, x[i] <= x[i+1]. For Rank 2+ arrays, values are compared
     * in 'c' (row major) order
     *
     * @param name Output name
     * @param x    Input variable
     * @return Scalar variable with value 1 if non-decreasing, or 0 otherwise
     */
    public SDVariable isNonDecreasing(String name, SDVariable x) {
        validateNumerical("isNonDecreasing", x);
        SDVariable result = f().isNonDecreasing(x);
        return updateVariableNameAndReference(result, name);
    }

    /**
     * Is the array strictly increasing?<br>
     * An array is strictly increasing if for every valid i, x[i] < x[i+1]. For Rank 2+ arrays, values are compared
     * in 'c' (row major) order
     *
     * @param x Input variable
     * @return Scalar variable with value 1 if strictly increasing, or 0 otherwise
     */
    public SDVariable isStrictlyIncreasing(SDVariable x) {
        return isStrictlyIncreasing(null, x);

    }

    /**
     * Is the array strictly increasing?<br>
     * An array is strictly increasing if for every valid i, x[i] < x[i+1]. For Rank 2+ arrays, values are compared
     * in 'c' (row major) order
     *
     * @param name Output variable name
     * @param x    Input variable
     * @return Scalar variable with value 1 if strictly increasing, or 0 otherwise
     */
    public SDVariable isStrictlyIncreasing(String name, SDVariable x) {
        validateNumerical("isStrictlyIncreasing", x);
        SDVariable result = f().isStrictlyIncreasing(x);
        return updateVariableNameAndReference(result, name);
    }

    /**
     * Jaccard similarity reduction operation. The output contains the Jaccard distance for each
     * tensor along the specified dimensions.
     *
     * @param x          Input variable x
     * @param y          Input variable y
     * @param dimensions Dimensions to calculate Jaccard similarity over
     * @return Output variable
     */
    public SDVariable jaccardDistance(SDVariable x, SDVariable y, int... dimensions) {
        return jaccardDistance(null, x, y, dimensions);
    }

    /**
     * Jaccard similarity reduction operation. The output contains the Jaccard distance for each
     * tensor along the specified dimensions.
     *
     * @param name       Name of the output variable
     * @param x          Input variable x
     * @param y          Input variable y
     * @param dimensions Dimensions to calculate Jaccard similarity over
     * @return Output variable
     */
    public SDVariable jaccardDistance(String name, SDVariable x, SDVariable y, int... dimensions) {
        validateNumerical("Jaccard distance reduction", x, y);
        SDVariable result = f().jaccardDistance(x, y, dimensions);
        return updateVariableNameAndReference(result, name);
    }

    /**
     * @see #lastIndex(String, SDVariable, Condition, int...)
     */
    public SDVariable lastIndex(SDVariable in, Condition condition, int... dimensions) {
        return lastIndex(null, in, condition, dimensions);
    }

    /**
     * Last index reduction operation.<br>
     * Returns a variable that contains the index of the last element that matches the specified condition (for each
     * slice along the specified dimensions)
     *
     * @param name       Name of the output variable
     * @param in         Input variable
     * @param condition  Condition to check on input variable
     * @param dimensions Dimensions to reduce over. If dimensions are not specified, full array reduction is performed
     * @return Reduced array of rank (input rank - num dimensions)
     */
    public SDVariable lastIndex(String name, SDVariable in, Condition condition, int... dimensions) {
        return lastIndex(name, in, condition, false, dimensions);
    }

    /**
     * Last index reduction operation.<br>
     * Returns a variable that contains the index of the last element that matches the specified condition (for each
     * slice along the specified dimensions)<br>
     * Note that if keepDims = true, the output variable has the same rank as the input variable,
     * with the reduced dimensions having size 1. This can be useful for later broadcast operations (such as subtracting
     * the mean along a dimension).<br>
     * Example: if input has shape [a,b,c] and dimensions=[1] then output has shape:
     * keepDims = true: [a,1,c]<br>
     * keepDims = false: [a,c]
     *
     * @param name       Name of the output variable
     * @param in         Input variable
     * @param condition  Condition to check on input variable
     * @param dimensions Dimensions to reduce over. If dimensions are not specified, full array reduction is performed
     * @return Reduced array of rank (input rank - num dimensions)
     */
    public SDVariable lastIndex(String name, SDVariable in, Condition condition, boolean keepDims, int... dimensions) {
        SDVariable ret = f().lastIndex(in, condition, keepDims, dimensions);
        return updateVariableNameAndReference(ret, name);
    }

    /**
     * @see #lastIndex(String, SDVariable, Condition, boolean, int...)
     */
    public SDVariable lastIndex(SDVariable in, Condition condition, boolean keepDims, int... dimensions) {
        return lastIndex(null, in, condition, keepDims, dimensions);
    }

    /**
     * List diff operation computes the difference between two 1d arrays, and also returns the indices - i.e., the positions
     * where the output appears in the input X.<br>
     * For inputs X and Y, listDiff returns everything in X but not in Y.<br>
     * For example, if {@code X=[1,10,3,7,6]} and {@code Y=[10, 6]), then:
     * output 0 (difference) = {@code [1,3,7]}<br>
     * output 1 (indices) = {@code [0, 2, 3]}<br>
     * @param x Input 1 - input values
     * @param y Input 2 - values to remove
     * @return 2 outputs - difference, and indices
     */
    public SDVariable[] listDiff(SDVariable x, SDVariable y){
        return f().listdiff(x, y);
    }

    /**
     * Element-wise logarithm function (base e - natural logarithm): out = log(x)
     *
     * @param x Input variable
     * @return Output variable
     */
    public SDVariable log(SDVariable x) {
        return log(null, x);
    }

    /**
     * Element-wise logarithm function (base e - natural logarithm): out = log(x)
     *
     * @param name Output variable name
     * @param x    Input variable
     * @return Output variable
     */
    public SDVariable log(String name, SDVariable x) {
        validateNumerical("log", x);
        SDVariable result = f().log(x);
        return updateVariableNameAndReference(result, name);
    }

    /**
     * Element-wise logarithm function (with specified base): out = log_{base}(x)
     *
     * @param in   Input variable
     * @param base Logarithm base
     * @return Output variable
     */
    public SDVariable log(SDVariable in, double base) {
        return log(null, in, base);
    }

    /**
     * Element-wise logarithm function (with specified base): out = log_{base}(x)
     *
     * @param name Name of the output variable
     * @param in   Input variable
     * @param base Logarithm base
     * @return Output variable
     */
    public SDVariable log(String name, SDVariable in, double base) {
        validateNumerical("log", in);
        SDVariable ret = f().log(in, base);
        return updateVariableNameAndReference(ret, name);
    }

    /**
     * Elementwise natural logarithm function: out = log_e (1 + x)
     *
     * @param x Input variable
     * @return Output variable
     */
    public SDVariable log1p(SDVariable x) {
        return log1p(null, x);
    }

    /**
     * Elementwise natural logarithm function: out = log_e (1 + x)
     *
     * @param name Output variable name
     * @param x    Input variable
     * @return Output variable
     */
    public SDVariable log1p(String name, SDVariable x) {
        validateNumerical("log1p", x);
        SDVariable result = f().log1p(x);
        return updateVariableNameAndReference(result, name);
    }

    /**
     * Log entropy reduction: log(-sum(x * log(x)))
     *
     * @param in         Input variable
     * @param dimensions Dimensions to reduce on (null for full array)
     * @return Output variable: reduced array of rank (input rank - num dimensions)
     */
    public SDVariable logEntropy(SDVariable in, int... dimensions) {
        return logEntropy(null, in, dimensions);
    }

    /**
     * Log entropy reduction: log(-sum(x * log(x)))
     *
     * @param name       Name of the output variable
     * @param in         Input variable
     * @param dimensions Dimensions to reduce on (null for full array)
     * @return Output variable: reduced array of rank (input rank - num dimensions)
     */
    public SDVariable logEntropy(String name, SDVariable in, int... dimensions) {
        validateNumerical("logEntropy reduction", in);
        SDVariable ret = f().logEntropy(in, dimensions);
        return updateVariableNameAndReference(ret, name);
    }

    /**
     * Log-sum-exp reduction (optionally along dimension).
     * Computes log(sum(exp(x))
     *
     * @param input      Input variable
     * @param dimensions Optional dimensions to reduce along
     * @return Output variable
     */
    public SDVariable logSumExp(SDVariable input, int... dimensions) {
        return logSumExp(null, input, dimensions);
    }

    /**
     * Log-sum-exp reduction (optionally along dimension).
     * Computes log(sum(exp(x))
     *
     * @param name       Name of the output variable
     * @param input      Input variable
     * @param dimensions Optional dimensions to reduce along
     * @return Output variable
     */
    public SDVariable logSumExp(String name, SDVariable input, int... dimensions) {
        return logSumExp(name, input, false, dimensions);
    }

    public SDVariable logSumExp(String name, SDVariable input, boolean keepDims, int... dimensions) {
        validateNumerical("logSumExp reduction", input);
        SDVariable ret = f().logSumExp(input, keepDims, dimensions);
        return updateVariableNameAndReference(ret, name);
    }

    /**
     * @see #manhattanDistance(String, SDVariable, SDVariable, int...)
     */
    public SDVariable manhattanDistance(SDVariable x, SDVariable y, int... dimensions) {
        return manhattanDistance(sd.generateNewVarName(ManhattanDistance.OP_NAME, 0), x, y, dimensions);
    }

    /**
     * Manhattan distance (l1 norm, l1 distance) reduction operation. The output contains the Manhattan distance for each
     * tensor/subset along the specified dimensions:<br>
     * out = sum_i abs(x[i]-y[i])
     *
     * @param name       Name of the output variable
     * @param x          Input variable x
     * @param y          Input variable y
     * @param dimensions Dimensions to calculate cosine similarity over
     * @return Output variable
     */
    public SDVariable manhattanDistance(String name, SDVariable x, SDVariable y, int... dimensions) {
        validateNumerical("manhattan distance", x, y);
        SDVariable result = f().manhattanDistance(x, y, dimensions);
        return updateVariableNameAndReference(result, name);
    }

    /**
     * @see #matrixDeterminant(String, SDVariable)
     */
    public SDVariable matrixDeterminant(SDVariable in) {
        return matrixDeterminant(null, in);
    }

    /**
     * Matrix determinant op. For 2D input, this returns the standard matrix determinant.
     * For higher dimensional input with shape [..., m, m] the matrix determinant is returned for each
     * shape [m,m] sub-matrix.
     *
     * @param name Name of the output variable
     * @param in   Input
     * @return Matrix determinant variable
     */
    public SDVariable matrixDeterminant(String name, SDVariable in) {
        validateNumerical("matrix determinant", in);
        SDVariable ret = f().matrixDeterminant(in);
        return updateVariableNameAndReference(ret, name);
    }

    /**
     * @see #matrixInverse(String, SDVariable)
     */
    public SDVariable matrixInverse(SDVariable in) {
        return matrixInverse(null, in);
    }

    /**
     * Matrix inverse op. For 2D input, this returns the standard matrix inverse.
     * For higher dimensional input with shape [..., m, m] the matrix inverse is returned for each
     * shape [m,m] sub-matrix.
     *
     * @param name Name of the output variable
     * @param in   Input
     * @return Matrix inverse variable
     */
    public SDVariable matrixInverse(String name, SDVariable in) {
        validateFloatingPoint("matrix inverse", in);
        SDVariable ret = f().matrixInverse(in);
        return updateVariableNameAndReference(ret, name);
    }

    /**
     * Merge add function: merges an arbitrary number of equal shaped arrays using elementwise addition:
     * out = sum_i in[i]
     *
     * @param x Input variables
     * @return Output variable
     */
    public SDVariable mergeAdd(SDVariable... x) {
        return mergeAdd(null, x);
    }

    /**
     * Merge add function: merges an arbitrary number of equal shaped arrays using element-wise addition:
     * out = sum_i in[i]
     *
     * @param name   Name of the output variable
     * @param inputs Input variables
     * @return Output variable
     */
    public SDVariable mergeAdd(String name, SDVariable... inputs) {
        validateSameType("mergeAdd", true, inputs);
        SDVariable ret = f().mergeAdd(inputs);
        return updateVariableNameAndReference(ret, name);
    }

    /**
     * Merge average function: merges an arbitrary number of equal shaped arrays using element-wise mean operation:
     * out = mean_i in[i]
     *
     * @param inputs Input variables
     * @return Output variable
     */
    public SDVariable mergeAvg(SDVariable... inputs) {
        return mergeAvg(null, inputs);
    }

    /**
     * Merge average function: merges an arbitrary number of equal shaped arrays using element-wise mean operation:
     * out = mean_i in[i]
     *
     * @param name   Name of the output variable
     * @param inputs Input variables
     * @return Output variable
     */
    public SDVariable mergeAvg(String name, SDVariable... inputs) {
        validateSameType("mergeAvg", true, inputs);
        SDVariable ret = f().mergeAvg(inputs);
        return updateVariableNameAndReference(ret, name);
    }

    /**
     * Merge max function: merges an arbitrary number of equal shaped arrays using element-wise maximum operation:
     * out = max_i in[i]
     *
     * @param x Input variables
     * @return Output variable
     */
    public SDVariable mergeMax(SDVariable... x) {
        return mergeMax(null, x);
    }

    /**
     * Merge max function: merges an arbitrary number of equal shaped arrays using element-wise maximum operation:
     * out = max_i in[i]
     *
     * @param inputs Input variables
     * @return Output variable
     */
    public SDVariable mergeMax(String name, SDVariable... inputs) {
        validateSameType("mergeMax", true, inputs);
        SDVariable ret = f().mergeMax(inputs);
        return updateVariableNameAndReference(ret, name);
    }

    /**
     * @see #meshgrid(List, SDVariable...)
     */
    public SDVariable[] meshgrid(SDVariable... inputs) {
        return meshgrid(null, inputs);
    }

    /**
     * Broadcast the 1D input variables onto an n-dimensional grid.<br>
     * The resulting variable can be used for example for evaluating functions at all locations on a grid.<br>
     * Example:<br>
     * <pre>
     * {@code input1 = [1, 2, 3]
     * input2 = [4, 5, 6]
     * SDVariable[] out = meshgrid(input1, input2)
     * out[0]:
     * [ 1, 2, 3]
     * [ 1, 2, 3]
     * [ 1, 2, 3]
     *
     * out[1]:
     * [ 4, 4, 4]
     * [ 5, 5, 5]
     * [ 6, 6, 6]}
     * </pre>
     * <br>
     *
     * @param names  List of names for the output variables. Must have exactly N names for N input arrays
     * @param inputs N x 1D input variables
     * @return an array of exactly N SDVariables (for N inputs), of rank N
     */
    public SDVariable[] meshgrid(List<String> names, SDVariable... inputs) {
        return meshgrid(names, true, inputs);
    }

    /**
     * @see #meshgrid(List, SDVariable...)
     */
    public SDVariable[] meshgrid(List<String> names, boolean cartesian, SDVariable... inputs) {
        Preconditions.checkState(names == null || names.size() == inputs.length,
                "Got %s names but %s inputs", (names == null ? 0 : names.size()), inputs.length);
        validateSameType("meshgrid", false, inputs);
        SDVariable[] ret = f().meshgrid(cartesian, inputs);
        for (int i = 0; i < ret.length; i++) {
            ret[i] = updateVariableNameAndReference(ret[i], names == null ? null : names.get(i));
        }
        return ret;
    }

    /**
     * @see #moments(String[], SDVariable, int...)
     */
    public SDVariable[] moments(SDVariable input, int... axes) {
        return moments(null, input, axes);
    }

    /**
     * Calculate the mean and (population) variance for the input variable, for the specified axis
     *
     * @param name  Name of the output variables. Can be null; if non-null, must be length 2
     * @param input Input to calculate moments for
     * @param axes  Dimensions to perform calculation over
     * @return Mean and variance variables
     */
    public SDVariable[] moments(String[] name, SDVariable input, int... axes) {
        validateNumerical("moments", input);
        SDVariable[] res = f().moments(input, axes);
        return sd.updateVariableNamesAndReferences(res, name);
    }

    /**
     * Elementwise negative operation: out = -x
     *
     * @param x Input variable
     * @return Output variable
     */
    public SDVariable neg(SDVariable x) {
        return neg(null, x);
    }

    /**
     * Elementwise negative operation: out = -x
     *
     * @param name Name of the output variable
     * @param x    Input variable
     * @return Output variable
     */
    public SDVariable neg(String name, SDVariable x) {
        validateNumerical("neg", x);
        SDVariable result = f().neg(x);
        return updateVariableNameAndReference(result, name);
    }

    /**
     * @see #normalizeMoments(String[], SDVariable, SDVariable, SDVariable, double)
     */
    public SDVariable[] normalizeMoments(SDVariable counts, SDVariable means, SDVariable variances, double shift) {
        return normalizeMoments(null, counts, means, variances, shift);
    }

    /**
     * Calculate the mean and variance from the sufficient statistics
     *
     * @param name      Name of the output variables. Can be null; if non-null, must be length 2
     * @param counts    Rank 0 (scalar) value with the total number of values used to calculate the sufficient statistics
     * @param means     Mean-value sufficient statistics: this is the SUM of all data values
     * @param variances Variaance sufficient statistics: this is the squared sum of all data values
     * @param shift     Shift value, possibly 0, used when calculating the sufficient statistics (for numerical stability)
     * @return Output variables: mean and population variance
     */
    public SDVariable[] normalizeMoments(String[] name, SDVariable counts, SDVariable means, SDVariable variances,
                                         double shift) {
        SDVariable[] res = f().normalizeMoments(counts, means, variances, shift);
        return sd.updateVariableNamesAndReferences(res, name);
    }

    /**
     * Boolean OR operation: elementwise (x != 0) || (y != 0)<br>
     * If x and y arrays have equal shape, the output shape is the same as these inputs.<br>
     * Note: supports broadcasting if x and y have different shapes and are broadcastable.<br>
     * Returns an array with values 1 where condition is satisfied, or value 0 otherwise.
     *
     * @param x Input 1
     * @param y Input 2
     * @return Output SDVariable with values 0 and 1 based on where the condition is satisfied
     */
    public SDVariable or(SDVariable x, SDVariable y) {
        return or(null, x, y);
    }

    /**
     * Boolean OR operation: elementwise (x != 0) || (y != 0)<br>
     * If x and y arrays have equal shape, the output shape is the same as these inputs.<br>
     * Note: supports broadcasting if x and y have different shapes and are broadcastable.<br>
     * Returns an array with values 1 where condition is satisfied, or value 0 otherwise.
     *
     * @param name Name of the output variable
     * @param x    Input 1
     * @param y    Input 2
     * @return Output SDVariable with values 0 and 1 based on where the condition is satisfied
     */
    public SDVariable or(String name, SDVariable x, SDVariable y) {
        validateBool("or", x, y);
        SDVariable result = f().or(x, y);
        return updateVariableNameAndReference(result, name);
    }

    /**
     * Element-wise power function: out = x^value
     *
     * @param x     Input variable
     * @param value Power to raise each element to
     * @return Output variable
     */
    public SDVariable pow(SDVariable x, double value) {
        return pow(null, x, value);
    }

    /**
     * Element-wise power function: out = x^value
     *
     * @param name  Output variable name
     * @param x     Input variable
     * @param value Power to raise each element to
     * @return Output variable
     */
    public SDVariable pow(String name, SDVariable x, double value) {
        validateNumerical("pow", x);
        SDVariable result = f().pow(x, value);
        return updateVariableNameAndReference(result, name);
    }

    /**
     * Element-wise (broadcastable) power function: out = x[i]^y[i]
     *
     * @param x Input variable
     * @param y Power
     * @return Output variable
     */
    public SDVariable pow(SDVariable x, SDVariable y) {
        return pow(null, x, y);
    }

    /**
     * Element-wise (broadcastable) power function: out = x[i]^y[i]
     *
     * @param name Output variable name
     * @param x    Input variable
     * @param y    Power
     * @return Output variable
     */
    public SDVariable pow(String name, SDVariable x, SDVariable y) {
        validateNumerical("pow", x, y);
        SDVariable result = f().pow(x, y);
        return updateVariableNameAndReference(result, name);
    }

    /**
     * Element-wise reciprocal (inverse) function: out[i] = 1 / in[i]
     *
     * @param a Input variable
     * @return Output variable
     */
    public SDVariable reciprocal(SDVariable a) {
        return reciprocal(null, a);
    }

    /**
     * Element-wise reciprocal (inverse) function: out[i] = 1 / in[i]
     *
     * @param name Name of the output variable
     * @param a    Input variable
     * @return Output variable
     */
    public SDVariable reciprocal(String name, SDVariable a) {
        validateNumerical("reciprocal", a);
        SDVariable ret = f().reciprocal(a);
        return updateVariableNameAndReference(ret, name);
    }

    /**
     * Elementwise round function: out = round(x).
     * Rounds (up or down depending on value) to the nearest integer value.
     *
     * @param x Input variable
     * @return Output variable
     */
    public SDVariable round(SDVariable x) {
        return round(null, x);
    }

    /**
     * Element-wise round function: out = round(x).
     * Rounds (up or down depending on value) to the nearest integer value.
     *
     * @param name Output variable name
     * @param x    Input variable
     * @return Output variable
     */
    public SDVariable round(String name, SDVariable x) {
        validateFloatingPoint("round", x);
        SDVariable result = f().round(x);
        return updateVariableNameAndReference(result, name);
    }

    /**
     * Element-wise reciprocal (inverse) of square root: out = 1.0 / sqrt(x)
     *
     * @param x Input variable
     * @return Output variable
     */
    public SDVariable rsqrt(SDVariable x) {
        return rsqrt(null, x);
    }

    /**
     * Element-wise reciprocal (inverse) of square root: out = 1.0 / sqrt(x)
     *
     * @param name Output variable name
     * @param x    Input variable
     * @return Output variable
     */
    public SDVariable rsqrt(String name, SDVariable x) {
        validateNumerical("rsqrt", x);
        SDVariable result = f().rsqrt(x);
        return updateVariableNameAndReference(result, name);
    }

    /**
     * @see #setDiag(String, SDVariable, SDVariable)
     */
    public SDVariable setDiag(SDVariable in, SDVariable diag) {
        return setDiag(null, in, diag);
    }

    /**
     * Set the diagonal value to the specified values<br>
     * If input is<br>
     * [ a, b, c]<br>
     * [ d, e, f]<br>
     * [ g, h, i]<br>
     * and diag = [ 1, 2, 3] then output is<br>
     * [ 1, b, c]<br>
     * [ d, 2, f]<br>
     * [ g, h, 3]<br>
     *
     * @param name Name of the output variable
     * @param in   Input variable
     * @param diag Diagonal
     * @return Output variable
     */
    public SDVariable setDiag(String name, SDVariable in, SDVariable diag) {
        SDVariable ret = f().setDiag(in, diag);
        return updateVariableNameAndReference(ret, name);
    }

    /**
     * Shannon Entropy reduction: -sum(x * log2(x))
     *
     * @param in         Input variable
     * @param dimensions Dimensions to reduce on (null/empty for full array)
     * @return Output variable
     */
    public SDVariable shannonEntropy(SDVariable in, int... dimensions) {
        return shannonEntropy(null, in, dimensions);
    }

    /**
     * Shannon Entropy reduction: -sum(x * log2(x))
     *
     * @param name       Name of the output variable
     * @param in         Input variable
     * @param dimensions Dimensions to reduce on (null/empty for full array)
     * @return Output variable: reduced array of rank (input rank - num dimensions)
     */
    public SDVariable shannonEntropy(String name, SDVariable in, int... dimensions) {
        validateNumerical("shannon entropy reduction", in);
        SDVariable ret = f().shannonEntropy(in, dimensions);
        return updateVariableNameAndReference(ret, name);
    }

    /**
     * Element-wise sign (signum) function:<br>
     * out = -1 if in < 0<br>
     * out = 0 if in = 0<br>
     * out = 1 if in > 0
     *
     * @param x Input variable
     * @return Output variable
     */
    public SDVariable sign(SDVariable x) {
        return sign(null, x);
    }

    /**
     * Element-wise sign (signum) function:<br>
     * out = -1 if in < 0<br>
     * out = 0 if in = 0<br>
     * out = 1 if in > 0
     *
     * @param name Output variable name
     * @param x    Input variable
     * @return Output variable
     */
    public SDVariable sign(String name, SDVariable x) {
        validateNumerical("sign", x);
        SDVariable result = f().sign(x);
        return updateVariableNameAndReference(result, name);
    }

    /**
     * Elementwise sine operation: out = sin(x)
     *
     * @param x Input variable
     * @return Output variable
     */
    public SDVariable sin(SDVariable x) {
        return sin(null, x);
    }

    /**
     * Elementwise sine operation: out = sin(x)
     *
     * @param name Output variable name
     * @param x    Input variable
     * @return Output variable
     */
    public SDVariable sin(String name, SDVariable x) {
        validateNumerical("sin", x);
        SDVariable result = f().sin(x);
        return updateVariableNameAndReference(result, name);
    }

    /**
     * Elementwise sinh (hyperbolic sine) operation: out = sinh(x)
     *
     * @param x Input variable
     * @return Output variable
     */
    public SDVariable sinh(SDVariable x) {
        return sinh(null, x);
    }

    /**
     * Elementwise sinh (hyperbolic sine) operation: out = sinh(x)
     *
     * @param name Output variable name
     * @param x    Input variable
     * @return Output variable
     */
    public SDVariable sinh(String name, SDVariable x) {
        validateNumerical("sinh", x);
        SDVariable result = f().sinh(x);
        return updateVariableNameAndReference(result, name);
    }

    /**
     * Element-wise square root function: out = sqrt(x)
     *
     * @param x Input variable
     * @return Output variable
     */
    public SDVariable sqrt(SDVariable x) {
        return sqrt(null, x);
    }

    /**
     * Element-wise square root function: out = sqrt(x)
     *
     * @param name Output variable name
     * @param x    Input variable
     * @return Output variable
     */
    public SDVariable sqrt(String name, SDVariable x) {
        validateNumerical("sqrt", x);
        SDVariable result = f().sqrt(x);
        return updateVariableNameAndReference(result, name);
    }

    /**
     * Element-wise square function: out = x^2
     *
     * @param x Input variable
     * @return Output variable
     */
    public SDVariable square(SDVariable x) {
        return square(null, x);
    }

    /**
     * Element-wise square function: out = x^2
     *
     * @param name Output variable name
     * @param x    Input variable
     * @return Output variable
     */
    public SDVariable square(String name, SDVariable x) {
        validateNumerical("square", x);
        SDVariable result = f().square(x);
        return updateVariableNameAndReference(result, name);
    }

    /**
     * Elementwise step function:<br>
     * out(x) = 1 if x >= cutoff<br>
     * out(x) = 0 otherwise<br>
     *
     * @param in     Input variable
     * @param cutoff Cutoff value for step function
     * @return Output variable
     */
    public SDVariable step(SDVariable in, double cutoff) {
        return step(null, in, cutoff);
    }

    /**
     * Elementwise step function:<br>
     * out(x) = 1 if x >= cutoff<br>
     * out(x) = 0 otherwise<br>
     *
     * @param name   Name of the output variable
     * @param in     Input variable
     * @param cutoff Cutoff value for step function
     * @return Output variable
     */
    public SDVariable step(String name, SDVariable in, double cutoff) {
        validateNumerical("step", in);
        SDVariable ret = f().step(in, cutoff);
        return updateVariableNameAndReference(ret, name);
    }

    /**
     * Standardize input variable along given axis
     *
     * @see #standardize(String, SDVariable, int...)
     *
     * @param x Input variable
     * @return Output variable
     */
    public SDVariable standardize(SDVariable x, int... dimensions) {
        return standardize(null, x, dimensions);
    }

    /**
     * Standardize input variable along given axis
     * <p>
     * out = (x - mean) / stdev
     * <p>
     * with mean and stdev being calculated along the given dimension.
     *
     * <p>
     * For example: given x as a mini batch of the shape [numExamples, exampleLength]:
     * <ul>
     *  <li>use dimension 1 too use the statistics (mean, stdev) for each example</li>
     *  <li>use dimension 0 if you want to use the statistics for each column across all examples</li>
     *  <li>use dimensions 0,1 if you want to use the statistics across all columns and examples</li>
     * </ul>
     *
     * @param name Name of the output variable
     * @param x    Input variable
     * @return Output variable
     */
    public SDVariable standardize(String name, SDVariable x, int... dimensions) {
        validateNumerical("standardize", x);
        SDVariable result = f().standardize(x, dimensions);
        return updateVariableNameAndReference(result, name);
    }

    /**
     * Elementwise tangent operation: out = tan(x)
     *
     * @param x Input variable
     * @return Output variable
     */
    public SDVariable tan(SDVariable x) {
        return tan(null, x);
    }

    /**
     * Elementwise tangent operation: out = tan(x)
     *
     * @param name Output variable name
     * @param x    Input variable
     * @return Output variable
     */
    public SDVariable tan(String name, SDVariable x) {
        validateNumerical("tan", x);
        SDVariable result = f().tan(x);
        return updateVariableNameAndReference(result, name);
    }

    /**
     * Elementwise tanh (hyperbolic tangent) operation: out = tanh(x)
     *
     * @param x Input variable
     * @return Output variable
     */
    public SDVariable tanh(SDVariable x) {
        return tanh(null, x);
    }

    /**
     * Elementwise tanh (hyperbolic tangent) operation: out = tanh(x)
     *
     * @param name Output variable name
     * @param x    Input variable
     * @return Output variable
     */
    public SDVariable tanh(String name, SDVariable x) {
        validateNumerical("tanh", x);
        SDVariable result = f().tanh(x);
        return updateVariableNameAndReference(result, name);
    }

    /**
     * @see #trace(String, SDVariable)
     */
    public SDVariable trace(SDVariable in) {
        return trace(null, in);
    }

    /**
     * Matrix trace operation
     * For rank 2 matrices, the output is a scalar vith the trace - i.e., sum of the main diagonal.<br>
     * For higher rank inputs, output[a,b,c] = trace(in[a,b,c,:,:])
     *
     * @param name Name of the output variable. May be null.
     * @param in   Input variable
     * @return Trace
     */
    public SDVariable trace(String name, SDVariable in) {
        validateNumerical("trace", in);
        SDVariable ret = f().trace(in);
        return updateVariableNameAndReference(ret, name);
    }

    /**
     * Boolean XOR (exclusive OR) operation: elementwise (x != 0) XOR (y != 0)<br>
     * If x and y arrays have equal shape, the output shape is the same as these inputs.<br>
     * Note: supports broadcasting if x and y have different shapes and are broadcastable.<br>
     * Returns an array with values 1 where condition is satisfied, or value 0 otherwise.
     *
     * @param x Input 1
     * @param y Input 2
     * @return Output SDVariable with values 0 and 1 based on where the condition is satisfied
     */
    public SDVariable xor(SDVariable x, SDVariable y) {
        return xor(null, x, y);
    }

    /**
     * Boolean XOR (exclusive OR) operation: elementwise (x != 0) XOR (y != 0)<br>
     * If x and y arrays have equal shape, the output shape is the same as these inputs.<br>
     * Note: supports broadcasting if x and y have different shapes and are broadcastable.<br>
     * Returns an array with values 1 where condition is satisfied, or value 0 otherwise.
     *
     * @param name Name of the output variable
     * @param x    Input 1
     * @param y    Input 2
     * @return Output SDVariable with values 0 and 1 based on where the condition is satisfied
     */
    public SDVariable xor(String name, SDVariable x, SDVariable y) {
        validateBool("xor", x, y);
        SDVariable result = f().xor(x, y);
        return updateVariableNameAndReference(result, name);
    }

    /**
     * Full array zero fraction array reduction operation, optionally along specified dimensions: out = (count(x == 0) / length(x))
     *
     * @param input Input variable
     * @return Reduced array of rank 0 (scalar)
     */
    public SDVariable zeroFraction(SDVariable input) {
        return zeroFraction(null, input);
    }

    /**
     * Full array zero fraction array reduction operation, optionally along specified dimensions: out = (count(x == 0) / length(x))
     *
     * @param name  Name of the output variable
     * @param input Input variable
     * @return Reduced array of rank 0 (scalar)
     */
    public SDVariable zeroFraction(String name, SDVariable input) {
        validateNumerical("zeroFraction", input);
        SDVariable res = f().zeroFraction(input);
        return updateVariableNameAndReference(res, name);
    }


}
