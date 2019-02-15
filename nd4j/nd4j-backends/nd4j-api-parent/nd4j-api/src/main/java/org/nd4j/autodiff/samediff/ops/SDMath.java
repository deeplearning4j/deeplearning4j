package org.nd4j.autodiff.samediff.ops;

import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ops.impl.shape.ConfusionMatrix;
import org.nd4j.linalg.indexing.conditions.Condition;

public class SDMath extends SDOps {
    public SDMath(SameDiff sameDiff) {
        super(sameDiff);
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
        SDVariable result = f().acosh(x);
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
        SDVariable result = f().asinh(x);
        return updateVariableNameAndReference(result, name);
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
        SDVariable result = f().confusionMatrix(labels, pred, numClasses, weights);
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
        SDVariable result = f().cosh(x);
        return updateVariableNameAndReference(result, name);
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
        SDVariable ret = f().erfc(x);
        return updateVariableNameAndReference(ret, name);
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
        SDVariable result = f().expm1(x);
        return updateVariableNameAndReference(result, name);
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
        SDVariable result = f().floor(x);
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
        SDVariable result = f().isStrictlyIncreasing(x);
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
        SDVariable ret = f().logEntropy(in, dimensions);
        return updateVariableNameAndReference(ret, name);
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
        SDVariable result = f().sign(x);
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
        SDVariable ret = f().step(in, cutoff);
        return updateVariableNameAndReference(ret, name);
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
        SDVariable result = f().tanh(x);
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
        SDVariable res = f().zeroFraction(input);
        return updateVariableNameAndReference(res, name);
    }

}
