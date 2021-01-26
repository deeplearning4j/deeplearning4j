/**
 * Generated using ExtractFromExisting.kt
 */
package org.nd4j.codegen.ops

import org.nd4j.codegen.api.AtLeast
import org.nd4j.codegen.api.DataType
import org.nd4j.codegen.api.DataType.*
import org.nd4j.codegen.api.Language
import org.nd4j.codegen.api.doc.DocScope
import org.nd4j.codegen.dsl.*
import org.nd4j.codegen.mixins.*


fun Math() =  Namespace("Math"){
    Op("abs", transformSame) {
        javaOpClass = "Abs"
        Doc(Language.ANY, DocScope.ALL){
            """
                 Elementwise absolute value operation: out = abs(x)
            """.trimIndent()
        }
    }

    Op("acos", transformStrict) {
        javaOpClass = "ACos"
        Doc(Language.ANY, DocScope.ALL){
            """
                 Elementwise acos (arccosine, inverse cosine) operation: out = arccos(x)
            """.trimIndent()
        }
    }

    Op("acosh", transformStrict) {
        javaOpClass = "ACosh"
        Doc(Language.ANY, DocScope.ALL){
            """
                Elementwise acosh (inverse hyperbolic cosine) function: out = acosh(x)
            """.trimIndent()
        }
    }

    Op("add", transformArithmetic){
        javaOpClass = "AddOp"
        Doc(Language.ANY, DocScope.ALL){
            """ 
                Pairwise addition operation, out = x + y
            """.trimIndent()
        }
        useMixin(broadcastingDoc)
    }

    Op("add", scalar){
        javaOpClass = "ScalarAdd"
        Doc(Language.ANY, DocScope.ALL){
            """ 
                Scalar add operation, out = in + scalar
            """.trimIndent()
        }
    }


    // TODO should we call these "reduceAMax", "reduceAMean", "reduceMin" etc?
    // TODO: There are 2 implementations of amax in org.nd4j.linalg.api.ops.impl
    Op("amax", reduceSame) {
        javaOpClass = "AMax"
        Doc(Language.ANY, DocScope.ALL){
            """
                Absolute max array reduction operation, optionally along specified dimensions: out = max(abs(x))
            """.trimIndent()
        }
    }

    Op("amean", reduceFloating) {
        javaOpClass = "AMean"
        Doc(Language.ANY, DocScope.ALL){
            """
                Absolute mean array reduction operation, optionally along specified dimensions: out = mean(abs(x))
            """.trimIndent()
        }
    }

    // TODO: There are 2 implementations of amax in org.nd4j.linalg.api.ops.impl
    Op("amin", reduceSame) {
        javaOpClass = "AMin"
        Doc(Language.ANY, DocScope.ALL){
            """
                Absolute min array reduction operation, optionally along specified dimensions: out = min(abs(x))
            """.trimIndent()
        }
    }

    Op("and") {
        legacy = true
        javaOpClass = "And"
        javaPackage = "org.nd4j.linalg.api.ops.impl.transforms.pairwise.bool"
        Input(BOOL, "x") { description = "Input 1" }
        Input(BOOL, "y") { description = "Input 2" }
        Output(BOOL, "output"){ description = "%INPUT_TYPE% with values 0 and 1 based on where the condition is satisfied" }
        Doc(Language.ANY, DocScope.ALL){
            """
                 Boolean AND operation: elementwise (x != 0) && (y != 0)
                 If x and y arrays have equal shape, the output shape is the same as these inputs.
                 Note: supports broadcasting if x and y have different shapes and are broadcastable.
                 Returns an array with values 1 where condition is satisfied, or value 0 otherwise.
            """.trimIndent()
        }
    }

    Op("asin", transformStrict) {
        javaOpClass = "ASin"
        Doc(Language.ANY, DocScope.ALL){
            """
                Elementwise asin (arcsin, inverse sine) operation: out = arcsin(x)
            """.trimIndent()
        }
    }

    // TODO: There are 2 implementations
    Op("asinh", transformStrict) {
        javaOpClass = "ASinh"
        Doc(Language.ANY, DocScope.ALL){
            """
                Elementwise asinh (inverse hyperbolic sine) function: out = asinh(x)
            """.trimIndent()
        }
    }

    Op("asum", reduceSame) {
        javaOpClass = "ASum"
        Doc(Language.ANY, DocScope.ALL){
            """
                Absolute sum array reduction operation, optionally along specified dimensions: out = sum(abs(x))
            """.trimIndent()
        }
    }

    Op("atan", transformStrict) {
        javaOpClass = "ATan"
        Doc(Language.ANY, DocScope.ALL){
            """
                Elementwise atan (arctangent, inverse tangent) operation: out = arctangent(x)
            """.trimIndent()
        }
    }

    Op("atan2") {// TODO: We need to generate a constructor that includes SameDiff().
        javaPackage = "org.nd4j.linalg.api.ops.impl.transforms.custom"
        javaOpClass = "ATan2"
        Input(NUMERIC, "y") { description = "Input Y variable" }
        Input(NUMERIC, "x") { description = "Input X variable" }
        Output(NUMERIC, "output"){ description = "Output variable" }
        Doc(Language.ANY, DocScope.ALL){
            """
                Elementwise atan (arctangent, inverse tangent) operation: out = atan2(x,y).
                Similar to atan(y/x) but sigts of x and y are used to determine the location of the result
            """.trimIndent()
        }
    }

    Op("atanh", transformStrict) {
        javaOpClass = "ATanh"
        Doc(Language.ANY, DocScope.ALL){
            """
                Elementwise atanh (inverse hyperbolic tangent) function: out = atanh(x)
            """.trimIndent()
        }
    }

    Op("ceil", transformSame) {
        Doc(Language.ANY, DocScope.ALL){
            """
                Element-wise ceiling function: out = ceil(x).
                Rounds each value up to the nearest integer value (if not already an integer)
            """.trimIndent()
        }
    }

    Op("clipByNorm") {
        javaPackage = "org.nd4j.linalg.api.ops.impl.transforms.clip"
        val x = Input(NUMERIC, "x") { description = "Input variable" }
        val clipValue = Arg(NUMERIC, "clipValue") { description = "Clipping value (maximum l2 norm)" }
        Arg(INT, "dimensions"){ count = AtLeast(0); description = "Dimensions to reduce over. If dimensions are not specified, full array reduction is performed"}  //; defaultValue = intArrayOf(0) }   //TODO
        Output(NUMERIC, "output"){ description = "Output variable" }

//        AllParamSignature(withOutput = false)
//        Signature(x, clipValue)
        Doc(Language.ANY, DocScope.ALL){
            """ 
                Clipping by L2 norm, optionally along dimension(s)
                if l2Norm(x,dimension) < clipValue, then input is returned unmodifed
                Otherwise, out[i] = in[i] * clipValue / l2Norm(in, dimensions) where each value is clipped according
                to the corresponding l2Norm along the specified dimensions
            """.trimIndent()
        }
    }

    Op("clipByValue") {
        javaPackage = "org.nd4j.linalg.api.ops.impl.transforms.clip"
        javaOpClass = "ClipByValue"
        Input(NUMERIC, "x") { description = "Input variable" }
        Arg(NUMERIC, "clipValueMin") { description = "Minimum value for clipping" }
        Arg(NUMERIC, "clipValueMax") { description = "Maximum value for clipping" }
        Output(NUMERIC, "output"){ description = "Output variable" }
        Doc(Language.ANY, DocScope.ALL){
            """
                Element-wise clipping function:
                out[i] = in[i] if in[i] >= clipValueMin and in[i] <= clipValueMax
                out[i] = clipValueMin if in[i] < clipValueMin
                out[i] = clipValueMax if in[i] > clipValueMax
            """.trimIndent()
        }
    }


    Op("ClipByAvgNorm") {
        javaPackage = "org.nd4j.linalg.api.ops.impl.transforms.clip"
        javaOpClass = "ClipByAvgNorm"
        Input(NUMERIC, "x") { description = "Input variable" }
        Arg(NUMERIC, "clipValue") { description = "Value for clipping" }
        Arg(INT, "dimensions"){ count = AtLeast(0); description = "Dimensions to reduce over"}
        Output(NUMERIC, "output"){ description = "Output variable" }
        Doc(Language.ANY, DocScope.ALL){
            """
        Clips tensor values to a maximum average L2-norm.
            """.trimIndent()
        }
    }

    //TODO consolidate these confusionMatrix ops into one?
    Op("confusionMatrix") {
        javaPackage = "org.nd4j.linalg.api.ops.impl.shape"
        Input(NUMERIC, "labels") { description = "Labels - 1D array of integer values representing label values" }
        Input(NUMERIC, "pred") { description = "Predictions - 1D array of integer values representing predictions. Same length as labels" }
        Arg(DATA_TYPE, "dataType") { description = "Data type" }

        Output(NUMERIC, "output"){ description = "variable (2D, shape [numClasses, numClasses})" }

        Doc(Language.ANY, DocScope.ALL){
            """ 
                Compute the 2d confusion matrix of size [numClasses, numClasses] from a pair of labels and predictions, both of
                which are represented as integer values. This version assumes the number of classes is 1 + max(max(labels), max(pred))
                For example, if labels = [0, 1, 1] and predicted = [0, 2, 1] then output is:
                [1, 0, 0]
                [0, 1, 1]
                [0, 0, 0]
            """.trimIndent()
        }
    }

    Op("confusionMatrix") {
        javaPackage = "org.nd4j.linalg.api.ops.impl.shape"
        Input(NUMERIC, "labels") { description = "Labels - 1D array of integer values representing label values" }
        Input(NUMERIC, "pred") { description = "Predictions - 1D array of integer values representing predictions. Same length as labels" }
        Arg(INT, "numClasses") { description = "Number of classes" }
        Output(NUMERIC, "output"){ description = "variable (2D, shape [numClasses, numClasses})" }
        Doc(Language.ANY, DocScope.ALL){
            """
                Compute the 2d confusion matrix of size [numClasses, numClasses] from a pair of labels and predictions, both of
                which are represented as integer values.
                For example, if labels = [0, 1, 1], predicted = [0, 2, 1], and numClasses=4 then output is:
                [1, 0, 0, 0]
                [0, 1, 1, 0]
                [0, 0, 0, 0]
                [0, 0, 0, 0]
            """.trimIndent()
        }
    }

    Op("confusionMatrix") {
        javaPackage = "org.nd4j.linalg.api.ops.impl.shape"
        Input(NUMERIC, "labels") { description = "Labels - 1D array of integer values representing label values" }
        Input(NUMERIC, "pred") { description = "Predictions - 1D array of integer values representing predictions. Same length as labels" }
        Input(NUMERIC, "weights") { description = "Weights - 1D array of values (may be real/decimal) representing the weight/contribution of each prediction. Must be same length as both labels and predictions arrays" }
        Output(NUMERIC, "output"){ description = "variable (2D, shape [numClasses, numClasses})" }
        Doc(Language.ANY, DocScope.ALL){
            """
                Compute the 2d confusion matrix of size [numClasses, numClasses] from a pair of labels and predictions, both of
                which are represented as integer values. This version assumes the number of classes is 1 + max(max(labels), max(pred))
                For example, if labels = [0, 1, 1], predicted = [0, 2, 1] and weights = [1, 2, 3]
                [1, 0, 0]
                [0, 3, 2]
                [0, 0, 0]
            """.trimIndent()
        }
    }

    Op("confusionMatrix") {
        javaPackage = "org.nd4j.linalg.api.ops.impl.shape"
        Input(NUMERIC, "labels") { description = "Labels - 1D array of integer values representing label values" }
        Input(NUMERIC, "pred") { description = "Predictions - 1D array of integer values representing predictions. Same length as labels" }
        Arg(INT, "numClasses") { description = "" }
        Input(NUMERIC, "weights") { description = "Weights - 1D array of values (may be real/decimal) representing the weight/contribution of each prediction. Must be same length as both labels and predictions arrays" }
        Output(NUMERIC, "output"){ description = "Output variable (2D, shape [numClasses, numClasses})" }
        Doc(Language.ANY, DocScope.ALL){
            """
                Compute the 2d confusion matrix of size [numClasses, numClasses] from a pair of labels and predictions, both of
                which are represented as integer values.
                For example, if labels = [0, 1, 1], predicted = [0, 2, 1], numClasses = 4, and weights = [1, 2, 3]
                [1, 0, 0, 0]
                [0, 3, 2, 0]
                [0, 0, 0, 0]
                [0, 0, 0, 0]
            """.trimIndent()
        }
    }

    Op("cos", transformStrict) {
        Doc(Language.ANY, DocScope.ALL){
            """
                Elementwise cosine operation: out = cos(x)
            """.trimIndent()
        }
    }

    Op("cosh", transformStrict) {
        Doc(Language.ANY, DocScope.ALL){
            """
                Elementwise cosh (hyperbolic cosine) operation: out = cosh(x)
            """.trimIndent()
        }
    }

    Op("cosineDistance", reduce3) {
        Doc(Language.ANY, DocScope.ALL){
            """
                Cosine distance reduction operation. The output contains the cosine distance for each
                tensor/subset along the specified dimensions:
                out = 1.0 - cosineSimilarity(x,y)
            """.trimIndent()
        }
    }

    Op("cosineSimilarity", reduce3) {
        Doc(Language.ANY, DocScope.ALL){
            """ 
                Cosine similarity pairwise reduction operation. The output contains the cosine similarity for each tensor/subset
                along the specified dimensions:
                out = (sum_i x[i] * y[i]) / ( sqrt(sum_i x[i]^2) * sqrt(sum_i y[i]^2)
            """.trimIndent()
        }
    }

    Op("countNonZero", reduceLong) {
        Doc(Language.ANY, DocScope.ALL){
            """
                Count non zero array reduction operation, optionally along specified dimensions: out = count(x != 0)
            """.trimIndent()
        }
    }

    Op("countZero", reduceLong) {
        Doc(Language.ANY, DocScope.ALL){
            """
                Count zero array reduction operation, optionally along specified dimensions: out = count(x == 0)
            """.trimIndent()
        }
    }

    Op("cross") {
        javaPackage = "org.nd4j.linalg.api.ops.impl.shape"
        Input(NUMERIC, "a") { description = "First input" }
        Input(NUMERIC, "b") { description = "Second input" }
        Output(NUMERIC, "output"){ description = "Element-wise cross product" }
        Doc(Language.ANY, DocScope.ALL){
            """
                Returns the pair-wise cross product of equal size arrays a and b: a x b = ||a||x||b|| sin(theta).
                Can take rank 1 or above inputs (of equal shapes), but note that the last dimension must have dimension 3
            """.trimIndent()
        }
    }

    Op("cube", transformSame) {
        Doc(Language.ANY, DocScope.ALL){
            """
                Element-wise cube function: out = x^3
            """.trimIndent()
        }
    }

    Op("diag") {
        javaPackage = "org.nd4j.linalg.api.ops.impl.shape"
        Input(NUMERIC, "x") { description = "Input variable" }
        Output(NUMERIC, "output"){ description = "Output variable" }
        Doc(Language.ANY, DocScope.ALL){
            """
                Returns an output variable with diagonal values equal to the specified values; off-diagonal values will be set to 0
                For example, if input = [1,2,3], then output is given by:
                [ 1, 0, 0]
                [ 0, 2, 0]
                [ 0, 0, 3]
                
                Higher input ranks are also supported: if input has shape [a,...,R-1] then output[i,...,k,i,...,k] = input[i,...,k].
                i.e., for input rank R, output has rank 2R
                """.trimIndent()
        }
    }

    Op("diagPart") {
        javaPackage = "org.nd4j.linalg.api.ops.impl.shape"
        Input(NUMERIC, "x") { description = "Input variable" }
        Output(NUMERIC, "output"){ description = "Diagonal part of the input" }
        Doc(Language.ANY, DocScope.ALL){
            """
                Extract the diagonal part from the input array.
                If input is
                [ 1, 0, 0]
                [ 0, 2, 0]
                [ 0, 0, 3]
                then output is [1, 2, 3].
                Supports higher dimensions: in general, out[i,...,k] = in[i,...,k,i,...,k]
                """.trimIndent()
        }
    }

    Op("div", transformArithmetic){
        javaOpClass = "DivOp"
        Doc(Language.ANY, DocScope.ALL){
            """ 
                Pairwise division operation, out = x / y
            """.trimIndent()
        }
        useMixin(broadcastingDoc)
    }

    Op("div", scalar){
        javaOpClass = "ScalarDivision"
        Doc(Language.ANY, DocScope.ALL){
            """ 
                Scalar division operation, out = in / scalar
            """.trimIndent()
        }
    }

    Op("entropy", reduceFloating) {
        Doc(Language.ANY, DocScope.ALL){
            """
                Entropy reduction: -sum(x * log(x))
            """.trimIndent()
        }
    }

    Op("erf", transformStrict) {
        Doc(Language.ANY, DocScope.ALL){
            """ 
                Element-wise Gaussian error function - out = erf(in)
            """.trimIndent()
        }
    }

    Op("erfc", transformStrict) {
        Doc(Language.ANY, DocScope.ALL){
            """
                Element-wise complementary Gaussian error function - out = erfc(in) = 1 - erf(in)
            """.trimIndent()
        }
    }

    Op("euclideanDistance", reduce3) {
        Doc(Language.ANY, DocScope.ALL){
            """
                Euclidean distance (l2 norm, l2 distance) reduction operation. The output contains the Euclidean distance for each
                tensor/subset along the specified dimensions:
                out = sqrt( sum_i (x[i] - y[i])^2 )
                """.trimIndent()
        }
    }

    Op("exp", transformStrict) {
        Doc(Language.ANY, DocScope.ALL){
            """
                Elementwise exponent function: out = exp(x) = 2.71828...^x
            """.trimIndent()
        }
    }

    Op("expm1", transformStrict) {
        Doc(Language.ANY, DocScope.ALL){
            """
                Elementwise 1.0 - exponent function: out = 1.0 - exp(x) = 1.0 - 2.71828...^x
            """.trimIndent()
        }
    }

    //TODO consolidate eye ops into one and use different signatures?
    Op("eye") {
        javaPackage = "org.nd4j.linalg.api.ops.impl.shape"
        Arg(INT, "rows") { description = "Number of rows" }
        Output(NUMERIC, "output"){ description = "Identity matrix" }
        Doc(Language.ANY, DocScope.ALL){
            """
                Generate an identity matrix with the specified number of rows and columns.
            """.trimIndent()
        }
    }

    Op("eye") {
        javaPackage = "org.nd4j.linalg.api.ops.impl.shape"
        Arg(INT, "rows") { description = "Number of rows" }
        Arg(INT, "cols") { description = "Number of columns" }
        Output(NUMERIC, "output"){ description = "" }
        Doc(Language.ANY, DocScope.ALL){
            """
                As per eye(String, int, int, DataType) but with the default datatype, Eye.DEFAULT_DTYPE
            """.trimIndent()
        }
    }

    Op("eye") {
        javaPackage = "org.nd4j.linalg.api.ops.impl.shape"
        Arg(INT, "rows") { description = "Number of rows" }
        Arg(INT, "cols") { description = "Number of columns" }
        Arg(DATA_TYPE, "dataType") { description = "Data type" } //TODO: Mapped DataType to INT.
        Arg(DataType.INT, "dimensions"){ count = AtLeast(0)}
        Output(NUMERIC, "output"){ description = "Identity matrix" }
        Doc(Language.ANY, DocScope.ALL){
            """
                Generate an identity matrix with the specified number of rows and columns
                Example:
                <pre>
                {@code %INPUT_TYPE% eye = eye(3,2)
                eye:
                [ 1, 0]
                [ 0, 1]
                [ 0, 0]}
                </pre>
                """.trimIndent()
        }
    }

    Op("eye") {
        javaPackage = "org.nd4j.linalg.api.ops.impl.shape"
        Input(INT, "rows") { description = "Number of rows" }
        Input(INT, "cols") { description = "Number of columns" }
        Output(NUMERIC, "output"){ description = "Identity matrix" }
        Doc(Language.ANY, DocScope.ALL){
            """
                As per eye(int, int) bit with the number of rows/columns specified as scalar %INPUT_TYPE%s
            """.trimIndent()
        }
    }

    Op("eye") {
        javaPackage = "org.nd4j.linalg.api.ops.impl.shape"
        Input(INT, "rows") { description = "Number of rows" }
        Output(NUMERIC, "output"){ description = "SDVaribable identity matrix" }
        Doc(Language.ANY, DocScope.ALL){
            """
                As per eye(String, int) but with the number of rows specified as a scalar %INPUT_TYPE%
            """.trimIndent()
        }
    }

    Op("firstIndex", indexAccum, keepSignatures=false) {
        var c = Arg(CONDITION, "condition") { description = "Condition to check on input variable" }
        Signature(this.inputs.get(0), c, this.args.get(1))                      //in, condition, dimensions - for vararg
        Signature(this.inputs.get(0), c, this.args.get(0), this.args.get(1))    //in, condition, keepDims, dimensions

        Doc(Language.ANY, DocScope.ALL){
            """
                First index reduction operation.
                Returns a variable that contains the index of the first element that matches the specified condition (for each
                slice along the specified dimensions)
                Note that if keepDims = true, the output variable has the same rank as the input variable,
                with the reduced dimensions having size 1. This can be useful for later broadcast operations (such as subtracting
                the mean along a dimension).
                Example: if input has shape [a,b,c] and dimensions=[1] then output has shape:
                keepDims = true: [a,1,c]
                keepDims = false: [a,c]
            """.trimIndent()
        }
    }

    Op("floor", transformSame) {
        Doc(Language.ANY, DocScope.ALL){
            """
                Element-wise floor function: out = floor(x).
                Rounds each value down to the nearest integer value (if not already an integer)
            """.trimIndent()
        }
    }

    Op("floorDiv", transformArithmetic){
        javaOpClass = "FloorDivOp"
        Doc(Language.ANY, DocScope.ALL){
            """ 
                Pairwise floor division operation, out = floor(x / y)
            """.trimIndent()
        }
        useMixin(broadcastingDoc)
    }

    Op("floorMod", transformArithmetic){
        javaOpClass = "FloorModOp"
        Doc(Language.ANY, DocScope.ALL){
            """ 
                Pairwise Modulus division operation
            """.trimIndent()
        }
        useMixin(broadcastingDoc)
    }

    Op("floorMod", scalar){
        javaOpClass = "ScalarFMod"
        Doc(Language.ANY, DocScope.ALL){
            """ 
                Scalar floor modulus operation
            """.trimIndent()
        }
    }

    Op("hammingDistance", reduce3) {
        Doc(Language.ANY, DocScope.ALL){
            """
                Hamming distance reduction operation. The output contains the cosine distance for each
                tensor/subset along the specified dimensions:
                out = count( x[i] != y[i] )
            """.trimIndent()
        }
    }

    Op("iamax", indexAccumCustom) {
        javaOpClass = "ArgMax"
        //Signature(in, dimensions)
        Doc(Language.ANY, DocScope.ALL){
            """
                Index of the max absolute value: argmax(abs(in))
                see argmax(String, %INPUT_TYPE%, boolean, int...)
            """.trimIndent()
        }
    }

    Op("iamin", indexAccumCustom) {
        javaOpClass = "ArgMin"
        Doc(Language.ANY, DocScope.ALL){
            """
                Index of the min absolute value: argmin(abs(in))
                see argmin(String, %INPUT_TYPE%, boolean, int...)
            """.trimIndent()
        }
    }

    Op("isFinite", transformBool) {
        Doc(Language.ANY, DocScope.ALL){
            """
                Is finite operation: elementwise isFinite(x)
                Returns an array with the same shape/size as the input, with values 1 where condition is satisfied, or
                value 0 otherwise
            """.trimIndent()
        }
    }

    Op("isInfinite", transformBool) {
        javaOpClass = "IsInf"
        Doc(Language.ANY, DocScope.ALL){
            """
                Is infinite operation: elementwise isInfinite(x)
                Returns an array with the same shape/size as the input, with values 1 where condition is satisfied, or
                value 0 otherwise
            """.trimIndent()
        }
    }

    Op("isMax", transformAny) {
        legacy = false
        Doc(Language.ANY, DocScope.ALL){
            """
                Is maximum operation: elementwise x == max(x)
                Returns an array with the same shape/size as the input, with values 1 where condition is satisfied, or
                value 0 otherwise
            """.trimIndent()
        }
    }

    Op("isNaN", transformBool) {
        Doc(Language.ANY, DocScope.ALL){
            """
                Is Not a Number operation: elementwise isNaN(x)
                Returns an array with the same shape/size as the input, with values 1 where condition is satisfied, or
                value 0 otherwise
            """.trimIndent()
        }
    }

    Op("isNonDecreasing") {
        javaPackage = "org.nd4j.linalg.api.ops.impl.transforms.custom"
        Input(NUMERIC, "x") { description = "Input variable" }
        Output(NUMERIC, "output"){ description = "Scalar variable with value 1 if non-decreasing, or 0 otherwise" }
        Doc(Language.ANY, DocScope.ALL){
            """
                Is the array non decreasing?
                An array is non-decreasing if for every valid i, x[i] <= x[i+1]. For Rank 2+ arrays, values are compared
                in 'c' (row major) order
            """.trimIndent()
        }
    }

    Op("isStrictlyIncreasing") {
        javaPackage = "org.nd4j.linalg.api.ops.impl.transforms.custom"
        Input(NUMERIC, "x") { description = "Input variable" }
        Output(NUMERIC, "output"){ description = "Scalar variable with value 1 if strictly increasing, or 0 otherwise" }
        Doc(Language.ANY, DocScope.ALL){
            """
                Is the array strictly increasing?
                An array is strictly increasing if for every valid i, x[i] < x[i+1]. For Rank 2+ arrays, values are compared
                in 'c' (row major) order
            """.trimIndent()
        }
    }

    Op("jaccardDistance", reduce3) {
        Doc(Language.ANY, DocScope.ALL){
            """Jaccard similarity reduction operation. The output contains the Jaccard distance for each
                tensor along the specified dimensions.
            """.trimIndent()
        }
    }

    Op("lastIndex", indexAccum, keepSignatures=false) {
        var c = Arg(CONDITION, "condition") { description = "Condition to check on input variable" }
        Signature(this.inputs.get(0), c, this.args.get(1))                      //in, condition, dimensions - for vararg
        Signature(this.inputs.get(0), c, this.args.get(0), this.args.get(1))    //in, condition, keepDims, dimensions
        Doc(Language.ANY, DocScope.ALL){
            """
                Last index reduction operation.
                Returns a variable that contains the index of the last element that matches the specified condition (for each
                slice along the specified dimensions)
                Note that if keepDims = true, the output variable has the same rank as the input variable,
                with the reduced dimensions having size 1. This can be useful for later broadcast operations (such as subtracting
                the mean along a dimension).
                Example: if input has shape [a,b,c] and dimensions=[1] then output has shape:
                keepDims = true: [a,1,c]
                keepDims = false: [a,c]
            """.trimIndent()
        }
    }

    Op("log", transformStrict) {
        Doc(Language.ANY, DocScope.ALL){
            """
                Element-wise logarithm function (base e - natural logarithm): out = log(x)
            """.trimIndent()
        }
    }

    Op("log", transformStrict) {
        javaOpClass = "LogX"
        javaPackage = "org.nd4j.linalg.api.ops.impl.scalar"
        Arg(NUMERIC, "base") { description = "Logarithm base" }
        Doc(Language.ANY, DocScope.ALL){
            """
                Element-wise logarithm function (with specified base): out = log_{base}(x)
            """.trimIndent()
        }
    }

    Op("log1p", transformStrict) {
        Doc(Language.ANY, DocScope.ALL){
            """
                Elementwise natural logarithm function: out = log_e (1 + x)
            """.trimIndent()
        }
    }

    Op("logEntropy", reduceFloating) {
        Doc(Language.ANY, DocScope.ALL){
            """
                Log entropy reduction: log(-sum(x * log(x)))
            """.trimIndent()
        }
    }

    Op("logSumExp") {
        javaPackage = "org.nd4j.linalg.api.ops.impl.reduce.custom"
        Input(NUMERIC, "input") { description = "Input variable" }
        Arg(INT, "dimensions"){ count = AtLeast(0); description = "Optional dimensions to reduce along" }
        Output(NUMERIC, "output"){ description = "Output variable" }
        Doc(Language.ANY, DocScope.ALL){
            """
                Log-sum-exp reduction (optionally along dimension).
                Computes log(sum(exp(x))
            """.trimIndent()
        }
    }

    Op("manhattanDistance", reduce3) {
        Doc(Language.ANY, DocScope.ALL){
            """
                Manhattan distance (l1 norm, l1 distance) reduction operation. The output contains the Manhattan distance for each
                tensor/subset along the specified dimensions:
                out = sum_i abs(x[i]-y[i])
            """.trimIndent()
        }
    }

    Op("matrixDeterminant") {
        javaPackage = "org.nd4j.linalg.api.ops.impl.transforms.custom"
        Input(NUMERIC, "in") { description = "Input" }
        Output(NUMERIC, "output"){ description = "Matrix determinant variable" }
        Doc(Language.ANY, DocScope.ALL){
            """
                Matrix determinant op. For 2D input, this returns the standard matrix determinant.
                For higher dimensional input with shape [..., m, m] the matrix determinant is returned for each 
                shape [m,m] sub-matrix.
            """.trimIndent()
        }
    }

    Op("matrixInverse") {
        javaPackage = "org.nd4j.linalg.api.ops.impl.transforms.custom"
        Input(NUMERIC, "in") { description = "Input" }
        Output(NUMERIC, "output"){ description = "Matrix inverse variable" }
        Doc(Language.ANY, DocScope.ALL){
            """
                Matrix inverse op. For 2D input, this returns the standard matrix inverse.
                For higher dimensional input with shape [..., m, m] the matrix inverse is returned for each
                shape [m,m] sub-matrix.
            """.trimIndent()
        }
    }

    Op("mergeAdd") {
        javaPackage = "org.nd4j.linalg.api.ops.impl.transforms.pairwise.arithmetic"
        javaOpClass = "MergeAddOp"
        Input(NUMERIC, "inputs"){ count = AtLeast(1); description = "Input variables" }
        Output(NUMERIC, "output"){ description = "Output variable" }
        Doc(Language.ANY, DocScope.ALL){
            """
                Merge add function: merges an arbitrary number of equal shaped arrays using element-wise addition:
                out = sum_i in[i]
            """.trimIndent()
        }
    }

    Op("mergeAvg") {
        javaPackage = "org.nd4j.linalg.api.ops.impl.shape"
        Input(NUMERIC, "inputs"){ count = AtLeast(1); description = "Input variables" }
        Output(NUMERIC, "output"){ description = "Output variable" }
        Doc(Language.ANY, DocScope.ALL){
            """
                Merge average function: merges an arbitrary number of equal shaped arrays using element-wise mean operation:
                out = mean_i in[i]
            """.trimIndent()
        }
    }

    Op("mergeMax") {
        javaPackage = "org.nd4j.linalg.api.ops.impl.shape"
        Input(NUMERIC, "inputs"){ count = AtLeast(1); description = "Input variables" }
        Output(NUMERIC, "output"){ description = "Output variable" }
        Doc(Language.ANY, DocScope.ALL){
            """
                Merge max function: merges an arbitrary number of equal shaped arrays using element-wise maximum operation:
                out = max_i in[i]
            """.trimIndent()
        }
    }

    Op("mod", transformArithmetic){
        javaOpClass = "ModOp"
        Doc(Language.ANY, DocScope.ALL){
            """ 
                Pairwise modulus (remainder) operation, out = x % y
            """.trimIndent()
        }
        useMixin(broadcastingDoc)
    }

    Op("moments") {
        javaPackage = "org.nd4j.linalg.api.ops.impl.reduce"
        Input(NUMERIC, "input") { description = "Input to calculate moments for" }
        Arg(INT, "axes"){ count = AtLeast(0); description = "Dimensions to perform calculation over" }
        Output(NUMERIC, "output_mean"){ description = "Mean variable" }
        Output(NUMERIC, "output_variance"){ description = "Variance variable" }
        Doc(Language.ANY, DocScope.ALL){
            """
                Calculate the mean and (population) variance for the input variable, for the specified axis
            """.trimIndent()
        }
    }

    Op("neg", transformSame) {
        javaOpClass = "Negative"
        Doc(Language.ANY, DocScope.ALL){
            """
                Elementwise negative operation: out = -x
            """.trimIndent()
        }
    }

    Op("normalizeMoments") {
        javaPackage = "org.nd4j.linalg.api.ops.impl.reduce"
        Input(NUMERIC, "counts") { description = "Rank 0 (scalar) value with the total number of values used to calculate the sufficient statistics" }
        Input(NUMERIC, "means") { description = "Mean-value sufficient statistics: this is the SUM of all data values" }
        Input(NUMERIC, "variances") { description = "Variaance sufficient statistics: this is the squared sum of all data values" }
        Arg(NUMERIC, "shift") { description = "Shift value, possibly 0, used when calculating the sufficient statistics (for numerical stability)" }
        Output(NUMERIC, "output_mean"){ description = "Mean variable" }
        Output(NUMERIC, "output_population"){ description = "Population variable" }
        Doc(Language.ANY, DocScope.ALL){
            """
                Calculate the mean and variance from the sufficient statistics
            """.trimIndent()
        }
    }

    Op("or") {
        javaPackage = "org.nd4j.linalg.api.ops.impl.transforms.pairwise.bool"
        Input(BOOL, "x") { description = "Input 1" }
        Input(BOOL, "y") { description = "Input 2" }
        Output(BOOL, "output"){ description = "%INPUT_TYPE% with values 0 and 1 based on where the condition is satisfied" }
        legacy = true
        Doc(Language.ANY, DocScope.ALL){
            """
                Boolean OR operation: elementwise (x != 0) || (y != 0)
                If x and y arrays have equal shape, the output shape is the same as these inputs.
                Note: supports broadcasting if x and y have different shapes and are broadcastable.
                Returns an array with values 1 where condition is satisfied, or value 0 otherwise.
            """.trimIndent()
        }
    }

    Op("pow", scalar) {
        Doc(Language.ANY, DocScope.ALL){
            """
                Element-wise power function: out = x^value
            """.trimIndent()
        }
    }

    Op("pow") {
        javaPackage = "org.nd4j.linalg.api.ops.impl.transforms.custom"
        Input(NUMERIC, "x") { description = "Input variable" }
        Input(NUMERIC, "y") { description = "Power" }
        Output(NUMERIC, "output"){ description = "Output variable" }
        Doc(Language.ANY, DocScope.ALL){
            """
                Element-wise (broadcastable) power function: out = x[i]^y[i]
            """.trimIndent()
        }
    }

    Op("rationalTanh", transformStrict) {
        Doc(Language.ANY, DocScope.ALL){
            """
                Rational Tanh Approximation elementwise function, as described in the paper:
                Compact Convolutional Neural Network Cascade for Face Detection
                This is a faster Tanh approximation
            """.trimIndent()
        }
    }

    Op("rectifiedTanh", transformStrict) {
        Doc(Language.ANY, DocScope.ALL){
            """
                Rectified tanh operation: max(0, tanh(in))
            """.trimIndent()
        }
    }

    Op("reciprocal", transformSame) {
        Doc(Language.ANY, DocScope.ALL){
            """
                Element-wise reciprocal (inverse) function: out[i] = 1 / in[i]
            """.trimIndent()
        }
    }

    Op("rdiv", transformArithmetic){
        javaOpClass = "RDivOp"
        Doc(Language.ANY, DocScope.ALL){
            """ 
                Pairwise reverse division operation, out = y / x
            """.trimIndent()
        }
        useMixin(broadcastingDoc)
    }

    Op("rdiv", scalar){
        javaOpClass = "ScalarReverseDivision"
        Doc(Language.ANY, DocScope.ALL){
            """ 
                Scalar reverse division operation, out = scalar / in
            """.trimIndent()
        }
    }

    Op("round", transformSame) {
        Doc(Language.ANY, DocScope.ALL){
            """
                Element-wise round function: out = round(x).
                Rounds (up or down depending on value) to the nearest integer value.
            """.trimIndent()
        }
    }

    Op("rsqrt", transformFloating) {
        javaOpClass = "RSqrt"
        Doc(Language.ANY, DocScope.ALL){
            """
                Element-wise reciprocal (inverse) of square root: out = 1.0 / sqrt(x)
            """.trimIndent()
        }
    }

    Op("rsub", transformArithmetic){
        javaOpClass = "RSubOp"
        Doc(Language.ANY, DocScope.ALL){
            """ 
                Pairwise reverse subtraction operation, out = y - x
            """.trimIndent()
        }
        useMixin(broadcastingDoc)
    }

    Op("rsub", scalar){
        javaOpClass = "ScalarReverseSubtraction"
        Doc(Language.ANY, DocScope.ALL){
            """ 
                Scalar reverse subtraction operation, out = scalar - in
            """.trimIndent()
        }
    }


    Op("setDiag") {
        javaPackage = "org.nd4j.linalg.api.ops.impl.transforms.custom"
        javaOpClass = "MatrixSetDiag"
        Input(NUMERIC, "in") { description = "Input variable" }
        Input(NUMERIC, "diag") { description = "Diagonal" }
        Output(NUMERIC, "output"){ description = "Output variable" }
        Doc(Language.ANY, DocScope.ALL){
            """
                Set the diagonal value to the specified values
                If input is
                [ a, b, c]
                [ d, e, f]
                [ g, h, i]
                and diag = [ 1, 2, 3] then output is
                [ 1, b, c]
                [ d, 2, f]
                [ g, h, 3]
            """.trimIndent()
        }
    }

    Op("shannonEntropy", reduceFloating) {
        Doc(Language.ANY, DocScope.ALL){
            """
                Shannon Entropy reduction: -sum(x * log2(x))
            """.trimIndent()
        }
    }

    Op("sign", transformSame) {
        Doc(Language.ANY, DocScope.ALL){
            """
                Element-wise sign (signum) function:
                out = -1 if in < 0
                out = 0 if in = 0
                out = 1 if in > 0
            """.trimIndent()
        }
    }

    Op("sin", transformStrict) {
        Doc(Language.ANY, DocScope.ALL){
            """
                Elementwise sine operation: out = sin(x)
            """.trimIndent()
        }
    }

    Op("sinh", transformStrict) {
        Doc(Language.ANY, DocScope.ALL){
            """
                Elementwise sinh (hyperbolic sine) operation: out = sinh(x)
            """.trimIndent()
        }
    }

    Op("sqrt", transformFloating) {
        Doc(Language.ANY, DocScope.ALL){
            """
                Element-wise square root function: out = sqrt(x)
            """.trimIndent()
        }
    }

    Op("square", transformSame) {
        Doc(Language.ANY, DocScope.ALL){
            """
                Element-wise square function: out = x^2
            """.trimIndent()
        }
    }

    Op("squaredDifference", transformArithmetic) {
        javaOpClass = "SquaredDifferenceOp"
        Doc(Language.ANY, DocScope.ALL){
            """
                Pairwise squared difference operation.
            """.trimIndent()
        }
        useMixin(broadcastingDoc)
    }

    Op("step", scalar) {
        Doc(Language.ANY, DocScope.ALL){
            """
                Elementwise step function:
                out(x) = 1 if x >= cutoff
                out(x) = 0 otherwise
            """.trimIndent()
        }
    }

    Op("standardize") {
        javaPackage = "org.nd4j.linalg.api.ops.impl.transforms.custom"
        Input(NUMERIC, "x") { description = "Input variable" }
        Arg(INT, "dimensions"){ count = AtLeast(1); description = "" } //TODO: Missing description for dimension.
        Output(NUMERIC, "output"){ description = "Output variable" }
        Doc(Language.ANY, DocScope.ALL){
            """
                Standardize input variable along given axis
                <p>
                out = (x - mean) / stdev
                <p>
                with mean and stdev being calculated along the given dimension.
                <p>
                For example: given x as a mini batch of the shape [numExamples, exampleLength]:
                <ul> 
                <li>use dimension 1 too use the statistics (mean, stdev) for each example</li>
                <li>use dimension 0 if you want to use the statistics for each column across all examples</li>
                <li>use dimensions 0,1 if you want to use the statistics across all columns and examples</li>
                </ul>
            """.trimIndent()
        }
    }

    Op("sub", transformArithmetic){
        javaOpClass = "SubOp"
        Doc(Language.ANY, DocScope.ALL){
            """ 
                Pairwise subtraction operation, out = x - y
            """.trimIndent()
        }
        useMixin(broadcastingDoc)
    }

    Op("sub", scalar){
        javaOpClass = "ScalarSubtraction"
        Doc(Language.ANY, DocScope.ALL){
            """ 
                Scalar subtraction operation, out = in - scalar
            """.trimIndent()
        }
    }

    Op("tan", transformStrict) {
        Doc(Language.ANY, DocScope.ALL){
            """
                Elementwise tangent operation: out = tan(x)
            """.trimIndent()
        }
    }

    Op("tanh", transformStrict) {
        Doc(Language.ANY, DocScope.ALL){
            """
                Elementwise tanh (hyperbolic tangent) operation: out = tanh(x)
            """.trimIndent()
        }
    }

    Op("trace") {
        javaPackage = "org.nd4j.linalg.api.ops.impl.transforms.custom"
        Input(NUMERIC, "in") { description = "Input variable" }
        Output(NUMERIC, "output"){ description = "Trace" }
        Doc(Language.ANY, DocScope.ALL){
            """
                Matrix trace operation
                For rank 2 matrices, the output is a scalar vith the trace - i.e., sum of the main diagonal.
                For higher rank inputs, output[a,b,c] = trace(in[a,b,c,:,:])
            """.trimIndent()
        }
    }

    Op("xor") {
        javaPackage = "org.nd4j.linalg.api.ops.impl.transforms.pairwise.bool"
        legacy = true
        Input(BOOL, "x") { description = "Input 1" }
        Input(BOOL, "y") { description = "Input 2" }
        Output(BOOL, "output"){ description = "%INPUT_TYPE% with values 0 and 1 based on where the condition is satisfied" }
        Doc(Language.ANY, DocScope.ALL){
            """
                Boolean XOR (exclusive OR) operation: elementwise (x != 0) XOR (y != 0)
                If x and y arrays have equal shape, the output shape is the same as these inputs.
                Note: supports broadcasting if x and y have different shapes and are broadcastable.
                Returns an array with values 1 where condition is satisfied, or value 0 otherwise.
            """.trimIndent()
        }
    }

    Op("zeroFraction") {
        javaPackage = "org.nd4j.linalg.api.ops.impl.reduce"
        Input(NUMERIC, "input") { description = "Input variable" }
        Output(NUMERIC, "output"){ description = "Reduced array of rank 0 (scalar)" }
        Doc(Language.ANY, DocScope.ALL){
            """
                Full array zero fraction array reduction operation, optionally along specified dimensions: out = (count(x == 0) / length(x))
            """.trimIndent()
        }
    }

    Op("listDiff") {
        javaPackage = "org.nd4j.linalg.api.ops.impl.transforms.custom"
        Input(NUMERIC, "x") { description = "Input variable X" }
        Input(NUMERIC, "y") { description = "Input variable Y" }
        Output(NUMERIC, "output1"){ description = "Calculated difference between X and Y" }
        Output(NUMERIC, "output2"){ description = "Calculated difference between X and Y" }
        Doc(Language.ANY, DocScope.ALL){
            """
                Calculates difference between inputs X and Y.
            """.trimIndent()
        }
    }

    Op("max", transformCustom2){
        javaOpClass = "Max"
        Doc(Language.ANY, DocScope.ALL){
            """ 
                Pairwise max operation, out = max(x, y)
            """.trimIndent()
        }
        useMixin(broadcastingDoc)
    }

    Op("meshgrid") {
        javaPackage = "org.nd4j.linalg.api.ops.impl.shape"
        javaOpClass = "MeshGrid"
        Arg(BOOL, "cartesian")
        Input(NUMERIC, "inputs") { count = AtLeast(0) }

        Output(NUMERIC, "output1"){ description = "Output array" }
        Output(NUMERIC, "output2"){ description = "Output array" }

        Doc(Language.ANY, DocScope.ALL){
            """
                Broadcasts parameters for evaluation on an N-D grid.
            """.trimIndent()
        }
    }

    Op("min", transformCustom2){
        javaOpClass = "Min"
        Doc(Language.ANY, DocScope.ALL){
            """ 
                Pairwise max operation, out = min(x, y)
            """.trimIndent()
        }
        useMixin(broadcastingDoc)
    }

    Op("mul", transformArithmetic){
        javaOpClass = "MulOp"
        Doc(Language.ANY, DocScope.ALL){
            """ 
                Pairwise multiplication operation, out = x * y
            """.trimIndent()
        }
        useMixin(broadcastingDoc)
    }

    Op("mul", scalar){
        javaOpClass = "ScalarMultiplication"
        Doc(Language.ANY, DocScope.ALL){
            """ 
                Scalar multiplication operation, out = in * scalar
            """.trimIndent()
        }
    }

    Op("bitShift") {
        javaPackage = "org.nd4j.linalg.api.ops.impl.transforms.custom"
        javaOpClass = "ShiftBits"
        Input(NUMERIC, "x") {description = "input"}
        Input(NUMERIC, "shift") {description = "shift value"}
        Output(NUMERIC, "output") {description = "shifted output"}

        Doc(Language.ANY, DocScope.ALL){
            """
                Bit shift operation
            """.trimIndent()
        }
    }

    Op("bitShiftRight") {
        javaPackage = "org.nd4j.linalg.api.ops.impl.transforms.custom"
        javaOpClass = "RShiftBits"
        Input(NUMERIC, "x") {description = "Input tensor"}
        Input(NUMERIC, "shift") {description = "shift argument"}
        Output(NUMERIC, "output") {description = "shifted output"}

        Doc(Language.ANY, DocScope.ALL){
            """
                Right bit shift operation
            """.trimIndent()
        }
    }

    Op("bitShiftRotl") {
        javaPackage = "org.nd4j.linalg.api.ops.impl.transforms.custom"
        javaOpClass = "CyclicShiftBits"
        Input(NUMERIC, "x") {description = "Input tensor"}
        Input(NUMERIC, "shift") {description = "shift argy=ument"}
        Output(NUMERIC, "output") {description = "shifted output"}

        Doc(Language.ANY, DocScope.ALL){
            """
                Cyclic bit shift operation
            """.trimIndent()
        }
    }

    Op("bitShiftRotr") {
        javaPackage = "org.nd4j.linalg.api.ops.impl.transforms.custom"
        javaOpClass = "CyclicRShiftBits"
        Input(NUMERIC, "x") {description = "Input tensor"}
        Input(NUMERIC, "shift") {description = "Shift argument"}
        Output(NUMERIC, "output") {description = "Shifted output"}

        Doc(Language.ANY, DocScope.ALL){
            """
                Cyclic right shift operation
            """.trimIndent()
        }
    }

    Op("EmbeddingLookup") {
        javaPackage = "org.nd4j.linalg.api.ops.impl.shape.tensorops"
        javaOpClass = "EmbeddingLookup"
        Input(NUMERIC, "x") {description = "Input tensor"}
        Input(INT, "indices") {description = "A Tensor containing the ids to be looked up."}
        Arg(ENUM, "PartitionMode") { possibleValues = listOf( "MOD","DIV"); description ="partition_mode == 0 - i.e. 'mod' , 1 - 'div'"}
        Output(NUMERIC, "output") {description = "Shifted output"}

        Doc(Language.ANY, DocScope.ALL){
            """
            Looks up ids in a list of embedding tensors.

            """.trimIndent()
        }
    }

    Op("MergeMaxIndex") {
        javaPackage = "org.nd4j.linalg.api.ops.impl.shape"
        javaOpClass = "MergeMaxIndex"
        Input(NUMERIC, "x") {count = AtLeast(1); description = "Input tensor"}
        Arg(DATA_TYPE, "dataType") { description = "Data type"; defaultValue = org.nd4j.linalg.api.buffer.DataType.INT }
        Output(INT, "output") {description = "Array max elements indices with along dimensions."}

        Doc(Language.ANY, DocScope.ALL){
            """
            Return array of max elements indices with along tensor dimensions 
            """.trimIndent()
        }
    }
}
