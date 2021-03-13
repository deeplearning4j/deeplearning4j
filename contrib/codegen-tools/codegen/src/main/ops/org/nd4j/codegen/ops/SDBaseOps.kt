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

import org.nd4j.codegen.api.AtLeast
import org.nd4j.codegen.api.Language
import org.nd4j.codegen.api.doc.DocScope
import org.nd4j.codegen.dsl.*
import org.nd4j.codegen.api.DataType.*
import org.nd4j.codegen.mixins.*
import java.lang.Boolean.FALSE

fun SDBaseOps() =  Namespace("BaseOps"){

    val keepDimsDoc = Mixin("keepDims"){
        Doc(Language.ANY, DocScope.ALL){
            """
                Note that if keepDims = true, the output variable has the same rank as the input variable,
                with the reduced dimensions having size 1. This can be useful for later broadcast operations (such as subtracting
                the mean along a dimension).
                Example: if input has shape [a,b,c] and dimensions=[1] then output has shape:
                keepDims = true: [a,1,c]
                keepDims = false: [a,c]
            """.trimIndent()
        }
    }

    val booleanReturnDoc = Mixin("booleanReturnDoc"){
        Doc(Language.ANY, DocScope.ALL) {
            """
                Return boolean array with values true where satisfied, or false otherwise.
            """.trimIndent()
        }
    }

    val scatterOp = Mixin("scatterOp "){
        javaPackage = "org.nd4j.linalg.api.ops.impl.scatter"
        Input(NUMERIC, "ref") { description = "Initial/source variable" }
        Input(NUMERIC, "indices") { description = "Indices array" }
        Input(NUMERIC, "updates") { description = "Updates to add to the initial/source array" }
        Output(NUMERIC, "output"){ description = "The updated variable" }
    }

    val scatterDoc = Mixin("scatterDoc "){
        Doc(Language.ANY, DocScope.ALL) {
            """
                If indices is rank 0 (a scalar), then out[index, ...] = out[index, ...] + op(updates[...])
                If indices is rank 1 (a vector), then for each position i, out[indices[i], ...] = out[indices[i], ...] + op(updates[i, ...])
                If indices is rank 2+, then for each position (i,...,k), out[indices[i], ..., indices[k], ...] = out[indices[i], ..., indices[k], ...]  + op(updates[i, ..., k, ...]) 
                Note that if multiple indices refer to the same location, the contributions from each is handled correctly. 
            """.trimIndent()
        }
    }

    val segmentOp = Mixin("segmentOp"){
        javaPackage = "org.nd4j.linalg.api.ops.impl.transforms.custom.segment"
        Input(NDARRAY, "data") { description = "Data to perform segment max on" }
        Input(NUMERIC, "segmentIds") { description = "Variable for the segment IDs" }
        Output(NUMERIC, "output"){ description = "Segment output" }
    }

    val segmentDoc = Mixin("segmentDoc") {
        Doc(Language.ANY, DocScope.ALL) {
            """
                If data =     [3, 6, 1, 4, 9, 2, 8]
                segmentIds =  [0, 0, 1, 1, 1, 2, 2]
                then output = [6, 9, 8] = [op(3,6), op(1,4,9), op(2,8)]
                Note that the segment IDs must be sorted from smallest to largest segment.
                See {unsortedSegment (String, SDVariable, SDVariable, int) ops
                for the same op without this sorted requirement
            """.trimIndent()
        }
    }

    val unsortedSegmentOp = Mixin("unsortedSegmentOp") {
        javaPackage = "org.nd4j.linalg.api.ops.impl.transforms.segment"
        Input(NUMERIC, "data") { description = "Data (variable) to perform unsorted segment max on" }
        Input(NUMERIC, "segmentIds") { description = "Variable for the segment IDs" }
        Arg(INT, "numSegments") { description = "Number of segments" }
        Output(NUMERIC, "output") { description = "Unsorted segment output" }
    }

    Op("argmax") {
        javaPackage = "org.nd4j.linalg.api.ops.impl.indexaccum.custom"
        javaOpClass = "ArgMax"
        Input(NUMERIC, "in") { description = "Input variable" }
        Arg(BOOL, "keepDims") { description = "If true: keep the dimensions that are reduced on (as size 1). False: remove the reduction dimensions"; defaultValue = false }
        Arg(INT, "dimensions"){ count = AtLeast(0); description = "Dimensions to reduce over. If dimensions are not specified, full array reduction is performed" }

        Output(NUMERIC, "output"){ description = "reduced array of rank (input rank - num dimensions) if keepDims = false, or\n" +
                " of rank (input rank) if keepdims = true" }
        Doc(Language.ANY, DocScope.ALL){
            """
                Argmax array reduction operation, optionally along specified dimensions.
                Output values are the index of the maximum value of each slice along the specified dimension.
            """.trimIndent()
        }
        useMixin(keepDimsDoc)
    }

    Op("argmin") {
        javaPackage = "org.nd4j.linalg.api.ops.impl.indexaccum.custom"
        javaOpClass = "ArgMin"
        Input(NUMERIC, "in") { description = "Input variable" }
        Arg(BOOL, "keepDims") { description = "If true: keep the dimensions that are reduced on (as size 1). False: remove the reduction dimensions"; defaultValue = false }
        Arg(INT, "dimensions"){ count = AtLeast(0); description = "Dimensions to reduce over. If dimensions are not specified, full array reduction is performed" }
        Output(NUMERIC, "output"){ description = "reduced array of rank (input rank - num dimensions) if keepDims = false, or of rank (input rank) if keepdims = true" }
        Doc(Language.ANY, DocScope.ALL){
            """
                Argmin array reduction operation, optionally along specified dimensions.
                Output values are the index of the minimum value of each slice along the specified dimension.
            """.trimIndent()
        }
        useMixin(keepDimsDoc)
        useMixin(broadcastingDoc)
    }

    Op("concat") {
        javaPackage = "org.nd4j.linalg.api.ops.impl.shape"
        javaOpClass = "Concat"
        argsFirst = true
        Arg(INT, "dimension"){ description = "Dimension to concatenate on" }
        val inputs = Input(NUMERIC, "inputs") {count = AtLeast(1); description = "Input variables" }
        Output(NUMERIC, "output"){ description = "" }
        Constraint("Input arrays must all be the same datatype"){ sameType(inputs) } //TODO: Fix, generates error in java,
        Doc(Language.ANY, DocScope.ALL){
            """
                Concatenate a set of inputs along the specified dimension.
                Note that inputs must have identical rank and identical dimensions, other than the dimension to stack on.
                For example, if 2 inputs have shape [a, x, c] and [a, y, c] and dimension = 1, then the output has shape [a, x+y, c]
            """.trimIndent()
        }
    }

    Op("cumprod") {
        javaPackage = "org.nd4j.linalg.api.ops.impl.transforms.custom"
        javaOpClass = "CumProd"
        Input(NUMERIC, "in") { description = "Input variable" }
        Arg(BOOL, "exclusive") { description = "If true: exclude the first value"; defaultValue = FALSE }
        Arg(BOOL, "reverse") { description = "If true: reverse the direction of the accumulation"; defaultValue = FALSE }
        Arg(INT, "axis") { count = AtLeast(1); description = "Scalar axis argument for dimension to perform cumululative sum operations along" }
        Output(NUMERIC, "output"){ description = "Output variable" }
        Doc(Language.ANY, DocScope.ALL){
            """
                Cumulative product operation.
                For input: [ a, b, c], output is:
                exclusive=false, reverse=false: [a, a*b, a*b*c]
                exclusive=true, reverse=false, [0, a, a*b]
                exclusive=false, reverse=true: [a*b*c, b*c, c]
                exclusive=true, reverse=true: [b*c, c, 0]
            """.trimIndent()
        }
    }

    Op("cumsum") {
        javaPackage = "org.nd4j.linalg.api.ops.impl.transforms.custom"
        javaOpClass = "CumSum"
        Input(NUMERIC, "in") { description = "Input variable" }
        Arg(BOOL, "exclusive") { description = "If true: exclude the first value"; defaultValue = FALSE }
        Arg(BOOL,  "reverse") { description = "If true: reverse the direction of the accumulation"; defaultValue = FALSE  }
        Arg(INT, "axis") { count = AtLeast(1); description = "Scalar axis argument for dimension to perform cumululative sum operations along" }
        Output(NUMERIC, "output"){ description = "" }

        Doc(Language.ANY, DocScope.ALL){
            """
                Cumulative sum operation.
                For input: [ a, b, c], output is:
                exclusive=false, reverse=false: [a, a+b, a+b+c]
                exclusive=true, reverse=false, [0, a, a+b]
                exclusive=false, reverse=true: [a+b+c, b+c, c]
                exclusive=true, reverse=true: [b+c, c, 0]
            """.trimIndent()
        }
    }

    Op("dot") {
        javaPackage = "org.nd4j.linalg.api.ops.impl.reduce3"
        javaOpClass = "Dot"
        legacy = true
        Input(NUMERIC, "x") { description = "first input" }
        Input(NUMERIC, "y") { description = "second input" }
        Arg(INT, "dimensions") {count = AtLeast(0); description = "Dimensions to reduce over. If dimensions are not specified, full array reduction is performed" }
        Output(NUMERIC, "output"){ description = "output variable" }
        Doc(Language.ANY, DocScope.ALL){
            """
                Pairwise dot product reduction along dimension
                output = sum(i=0 ... size(dim)-1) x[i] * y[i]
            """.trimIndent()
        }
    }

    Op("dynamicPartition") {
        javaPackage = "org.nd4j.linalg.api.ops.impl.transforms.custom"
        javaOpClass = "DynamicPartition"
        Input(NUMERIC, "x") { description = "Input variable" }
        Input(INT, "partitions") { description = "1D input with values 0 to numPartitions-1" }
        Arg(INT, "numPartitions") { description = "Number of partitions, >= 1" }
        Output(NUMERIC, "output"){ multiOutput=true; description = "Output variables (equal in number to numPartitions)" }

        Doc(Language.ANY, DocScope.ALL){
            """
                Dynamically partition the input variable values into the specified number of paritions, using the indices.
                Example:
                <pre>
                input = [1,2,3,4,5]
                numPartitions = 2
                partitions = [1,0,0,1,0]
                out[0] = [2,3,5]
                out[1] = [1,4] }
                </pre>
            """.trimIndent()
        }
    }

    Op("dynamicStitch") {
        javaPackage = "org.nd4j.linalg.api.ops.impl.transforms.custom"
        javaOpClass = "DynamicStitch"
        Input(INT, "indices") {count = AtLeast(1); description = "Indices to use when merging. Must be >= 1, same length as input variables" }
        Input(NUMERIC, "x") { count = AtLeast(1); description = "Input variables." }
        Output(NUMERIC, "output"){ description = "Merged output variable" }

        Doc(Language.ANY, DocScope.ALL){
            """
                Dynamically merge the specified input arrays into a single array, using the specified indices
            """.trimIndent()
        }
    }

    Op("eq") {
        javaPackage = "org.nd4j.linalg.api.ops.impl.scalar.comparison"
        javaOpClass = "ScalarEquals"
        legacy = true
        Input(NUMERIC, "x") { description = "Input array" }
        Arg(NUMERIC, "y") { description = "Double value argument to use in operation" }
        Output(NUMERIC, "output"){ description = "Boolean array out, with values true/false based on where the condition is satisfied" }

        Doc(Language.ANY, DocScope.ALL){
            """
                Equals operation: elementwise x == y
            """.trimIndent()
        }
        useMixin(booleanReturnDoc)
    }

    Op("eq") {
        javaPackage = "org.nd4j.linalg.api.ops.impl.transforms.custom"
        javaOpClass = "EqualTo"
        Input(NUMERIC, "x") { description = "Input 1" }
        Input(NUMERIC, "y") { description = "Input 2" }
        Output(NUMERIC, "output"){ description = "Boolean array out, with values true/false based on where the condition is satisfied" }
        Doc(Language.ANY, DocScope.ALL){
            """
                Equal to operation: elementwise x == y
                If x and y arrays have equal shape, the output shape is the same as these inputs.
            """.trimIndent()
        }
        useMixin(broadcastingDoc)
        useMixin(booleanReturnDoc)
    }

    Op("expandDims") {
        javaPackage = "org.nd4j.linalg.api.ops.impl.shape"
        javaOpClass = "ExpandDims"
        Input(NDARRAY, "x") { description = "Input variable" }
        Arg(INT, "axis") { description = "Axis to expand" }
        Output(NUMERIC, "output"){ description = "Output variable" }
        Doc(Language.ANY, DocScope.ALL){
            """
                Reshape the input by adding a 1 at the specified location.
                For example, if input has shape [a, b], then output shape is:
                axis = 0: [1, a, b]
                axis = 1: [a, 1, b]
                axis = 2: [a, b, 1]
            """.trimIndent()
        }
    }

    Op("fill") {
        javaPackage = "org.nd4j.linalg.api.ops.impl.transforms.custom"
        Input(INT, "shape") { description = "Shape: must be a 1D array/variable" }
        Arg(DATA_TYPE, "dataType") { description = "Datatype of the output array" }
        Arg(NUMERIC, "value") { description = "Value to set all elements to" }
        Output(NUMERIC, "output"){ description = "Output variable" }
        Doc(Language.ANY, DocScope.ALL){
            """
                Generate an output variable with the specified (dynamic) shape with all elements set to the specified value
            """.trimIndent()
        }
    }

    Op("gather") {
        javaPackage = "org.nd4j.linalg.api.ops.impl.shape"
        Input(NUMERIC, "df") { description = "Input variable" }
        Arg(INT, "indices") { count = AtLeast(1); description = "Indices to get" }
        Arg(INT, "axis") { description = "Axis that the indices refer to" }
        Output(NUMERIC, "output"){ description = "Output variable with slices pulled from the specified axis" }
        Doc(Language.ANY, DocScope.ALL){
            """
                Gather slices from the input variable where the indices are specified as fixed int[] values.
                Output shape is same as input shape, except for axis dimension, which has size equal to indices.length.
            """.trimIndent()
        }
    }

    Op("gather") {
        javaPackage = "org.nd4j.linalg.api.ops.impl.shape"
        Input(NUMERIC, "df") { description = "Input variable" }
        Input(INT, "indices") { description = "Indices to get slices for. Rank 0 or 1 input" }
        Arg(INT, "axis") { description = "Axis that the indices refer to" }
        Output(NUMERIC, "output"){ description = "Output variable with slices pulled from the specified axis" }
        Doc(Language.ANY, DocScope.ALL){
            """
                Gather slices from the input variable where the indices are specified as dynamic array values.
                Output shape is same as input shape, except for axis dimension, which has size equal to indices.length.
            """.trimIndent()
        }
    }

    Op("gatherNd") {
        javaPackage = "org.nd4j.linalg.api.ops.impl.shape"
        javaOpClass = "GatherNd"
        Input(NUMERIC, "df") {description = "" }
        Input(NUMERIC, "indices") {description = "" }
        Output(NUMERIC, "output"){ description = "" }
        Doc(Language.ANY, DocScope.ALL){
            """
               Gather slices from df with shape specified by indices. 
            """.trimIndent()
        }
    }

    Op("gt") {
        javaPackage = "org.nd4j.linalg.api.ops.impl.scalar.comparison"
        javaOpClass = "ScalarGreaterThan"
        legacy = true
        Input(NUMERIC, "x") { description = "Input array" }
        Arg(NUMERIC, "y") { description = "Double value argument to use in operation" }
        Output(NUMERIC, "output"){ description = "Boolean array out, with values true/false based on where the condition is satisfied" }
        Doc(Language.ANY, DocScope.ALL){
            """
                Greater than operation: elementwise x > y
            """.trimIndent()
        }
        useMixin(booleanReturnDoc)
    }

    Op("gt") {
        javaPackage = "org.nd4j.linalg.api.ops.impl.transforms.custom"
        javaOpClass = "GreaterThan"
        Input(NUMERIC, "x") { description = "Input 1" }
        Input(NUMERIC, "y") { description = "Input 2" }
        Output(NUMERIC, "output"){ description = "Output Boolean array out, with values true/false based on where the condition is satisfied" }

        Doc(Language.ANY, DocScope.ALL){
            """
                Greater than operation: elementwise x > y
                If x and y arrays have equal shape, the output shape is the same as these inputs.
            """.trimIndent()
        }
        useMixin(broadcastingDoc)
        useMixin(booleanReturnDoc)
    }

    Op("gte") {
        javaPackage = "org.nd4j.linalg.api.ops.impl.scalar.comparison"
        javaOpClass = "ScalarGreaterThanOrEqual"
        legacy = true
        Input(NUMERIC, "x") { description = "Input array" }
        Arg(NUMERIC, "y") {  description = "Double value argument to use in operation" }
        Output(NUMERIC, "output"){ description = "Output Boolean array out, with values true/false based on where the condition is satisfied" }
        Doc(Language.ANY, DocScope.ALL){
            """
                Greater than or equals operation: elementwise x >= y
            """.trimIndent()
        }
        useMixin(booleanReturnDoc)
    }

    Op("gte") {
        javaPackage = "org.nd4j.linalg.api.ops.impl.transforms.custom"
        javaOpClass = "GreaterThanOrEqual"
        Input(NUMERIC, "x") { description = "Input 1" }
        Input(NUMERIC, "y") { description = "Input 2" }
        Output(NUMERIC, "output"){ description = "" }
        Doc(Language.ANY, DocScope.ALL){
            """
                Greater than or equal to operation: elementwise x >= y
                If x and y arrays have equal shape, the output shape is the same as these inputs.
            """.trimIndent()
        }
        useMixin(broadcastingDoc)
        useMixin(booleanReturnDoc)
    }

    Op("identity") {
        javaPackage = "org.nd4j.linalg.api.ops.impl.transforms.same"
        Input(NUMERIC, "input") { description = "Input variable" }
        Output(NUMERIC, "output"){ description = "Output variable" }
        Doc(Language.ANY, DocScope.ALL){
            """
                Elementwise identity operation: out = x
            """.trimIndent()
        }
    }

    Op("invertPermutation") {
        javaPackage = "org.nd4j.linalg.api.ops.impl.transforms.custom"
        Input(INT, "input") { description = "1D indices for permutation" }
        Output(INT, "output"){ description = "1D inverted permutation" }
        Doc(Language.ANY, DocScope.ALL){
            """
                Compute the inverse permutation indices for a permutation operation
                Example: if input is [2, 0, 1] then output is [1, 2, 0]
                The idea is that x.permute(input).permute(invertPermutation(input)) == x
            """.trimIndent()
        }
    }

    Op("isNumericTensor") {
        javaPackage = "org.nd4j.linalg.api.ops.impl.transforms.custom"
        Input(NUMERIC, "x") { description = "Input variable" }
        Output(NDARRAY, "output"){ description = "scalar boolean with value true or false" }
        Doc(Language.ANY, DocScope.ALL){
            """
                Is the director a numeric tensor? In the current version of ND4J/SameDiff, this always returns true/1
            """.trimIndent()
        }
    }

    Op("linspace") {
        javaPackage = "org.nd4j.linalg.api.ops.impl.shape"
        Arg(DATA_TYPE, "dataType") { description = "Data type of the output array" }
        Arg(NUMERIC, "start") { description = "Start value" }
        Arg(NUMERIC, "stop") { description = "Stop value" }
        Arg(LONG, "number") { description = "Number of values to generate" }
        Output(NUMERIC, "output"){ description = "INDArray  with linearly spaced elements" }

        Doc(Language.ANY, DocScope.ALL){
            """
                Create a new 1d array with values evenly spaced between values 'start' and 'stop'
                For example, linspace(start=3.0, stop=4.0, number=3) will generate [3.0, 3.5, 4.0]
            """.trimIndent()
        }
    }

    Op("linspace") {
        javaPackage = "org.nd4j.linalg.api.ops.impl.shape"
        Input(NUMERIC, "start") { description = "Start value" }
        Input(NUMERIC, "stop") { description = "Stop value" }
        Input(LONG, "number") { description = "Number of values to generate" }
        Arg(DATA_TYPE, "dataType") { description = "Data type of the output array" }
        Output(NUMERIC, "output"){ description = "INDArray  with linearly spaced elements" }

        Doc(Language.ANY, DocScope.ALL){
            """
                Create a new 1d array with values evenly spaced between values 'start' and 'stop'
                For example, linspace(start=3.0, stop=4.0, number=3) will generate [3.0, 3.5, 4.0]
            """.trimIndent()
        }
    }

    Op("lt") {
        javaPackage = "org.nd4j.linalg.api.ops.impl.scalar.comparison"
        javaOpClass = "ScalarLessThan"
        legacy = true
        Input(NUMERIC, "x") { description = "Input array" }
        Arg(NUMERIC, "y") { description = "Double value argument to use in operation" }
        Output(NUMERIC, "output"){ description = "Boolean array out, with values true/false based on where the condition is satisfied" }
        Doc(Language.ANY, DocScope.ALL){
            """
                Less than operation: elementwise x < y
            """.trimIndent()
        }
        useMixin(booleanReturnDoc)
    }

    Op("lt") {
        javaPackage = "org.nd4j.linalg.api.ops.impl.transforms.custom"
        javaOpClass = "LessThan"
        Input(NUMERIC, "x") {description = "Input 1" }
        Input(NUMERIC, "y") {description = "Input 2" }
        Output(NUMERIC, "output"){ description = "Output Boolean array out, with values true/false based on where the condition is satisfied" }

        Doc(Language.ANY, DocScope.ALL){
            """ 
                Less than operation: elementwise x < y
                If x and y arrays have equal shape, the output shape is the same as these inputs.
            """.trimIndent()
        }
        useMixin(broadcastingDoc)
        useMixin(booleanReturnDoc)
    }

    Op("lte") {
        javaPackage = "org.nd4j.linalg.api.ops.impl.scalar.comparison"
        javaOpClass = "ScalarLessThanOrEqual"
        legacy = true
        Input(NUMERIC, "x") { description = "Input array" }
        Arg(NUMERIC, "y") { description = "Double value argument to use in operation" }
        Output(NUMERIC, "output"){ description = "Boolean array out, with values true/false based on where the condition is satisfied" }
        Doc(Language.ANY, DocScope.ALL){
            """
                Less than or equals operation: elementwise x <= y
            """.trimIndent()
        }
        useMixin(booleanReturnDoc)
    }

    Op("lte") {
        javaPackage = "org.nd4j.linalg.api.ops.impl.transforms.custom"
        javaOpClass = "LessThanOrEqual"
        Input(NUMERIC, "x") { description = "Input 1" }
        Input(NUMERIC, "y") { description = "Input 2" }
        Output(NUMERIC, "output"){ description = "Output Boolean array out, with values true/false based on where the condition is satisfied" }
        Doc(Language.ANY, DocScope.ALL){
            """ 
                Less than or equal to operation: elementwise x <= y
                If x and y arrays have equal shape, the output shape is the same as these inputs.
            """.trimIndent()
        }
        useMixin(broadcastingDoc)
        useMixin(booleanReturnDoc)
    }

    Op("matchCondition") {
        javaPackage = "org.nd4j.linalg.api.ops.impl.transforms.bool"
        javaOpClass = "MatchConditionTransform"
        legacy = true
        Input(NUMERIC, "in") { description = "Input" }
        Arg(CONDITION, "condition") { description = "Condition" }
        Output(NUMERIC, "output"){ description = "Boolean mask" }
        Doc(Language.ANY, DocScope.ALL){
            """
                Returns a boolean mask of equal shape to the input, where the condition is satisfied - value 1 where satisfied, 0 otherwise
            """.trimIndent()
        }
    }

    Op("matchConditionCount") {
        javaPackage = "org.nd4j.linalg.api.ops.impl.reduce.longer"
        javaOpClass = "MatchCondition"
        legacy = true
        Input(NUMERIC, "in") { description = "Input" }
        Arg(CONDITION, "condition") { description = "Condition" }
        Output(NUMERIC, "output"){ description = "Number of elements that the condition is satisfied for" }
        Doc(Language.ANY, DocScope.ALL){
            """
                Returns a count of the number of elements that satisfy the condition
            """.trimIndent()
        }
    }

    Op("matchConditionCount") {
        javaPackage = "org.nd4j.linalg.api.ops.impl.reduce.longer"
        javaOpClass = "MatchCondition"
        legacy = true
        Input(NUMERIC, "in") { description = "Input variable" }
        Arg(CONDITION, "condition") { description = "Condition" }
        Arg(BOOL, "keepDim") { description = "If true: keep the dimensions that are reduced on (as size 1). False: remove the reduction dimensions"; defaultValue=FALSE}
        Arg(INT, "dimensions") {count = AtLeast(0); description = "Dimensions to reduce over. If dimensions are not specified, full array reduction is performed" }
        Output(NUMERIC, "output"){ description = "Number of elements that the condition is satisfied for" }
        Doc(Language.ANY, DocScope.ALL){
            """
                Returns a count of the number of elements that satisfy the condition (for each slice along the specified dimensions)
            """.trimIndent()
        }
        useMixin(keepDimsDoc)
    }

    Op("max") {
        javaPackage = "org.nd4j.linalg.api.ops.impl.reduce.same"
        legacy = true
        Input(NUMERIC, "x") { description = "Input variable" }
        Arg(BOOL, "keepDims") { description = "If true: keep the dimensions that are reduced on (as size 1). False: remove the reduction dimensions"
            ; defaultValue=FALSE }
        Arg(INT, "dimensions") { count = AtLeast(0); description = "Dimensions to reduce over. If dimensions are not specified, full array reduction is performed" }
        Output(NUMERIC, "output"){ description = "Reduced array of rank (input rank - num dimensions)" }
        Doc(Language.ANY, DocScope.ALL){
            """
                Max array reduction operation, optionally along specified dimensions
            """.trimIndent()
        }
        useMixin(keepDimsDoc)
    }

    Op("max") {
        javaPackage = "org.nd4j.linalg.api.ops.impl.transforms.custom"
        Input(NUMERIC, "first") { description = "First input array" }
        Input(NUMERIC, "second") { description = "Second input array" }
        Output(NUMERIC, "output"){ description = "Output variable" }
        Doc(Language.ANY, DocScope.ALL){
            """
                Element-wise maximum operation: out[i] = max(first[i], second[i])
            """.trimIndent()
        }
        useMixin(broadcastingDoc)
    }

    Op("mean") {
        javaPackage = "org.nd4j.linalg.api.ops.impl.reduce.floating"
        legacy = true
        Input(NUMERIC, "x") { description = "Input variable" }
        Arg(BOOL, "keepDims") { description = "If true: keep the dimensions that are reduced on (as size 1). False: remove the reduction dimensions"; defaultValue=false }
        Arg(INT, "dimensions") { count = AtLeast(0); description = "Dimensions to reduce over. If dimensions are not specified, full array reduction is performed" }
        Output(NUMERIC, "output"){ description = "Reduced array of rank (input rank - num dimensions)" }
        Doc(Language.ANY, DocScope.ALL){
            """
                Mean (average) array reduction operation, optionally along specified dimensions
            """.trimIndent()
        }
        useMixin(keepDimsDoc)
    }

    Op("merge") {
        javaPackage = "org.nd4j.linalg.api.ops.impl.controlflow.compat"
        Input(NUMERIC, "x") { description = "Input variable" }
        Input(NUMERIC, "y") { description = "Input variable" }
        Output(NUMERIC, "output"){ description = "Output" }
        Doc(Language.ANY, DocScope.ALL){
            """
                The merge operation is a control operation that forwards the either of the inputs to the output, when
                the first of them becomes available. If both are available, the output is undefined (either input could
                be forwarded to the output)
            """.trimIndent()
        }
    }


    Op("min") {
        javaPackage = "org.nd4j.linalg.api.ops.impl.reduce.same"
        legacy = true
        Input(NUMERIC, "x") { description = "Input variable" }
        Arg(BOOL, "keepDims") { description = "If true: keep the dimensions that are reduced on (as size 1). False: remove the reduction dimensions"; defaultValue=false }
        Arg(INT, "dimensions") { count = AtLeast(0); description = "Dimensions to reduce over. If dimensions are not specified, full array reduction is performed" }
        Output(NUMERIC, "output"){ description = "Reduced array of rank (input rank - num dimensions)" }
        Doc(Language.ANY, DocScope.ALL){
            """
                Minimum array reduction operation, optionally along specified dimensions. out = min(in)
            """.trimIndent()
        }
        useMixin(keepDimsDoc)
    }

    Op("min") {
        javaPackage = "org.nd4j.linalg.api.ops.impl.transforms.custom"
        Input(NUMERIC, "first") { description = "First input array" }
        Input(NUMERIC, "second") { description = "Second input array" }
        Output(NUMERIC, "output"){ description = "Second input array" }
        Doc(Language.ANY, DocScope.ALL){
            """
                Element-wise minimum operation: out[i] = min(first[i], second[i])
            """.trimIndent()
        }
        useMixin(broadcastingDoc)
    }

    Op("mmul") {
        javaPackage = "org.nd4j.linalg.api.ops.impl.reduce"
        Input(NUMERIC, "x") { description = "First input variable" }
        Input(NUMERIC, "y") { description = "Second input variable" }
        Arg(BOOL, "transposeX") { description = "Transpose x (first argument)"; defaultValue=false }
        Arg(BOOL, "transposeY") { description = "Transpose y (second argument)"; defaultValue=false }
        Arg(BOOL, "transposeZ") { description = "Transpose result array"; defaultValue=false }
        Output(NUMERIC, "output"){ description = "" }
        Doc(Language.ANY, DocScope.ALL){
            """
                Matrix multiplication: out = mmul(x,y)
                Supports specifying transpose argument to perform operation such as mmul(a^T, b), etc.
            """.trimIndent()
        }
    }

    Op("neq") {
        javaPackage = "org.nd4j.linalg.api.ops.impl.scalar.comparison"
        javaOpClass = "ScalarNotEquals"
        legacy = true
        Input(NUMERIC, "x") {  description = "Input array" }
        Arg(NUMERIC, "y") {  description = "Double value argument to use in operation" }
        Output(NUMERIC, "output"){ description = "Boolean array out, with values true/false based on where the condition is satisfied" }
        Doc(Language.ANY, DocScope.ALL){
            """
                Not equals operation: elementwise x != y
            """.trimIndent()
        }
        useMixin(booleanReturnDoc)
    }

    Op("neq") {
        javaPackage = "org.nd4j.linalg.api.ops.impl.transforms.custom"
        javaOpClass = "NotEqualTo"
        Input(NUMERIC, "x") { description = "Input 1" }
        Input(NUMERIC, "y") { description = "Input 2" }
        Output(NUMERIC, "output"){ description = "Boolean array out, with values true/false based on where the condition is satisfied" }
        Doc(Language.ANY, DocScope.ALL){
            """
                Not equal to operation: elementwise x != y
                If x and y arrays have equal shape, the output shape is the same as these inputs.
            """.trimIndent()
        }
        useMixin(broadcastingDoc)
        useMixin(booleanReturnDoc)
    }

    Op("norm1") {
        javaPackage = "org.nd4j.linalg.api.ops.impl.reduce.floating"
        legacy = true
        Input(NUMERIC, "x") { description = "Input variable" }
        Arg(BOOL, "keepDims") { description = "If true: keep the dimensions that are reduced on (as size 1). False: remove the reduction dimensions"; defaultValue=false }
        Arg(INT, "dimensions") { count = AtLeast(0);  description = "dimensions to reduce over" }
        Output(NUMERIC, "output"){ description = "Output variable" }
        Doc(Language.ANY, DocScope.ALL){
            """
                Norm1 (L1 norm) reduction operation: The output contains the L1 norm for each tensor/subset along the specified dimensions: 
                out = sum_i abs(x[i])
            """.trimIndent()
        }
        useMixin(keepDimsDoc)
    }

    Op("norm2") {
        javaPackage = "org.nd4j.linalg.api.ops.impl.reduce.floating"
        legacy = true
        Input(NUMERIC, "x") { description = "Input variable" }
        Arg(BOOL, "keepDims") { description = "If true: keep the dimensions that are reduced on (as size 1). False: remove the reduction dimensions"; defaultValue=false }
        Arg(INT, "dimensions") { count = AtLeast(0); description = "dimensions dimensions to reduce over" }
        Output(NUMERIC, "output"){ description = "Output variable" }
        Doc(Language.ANY, DocScope.ALL){
            """
                Norm2 (L2 norm) reduction operation: The output contains the L2 norm for each tensor/subset along the specified dimensions:
                out = sqrt(sum_i x[i]^2)
            """.trimIndent()
        }
        useMixin(keepDimsDoc)
    }

    Op("normmax") {
        javaPackage = "org.nd4j.linalg.api.ops.impl.reduce.floating"
        javaOpClass = "NormMax"
        legacy = true
        Input(NUMERIC, "x") { description = "Input variable" }
        Arg(BOOL, "keepDims") { description = "If true: keep the dimensions that are reduced on (as size 1). False: remove the reduction dimensions"; defaultValue=false }
        Arg(INT, "dimensions") { count = AtLeast(0); description = "dimensions to reduce over" }
        Output(NUMERIC, "output"){ description = "Output variable" }
        Doc(Language.ANY, DocScope.ALL){
            """
                Max norm (infinity norm) reduction operation: The output contains the max norm for each tensor/subset along the
                specified dimensions:
                out = max(abs(x[i]))
            """.trimIndent()
        }
        useMixin(keepDimsDoc)
    }

    Op("split")  {
        javaPackage = "org.nd4j.linalg.api.ops.impl.shape"
        javaOpClass = "Split"
        Input(NUMERIC,"input") {description = "Input to split"}
        Arg(INT, "numSplit") { description = "Number of splits" }
        Arg(INT, "splitDim") { description = "The dimension to split on" }
        Doc(Language.ANY, DocScope.ALL){
            """
               Split a value in to a list of ndarrays.
            """.trimIndent()
        }
    }

    Op("oneHot") {
        javaPackage = "org.nd4j.linalg.api.ops.impl.shape"
        Input(NUMERIC, "indices") { description = "Indices - value 0 to depth-1" }
        Arg(INT, "depth") { description = "Number of classes" }
        Arg(INT, "axis") { description = "" }
        Arg(NUMERIC, "on") { description = "" }
        Arg(NUMERIC, "off") { description = "" }
        Arg(DATA_TYPE, "dataType") { description = "Output data type"; defaultValue = org.nd4j.linalg.api.buffer.DataType.FLOAT }
        Output(NUMERIC, "output"){ description = "Output variable" }

        Doc(Language.ANY, DocScope.ALL){
            """
                Convert the array to a one-hot array with walues and  for each entry
                If input has shape [ a, ..., n] then output has shape [ a, ..., n, depth],
                with {out[i, ..., j, in[i,...,j]]  with other values being set to
            """.trimIndent()
        }
    }

    Op("oneHot") {
        javaPackage = "org.nd4j.linalg.api.ops.impl.shape"
        Input(NUMERIC, "indices") { description = "Indices - value 0 to depth-1" }
        Arg(INT, "depth") { description = "Number of classes" }
        Output(NUMERIC, "output"){ description = "Output variable" }
        Doc(Language.ANY, DocScope.ALL){
            """
                Convert the array to a one-hot array with walues 0 and 1 for each entry
                If input has shape [ a, ..., n] then output has shape [ a, ..., n, depth],
                with out[i, ..., j, in[i,...,j]] = 1 with other values being set to 0
                see oneHot(SDVariable, int, int, double, double)
            """.trimIndent()
        }
    }

    Op("onesLike") {
        javaPackage = "org.nd4j.linalg.api.ops.impl.shape"
        Input(NUMERIC, "input") { description = "Input INDArray " }
        Output(NUMERIC, "output"){ description = "A new INDArray  with the same (dynamic) shape as the input" }
        Doc(Language.ANY, DocScope.ALL){
            """
                Return a variable of all 1s, with the same shape as the input variable. Note that this is dynamic:
                if the input shape changes in later execution, the returned variable's shape will also be updated
            """.trimIndent()
        }
    }

    Op("onesLike") {
        javaPackage = "org.nd4j.linalg.api.ops.impl.shape"
        Input(NUMERIC, "input") { description = "" }
        Arg(DATA_TYPE, "dataType") { description = "" }
        Output(NUMERIC, "output"){ description = "" }
        Doc(Language.ANY, DocScope.ALL){
            """
                As per onesLike(String, SDVariable) but the output datatype may be specified
            """.trimIndent()
        }
    }

    Op("permute") {
        javaPackage = "org.nd4j.linalg.api.ops.impl.shape"
        Input(NUMERIC, "x") { description = "Input variable" }
        Input(INT, "dimensions") { description = "Permute dimensions" }
        Output(NUMERIC, "output"){ description = "Output variable (permuted input)" }
        Doc(Language.ANY, DocScope.ALL){
            """
                Array permutation operation: permute the dimensions according to the specified permutation indices.
                Example: if input has shape [a,b,c] and dimensions = [2,0,1] the output has shape [c,a,b]
            """.trimIndent()
        }
    }

    Op("permute") {
        javaPackage = "org.nd4j.linalg.api.ops.impl.shape"
        Input(NUMERIC, "x") { description = "Input variable" }
        Arg(INT, "dimensions") { count = AtLeast(0); description = "" }
        Output(NUMERIC, "output"){ description = "Output variable (permuted input)" }
        Doc(Language.ANY, DocScope.ALL){
            """
                Array permutation operation: permute the dimensions according to the specified permutation indices.
                Example: if input has shape [a,b,c] and dimensions = [2,0,1] the output has shape [c,a,b]
            """.trimIndent()
        }
    }

    Op("prod") {
        javaPackage = "org.nd4j.linalg.api.ops.impl.reduce.same"
        legacy = true
        Input(NUMERIC, "x") { description = "Input variable" }
        Arg(BOOL, "keepDims") { description = "If true: keep the dimensions that are reduced on (as size 1). False: remove the reduction dimensions"; defaultValue=false }
        Arg(INT, "dimensions") { count = AtLeast(0); description = "Dimensions to reduce over. If dimensions are not specified, full array reduction is performed" }
        Output(NUMERIC, "output"){ description = "" }
        Doc(Language.ANY, DocScope.ALL){
            """
                Product array reduction operation, optionally along specified dimensions
            """.trimIndent()
        }
        useMixin(keepDimsDoc)
    }

    Op("range") {
        javaPackage = "org.nd4j.linalg.api.ops.random.impl"
        Arg(NUMERIC, "from") { description = "Initial/smallest value" }
        Arg(NUMERIC, "to") { description = "Largest value (exclusive)" }
        Arg(NUMERIC, "step") { description = "Step size" }
        Arg(DATA_TYPE, "dataType") { description = "" }
        Output(NUMERIC, "output"){ description = "INDArray  with the specified values" }

        Doc(Language.ANY, DocScope.ALL){
            """
                Create a new variable with a 1d array, where the values start at from and increment by step
                up to (but not including) limit.
                For example, range(1.0, 3.0, 0.5) will return [1.0, 1.5, 2.0, 2.5]
            """.trimIndent()
        }
    }

    Op("range") {
        javaPackage = "org.nd4j.linalg.api.ops.random.impl"
        Input(NUMERIC, "from") { description = "Initial/smallest value" }
        Input(NUMERIC, "to") { description = "Largest value (exclusive)" }
        Input(NUMERIC, "step") { description = "Step size" }
        Arg(DATA_TYPE, "dataType") { description = "" }
        Output(NUMERIC, "output"){ description = "INDArray  with the specified values" }

        Doc(Language.ANY, DocScope.ALL){
            """
                Create a new variable with a 1d array, where the values start at from and increment by step
                up to (but not including) limit.
                For example, range(1.0, 3.0, 0.5) will return [1.0, 1.5, 2.0, 2.5]
            """.trimIndent()
        }
    }

    Op("rank") {
        javaPackage = "org.nd4j.linalg.api.ops.impl.shape"
        Input(NUMERIC, "in") { description = "Input variable" }
        Output(NUMERIC, "output"){ description = "(scalar) output variable with value equal to the rank of the input variable" }
        Doc(Language.ANY, DocScope.ALL){
            """
                Returns the rank (number of dimensions, i.e., length(shape)) of the specified INDArray  as a 0D scalar variable
            """.trimIndent()
        }
    }

    /*
    Op("repeat") {
        javaPackage = "org.nd4j.linalg.api.ops.impl.shape"
        Input(NUMERIC, "df") { description = "" }
        Arg(INT, "axis") { description = "" }
        Output(NUMERIC, "output"){ description = "" }
        Doc(Language.ANY, DocScope.ALL){
            """
                see repeat(String, SDVariable, int)
            """.trimIndent()
        }
    }
    */

    Op("replaceWhere") {
        javaPackage = "org.nd4j.linalg.api.ops.impl.transforms.comparison"
        javaOpClass = "CompareAndReplace"
        legacy = true
        Input(NUMERIC, "update") { description = "Source array" }
        Input(NUMERIC, "from") { description = "Replacement values array (used conditionally). Must be same shape as 'update' array" }
        Arg(CONDITION, "condition") { description = "Condition to check on update array elements" }
        Output(NUMERIC, "output"){ description = "New array with values replaced where condition is satisfied" }
        Doc(Language.ANY, DocScope.ALL){
            """
                Element-wise replace where condition:
                out[i] = from[i] if condition(update[i]) is satisfied, or
                out[i] = update[i] if condition(update[i]) is NOT satisfied
            """.trimIndent()
        }
    }

    Op("replaceWhere") {
        javaPackage = "org.nd4j.linalg.api.ops.impl.transforms.comparison"
        javaOpClass = "CompareAndSet"
        legacy = true
        Input(NUMERIC, "update") { description = "Source array" }
        Arg(NUMERIC, "value") { description = "Value to set at the output, if the condition is satisfied" }
        Arg(CONDITION, "condition") { description = "Condition to check on update array elements" }
        Output(NUMERIC, "output"){ description = "New array with values replaced where condition is satisfied" }
        Doc(Language.ANY, DocScope.ALL){
            """
                Element-wise replace where condition:
                out[i] = value if condition(update[i]) is satisfied, or
                out[i] = update[i] if condition(update[i]) is NOT satisfied
            """.trimIndent()
        }
    }

    Op("reshape") {
        javaPackage = "org.nd4j.linalg.api.ops.impl.shape"
        Input(NUMERIC, "x") { description = "Input variable" }
        Input(NUMERIC, "shape") { description = "New shape for variable" }
        Output(NUMERIC, "output"){ description = "Output variable" }
        Doc(Language.ANY, DocScope.ALL){
            """
                Reshape the input variable to the specified (fixed) shape. The output variable will have the same values as the
                input, but with the specified shape.
                Note that prod(shape) must match length(input) == prod(input.shape)
            """.trimIndent()
        }
    }

    Op("reshape") {
        javaPackage = "org.nd4j.linalg.api.ops.impl.shape"
        Input(NUMERIC, "x") { description = "Input variable" }
        Arg(LONG, "shape") { count=AtLeast(0); description = "New shape for variable" }
        Output(NUMERIC, "output"){ description = "Output variable" }
        Doc(Language.ANY, DocScope.ALL){
            """
                Reshape the input variable to the specified (fixed) shape. The output variable will have the same values as the
                input, but with the specified shape.
                Note that prod(shape) must match length(input) == prod(input.shape)
            """.trimIndent()
        }
    }

    Op("reverse") {
        javaPackage = "org.nd4j.linalg.api.ops.impl.transforms.custom"
        Input(NUMERIC, "x") { description = "Input variable" }
        Arg(INT, "dimensions") { count = AtLeast(0); description = "Input variable" }
        Output(NUMERIC, "output"){ description = "Output variable" }
        Doc(Language.ANY, DocScope.ALL){
            """
                Reverse the values of an array for the specified dimensions
                If input is:
                [ 1, 2, 3]
                [ 4, 5, 6]
                then
                reverse(in, 0):
                [3, 2, 1]
                [6, 5, 4]
                reverse(in, 1):
                [4, 5, 6]
                [1, 2 3]
            """.trimIndent()
        }
    }

    Op("reverseSequence") {
        javaPackage = "org.nd4j.linalg.api.ops.impl.transforms.custom"
        Input(NUMERIC, "x") { description = "Input variable" }
        Input(INT, "seq_lengths") { description = "Length of the sequences" }
        Arg(INT, "seqDim") { description = "Sequence dimension"; defaultValue=-1}
        Arg(INT, "batchDim") { description = "Batch dimension"; defaultValue=0 }
        Output(NUMERIC, "output"){ description = "Reversed sequences" }
        Doc(Language.ANY, DocScope.ALL){
            """
                Reverse sequence op: for each slice along dimension seqDimension, the first seqLength values are reversed
            """.trimIndent()
        }
    }

    Op("scalarFloorMod") {
        javaPackage = "org.nd4j.linalg.api.ops.impl.scalar"
        javaOpClass = "ScalarFMod"
        legacy = true
        Input(NUMERIC, "in") { description = "Input variable" }
        Arg(NUMERIC, "value") { description = "Scalar value to compare" }
        Output(NUMERIC, "output"){ description = "Output variable" }
        Doc(Language.ANY, DocScope.ALL){
            """
                Element-wise scalar floor modulus operation: out = floorMod(in, value).
                i.e., returns the remainder after division by 'value'
            """.trimIndent()
        }
    }

    Op("scalarMax") {
        javaPackage = "org.nd4j.linalg.api.ops.impl.scalar"
        legacy = true
        Input(NUMERIC, "in") { description = "Input variable" }
        Arg(NUMERIC, "value") { description = "Scalar value to compare" }
        Output(NUMERIC, "output"){ description = "Scalar value to compare" }
        Doc(Language.ANY, DocScope.ALL){
            """
                Element-wise scalar maximum operation: out = max(in, value)
            """.trimIndent()
        }
    }

    Op("scalarMin") {
        javaPackage = "org.nd4j.linalg.api.ops.impl.scalar"
        legacy = true
        Input(NUMERIC, "in") { description = "Input variable" }
        Arg(NUMERIC, "value") { description = "Scalar value to compare" }
        Output(NUMERIC, "output"){ description = "Output variable" }
        Doc(Language.ANY, DocScope.ALL){
            """
                Element-wise scalar minimum operation: out = min(in, value)
            """.trimIndent()
        }
    }

    Op("scalarSet") {
        javaPackage = "org.nd4j.linalg.api.ops.impl.scalar"
        legacy = true
        Input(NUMERIC, "in") { description = "Input variable" }
        Arg(NUMERIC, "set") { description = "Value to set" }
        Output(NUMERIC, "output"){ description = "Output variable" }
        Doc(Language.ANY, DocScope.ALL){
            """
                Return a variable with equal shape to the input, but all elements set to value 'set'
            """.trimIndent()
        }
    }

    Op("scatterAdd") {
        useMixin(scatterOp)
        Doc(Language.ANY, DocScope.ALL){
            """
                Scatter addition operation.
            """.trimIndent()
        }
        useMixin(scatterDoc)
    }

    Op("scatterDiv") {
        useMixin(scatterOp)
        Doc(Language.ANY, DocScope.ALL){
            """
                Scatter division operation.
            """.trimIndent()
        }
        useMixin(scatterDoc)
    }

    Op("scatterMax") {
        useMixin(scatterOp)
        Doc(Language.ANY, DocScope.ALL){
            """
                Scatter max operation.
            """.trimIndent()
        }
        useMixin(scatterDoc)
    }

    Op("scatterMin") {
        useMixin(scatterOp)
        Doc(Language.ANY, DocScope.ALL){
            """
                Scatter min operation.
            """.trimIndent()
        }
        useMixin(scatterDoc)
    }

    Op("scatterMul") {
        useMixin(scatterOp)
        Doc(Language.ANY, DocScope.ALL){
            """
                Scatter multiplication operation.
            """.trimIndent()
        }
        useMixin(scatterDoc)
    }

    Op("scatterSub") {
        useMixin(scatterOp)
        Doc(Language.ANY, DocScope.ALL){
            """
                Scatter subtraction operation.
            """.trimIndent()
        }
        useMixin(scatterDoc)
    }

    Op("scatterUpdate") {
        useMixin(scatterOp)
        Doc(Language.ANY, DocScope.ALL){
            """
                Scatter update operation.
            """.trimIndent()
        }
        useMixin(scatterDoc)
    }

    Op("segmentMax") {
        useMixin(segmentOp)
        Doc(Language.ANY, DocScope.ALL){
            """
                Segment max operation.
            """.trimIndent()
        }
        useMixin(segmentDoc)
    }

    Op("segmentMean") {
        useMixin(segmentOp)
        Doc(Language.ANY, DocScope.ALL){
            """
                Segment mean operation.
            """.trimIndent()
        }
        useMixin(segmentDoc)
    }

    Op("segmentMin") {
        useMixin(segmentOp)
        Doc(Language.ANY, DocScope.ALL){
            """
                Segment min operation.
            """.trimIndent()
        }
        useMixin(segmentDoc)
    }

    Op("segmentProd") {
        useMixin(segmentOp)
        Doc(Language.ANY, DocScope.ALL){
            """
                Segment product operation.
            """.trimIndent()
        }
        useMixin(segmentDoc)
    }

    Op("segmentSum") {
        useMixin(segmentOp)
        Doc(Language.ANY, DocScope.ALL){
            """
                Segment sum operation.
            """.trimIndent()
        }
        useMixin(segmentDoc)
    }

    Op("sequenceMask") {
        javaPackage = "org.nd4j.linalg.api.ops.impl.shape"
        Input(NUMERIC, "lengths") { description = "Lengths of the sequences" }
        Arg(INT, "maxLen") { description = "Maximum sequence length" }
        Arg(DATA_TYPE, "dataType") { description = "" }
        Output(NUMERIC, "output"){ description = "Output variable" }
        Doc(Language.ANY, DocScope.ALL){
            """
                 Generate a sequence mask (with values 0 or 1) based on the specified lengths 
                 Specifically, out[i, ..., k, j] = (j < lengths[i, ..., k] ? 1.0 : 0.0)
            """.trimIndent()
        }
    }

    Op("sequenceMask") {
        javaPackage = "org.nd4j.linalg.api.ops.impl.shape"
        Input(NUMERIC, "lengths") { description = "Lengths of the sequences" }
        Input(INT, "maxLen") { description = "Maximum sequence length" }
        Arg(DATA_TYPE, "dataType") { description = "" }
        Output(NUMERIC, "output"){ description = "Output variable" }
        Doc(Language.ANY, DocScope.ALL){
            """
                 Generate a sequence mask (with values 0 or 1) based on the specified lengths 
                 Specifically, out[i, ..., k, j] = (j < lengths[i, ..., k] ? 1.0 : 0.0)
            """.trimIndent()
        }
    }

    Op("sequenceMask") {
        javaPackage = "org.nd4j.linalg.api.ops.impl.shape"
        Input(NUMERIC, "lengths") { description = "" }
        Arg(DATA_TYPE,  "dataType") { description = "" }
        Output(NUMERIC, "output"){ description = "" }

        Doc(Language.ANY, DocScope.ALL){
            """
                see sequenceMask(String, SDVariable, SDVariable, DataType)
            """.trimIndent()
        }
    }

    Op("shape") {
        javaPackage = "org.nd4j.linalg.api.ops.impl.shape"
        Input(NUMERIC, "input") { description = "Input variable" }
        Output(NUMERIC, "output"){ description = "1D output variable with contents equal to the shape of the input" }
        Doc(Language.ANY, DocScope.ALL){
            """
                Returns the shape of the specified INDArray  as a 1D INDArray 
            """.trimIndent()
        }
    }

    Op("size") {
        javaPackage = "org.nd4j.linalg.api.ops.impl.shape"
        Input(NUMERIC, "in") { description = "Input variable" }
        Output(NUMERIC, "output"){ description = "0D (scalar) output variable with value equal to the number of elements in the specified array" }
        Doc(Language.ANY, DocScope.ALL){
            """ 
                Returns the size (number of elements, i.e., prod(shape)) of the specified INDArray  as a 0D scalar variable
            """.trimIndent()
        }
    }

    Op("sizeAt") {
        javaPackage = "org.nd4j.linalg.api.ops.impl.shape"
        Input(NUMERIC, "in") { description = "Input variable" }
        Arg(INT, "dimension") { description = "Dimension to get size of" }
        Output(NUMERIC, "output"){ description = "Scalar INDArray  for size at specified variable" }
        Doc(Language.ANY, DocScope.ALL){
            """
                Returns a rank 0 (scalar) variable for the size of the specified dimension.
                For example, if X has shape [10,20,30] then sizeAt(X,1)=20. Similarly, sizeAt(X,-1)=30
            """.trimIndent()
        }
    }

    Op("slice") {
        javaPackage = "org.nd4j.linalg.api.ops.impl.shape"
        Input(NUMERIC, "input") { description = "input Variable to get subset of" }
        Arg(INT, "begin") { count = AtLeast(1); description = "Beginning index. Must be same length as rank of input array" }
        Arg(INT, "size") { count = AtLeast(1); description = "Size of the output array. Must be same length as rank of input array" }
        Output(NUMERIC, "output"){ description = "Subset of the input" }
        Doc(Language.ANY, DocScope.ALL){
            """
                Get a subset of the specified input, by specifying the first element and the size of the array.
                For example, if input is:
                [a, b, c]
                [d, e, f]
                then slice(input, begin=[0,1], size=[2,1] will return:
                [b]
                [e]
                Note that for each dimension i, begin[i] + size[i] <= input.size(i)
            """.trimIndent()
        }
    }

    Op("slice") {
        javaPackage = "org.nd4j.linalg.api.ops.impl.shape"
        Input(NUMERIC, "input") { description = "input Variable to get subset of" }
        Input(INT, "begin") { description = "Beginning index. Must be same length as rank of input array" }
        Input(INT, "size") { description = "Size of the output array. Must be same length as rank of input array" }
        Output(NUMERIC, "output"){ description = "Subset of the input" }
        Doc(Language.ANY, DocScope.ALL){
            """
                Get a subset of the specified input, by specifying the first element and the size of the array.
                For example, if input is:
                [a, b, c]
                [d, e, f]
                then slice(input, begin=[0,1], size=[2,1] will return:
                [b]
                [e]
                Note that for each dimension i, begin[i] + size[i] <= input.size(i)
            """.trimIndent()
        }
    }

    Op("squaredNorm") {
        javaPackage = "org.nd4j.linalg.api.ops.impl.reduce.floating"
        legacy = true
        Input(NUMERIC, "x") { description = "" }
        Arg(BOOL, "keepDims") { description = ""; defaultValue=false }
        Arg(INT, "dimensions") { count = AtLeast(0); description = "" }
        Output(NUMERIC, "output"){ description = "" }
        Doc(Language.ANY, DocScope.ALL){
            """
                Squared L2 norm: see norm2(String, SDVariable, boolean, int...)
            """.trimIndent()
        }
        useMixin(keepDimsDoc)
    }

    Op("squeeze") {
        javaPackage = "org.nd4j.linalg.api.ops.impl.shape"
        Input(NUMERIC, "x") { description = "Input variable" }
        Arg(INT, "axis") { description = "Size 1 dimension to remove" }
        Output(NUMERIC, "output"){ description = "Output variable" }
        Doc(Language.ANY, DocScope.ALL){
            """
                Remove a single dimension of size 1.
                For example, if input has shape [a,b,1,c] then squeeze(input, 2) returns an array of shape [a,b,c]
            """.trimIndent()
        }
    }

    Op("stack") {
        javaPackage = "org.nd4j.linalg.api.ops.impl.shape"
        argsFirst = true
        Input(NDARRAY, "values") { count=AtLeast(1); description = "Input variables to stack. Must have the same shape for all inputs" }
        Arg(INT, "axis") { description = "Axis to stack on" }
        Output(NDARRAY, "output"){ description = "Output variable" }
        Doc(Language.ANY, DocScope.ALL){
            """
                Stack a set of N INDArray of rank X into one rank X+1 variable.
                If inputs have shape [a,b,c] then output has shape:
                axis = 0: [N,a,b,c]
                axis = 1: [a,N,b,c]
                axis = 2: [a,b,N,c]
                axis = 3: [a,b,c,N]
                see unstack(String[], SDVariable, int, int)
            """.trimIndent()
        }
    }

    Op("standardDeviation") {
        javaPackage = "org.nd4j.linalg.api.ops.impl.summarystats"
        legacy = true
        Input(NUMERIC, "x") { description = "Input variable" }
        Arg(BOOL, "biasCorrected") { description = "If true: divide by (N-1) (i.e., sample stdev). If false: divide by N (population stdev)" }
        Arg(BOOL, "keepDims") { description = "If true: keep the dimensions that are reduced on (as size 1). False: remove the reduction dimensions"; defaultValue=false }
        Arg(INT, "dimensions") { count= AtLeast(0); description = "Dimensions to reduce over. If dimensions are not specified, full array reduction is performed" }
        Output(NUMERIC, "output"){ description = "reduced array of rank (input rank - num dimensions)" }
        Doc(Language.ANY, DocScope.ALL){
            """
                Stardard deviation array reduction operation, optionally along specified dimensions
            """.trimIndent()
        }
        useMixin(keepDimsDoc)
    }

    Op("stridedSlice") {
        javaPackage = "org.nd4j.linalg.api.ops.impl.shape"
        Input(NUMERIC, "in") { description = "Variable to get subset of" }
        Arg(LONG, "begin") { count = AtLeast(1); description = "Beginning index" }
        Arg(LONG, "end") { count = AtLeast(1); description = "End index" }
        Arg(LONG, "strides") { count = AtLeast(1); description = "Stride (\"step size\") for each dimension. For example, stride of 2 means take every second element." }
        Arg(INT, "beginMask") { description = "Bit mask: If the ith bit is set to 1, then the value in the begin long[] is ignored, and a value of 0 is used instead for the beginning index for that dimension"; defaultValue=0 }
        Arg(INT, "endMask") { description = "Bit mask: If the ith bit is set to 1, then the value in the end long[] is ignored, and a value of size(i)-1 is used instead for the end index for that dimension"; defaultValue=0 }
        Arg(INT, "ellipsisMask") { description = "Bit mask: only one non-zero value is allowed here. If a non-zero value is set, then other dimensions are inserted as required at the specified position"; defaultValue=0 }
        Arg(INT, "newAxisMask") { description = "Bit mask: if the ith bit is set to 1, then the begin/end/stride values are ignored, and a size 1 dimension is inserted at this point"; defaultValue=0 }
        Arg(INT, "shrinkAxisMask") { description = "Bit mask: if the ith bit is set to 1, then the begin/end/stride values are ignored, and a size 1 dimension is removed at this point. Note that begin/end/stride values must result in a size 1 output for these dimensions"; defaultValue=0 }
        Output(NUMERIC, "output"){ description = "A subset of the input array" }
        Doc(Language.ANY, DocScope.ALL){
            """
                Get a subset of the specified input, by specifying the first element, last element, and the strides.
                For example, if input is:
                [a, b, c]
                [d, e, f]
                [g, h, i]
                then stridedSlice(input, begin=[0,1], end=[2,2], strides=[2,1], all masks = 0) will return:
                [b, c]
                [h, i]
            """.trimIndent()
        }
    }

    Op("sum") {
        javaPackage = "org.nd4j.linalg.api.ops.impl.reduce.same"
        legacy = true
        Input(NUMERIC, "x") { description = "Input variable" }
        Arg(BOOL, "keepDims") { description = "If true: keep the dimensions that are reduced on (as length 1). False: remove the reduction dimensions"; defaultValue=false }
        Arg(INT, "dimensions") { count= AtLeast(0); description = "Dimensions to reduce over. If dimensions are not specified, full array reduction is performed" }
        Output(NUMERIC, "output"){ description = "reduced array of rank (input rank - num dimensions) if keepDims = false, or of rank (input rank) if keepdims = true" }
        Doc(Language.ANY, DocScope.ALL){
            """
                Sum array reduction operation, optionally along specified dimensions.
            """.trimIndent()
        }
        useMixin(keepDimsDoc)
    }

    Op("switchOp") {
        javaPackage = "org.nd4j.linalg.api.ops.impl.controlflow.compat"
        javaOpClass = "Switch"
        legacy = false
        Input(NDARRAY, "x") { description = "Input variable" }
        Input(BOOL, "predicate"){ description = "Predictate - if false, values are output to left (first) branch/output; if true, to right (second) branch/output"}
        Output(NUMERIC, "outputLeft"){ description = "Output when predicate is false" }
        Output(NUMERIC, "outputRight"){ description = "Output when predicate is false" }
        Doc(Language.ANY, DocScope.ALL){
            """
                Switch operation
                Predictate - if false, values are output to left (first) branch/output; if true, to right (second) branch/output
            """.trimIndent()
        }
    }

    Op("tensorMmul") {
        javaPackage = "org.nd4j.linalg.api.ops.impl.reduce"
        Input(NUMERIC, "x") { description = "Input variable x" }
        Input(NUMERIC, "y") { description = "Input variable y" }
        Arg(INT, "dimensionsX") { count=AtLeast(1); description = "dimensions for first input array (x)" }
        Arg(INT, "dimensionsY") { count=AtLeast(1); description = "dimensions for second input array (y)" }
        Arg(BOOL, "transposeX") { description = "Transpose x (first argument)"; defaultValue=false }
        Arg(BOOL, "transposeY") { description = "Transpose y (second argument)"; defaultValue=false }
        Arg(BOOL, "transposeZ") { description = "Transpose result array"; defaultValue=false }
        Output(NUMERIC, "output"){ description = "Output variable" }
        Doc(Language.ANY, DocScope.ALL){
            """
                //TODO: Ops must be documented.
            """.trimIndent()
        }
    }

    Op("tile") {
        javaPackage = "org.nd4j.linalg.api.ops.impl.shape"
        Input(NDARRAY, "x") { description = "Input variable" }
        Input(INT, "repeat") { description = "Number of times to repeat in each axis. Must have length equal to the rank of the input array" }
        Output(NDARRAY, "output"){ description = "Output variable" }
        Doc(Language.ANY, DocScope.ALL){
            """
                Repeat (tile) the input tensor the specified number of times.
                For example, if input is
                [1, 2]
                [3, 4]
                and repeat is [2, 3]
                then output is
                [1, 2, 1, 2, 1, 2]
                [3, 4, 3, 4, 3, 4]
                [1, 2, 1, 2, 1, 2]
                [3, 4, 3, 4, 3, 4]
            """.trimIndent()
        }
    }

    Op("tile") {
        javaPackage = "org.nd4j.linalg.api.ops.impl.shape"
        Input(NDARRAY, "x") { description = "" }
        Arg(INT, "repeat") { count= AtLeast(1); description = "" }
        Output(NDARRAY, "output"){ description = "" }
        Doc(Language.ANY, DocScope.ALL){
            """
                see tile(String, SDVariable, int...)
            """.trimIndent()
        }
    }

    Op("transpose") {
        javaPackage = "org.nd4j.linalg.api.ops.impl.shape"
        Input(NDARRAY, "x") { description = "Input variable" }
        Output(NDARRAY, "output"){ description = "transposed input" }
        Doc(Language.ANY, DocScope.ALL){
            """
                Matrix transpose operation: If input has shape [a,b] output has shape [b,a]
            """.trimIndent()
        }
    }

    Op("unsortedSegmentMax") {
        useMixin(unsortedSegmentOp)
        Doc(Language.ANY, DocScope.ALL){
            """
                Unsorted segment max operation. As per segmentMax(String, SDVariable, SDVariable) but without
                the requirement for the indices to be sorted.
                If data =     [1, 3, 2, 6, 4, 9, 8]
                segmentIds =  [1, 0, 2, 0, 1, 1, 2]
                then output = [6, 9, 8] = [max(3,6), max(1,4,9), max(2,8)]
            """.trimIndent()
        }
    }

    Op("unsortedSegmentMean") {
        useMixin(unsortedSegmentOp)
        Doc(Language.ANY, DocScope.ALL){
            """
                Unsorted segment mean operation. As per segmentMean(String, SDVariable, SDVariable) but without
                the requirement for the indices to be sorted.
                If data =     [1, 3, 2, 6, 4, 9, 8]
                segmentIds =  [1, 0, 2, 0, 1, 1, 2]
                then output = [4.5, 4.666, 5] = [mean(3,6), mean(1,4,9), mean(2,8)]
            """.trimIndent()
        }
    }

    Op("unsortedSegmentMin") {
        useMixin(unsortedSegmentOp)
        Doc(Language.ANY, DocScope.ALL){
            """
                Unsorted segment min operation. As per segmentMin(String, SDVariable, SDVariable) but without
                the requirement for the indices to be sorted.
                If data =     [1, 3, 2, 6, 4, 9, 8]
                segmentIds =  [1, 0, 2, 0, 1, 1, 2]
                then output = [3, 1, 2] = [min(3,6), min(1,4,9), min(2,8)]
            """.trimIndent()
        }
    }

    Op("unsortedSegmentProd") {
        useMixin(unsortedSegmentOp)
        Doc(Language.ANY, DocScope.ALL){
            """
                Unsorted segment product operation. As per segmentProd(String, SDVariable, SDVariable) but without
                the requirement for the indices to be sorted.
                If data =     [1, 3, 2, 6, 4, 9, 8]
                segmentIds =  [1, 0, 2, 0, 1, 1, 2]
                then output = [4.5, 4.666, 5] = [mean(3,6), mean(1,4,9), mean(2,8)]
            """.trimIndent()
        }
    }

    Op("unsortedSegmentSqrtN") {
        useMixin(unsortedSegmentOp)
        Doc(Language.ANY, DocScope.ALL){
            """
                Unsorted segment sqrtN operation. Simply returns the sqrt of the count of the number of values in each segment
                If data =     [1, 3, 2, 6, 4, 9, 8]
                segmentIds =  [1, 0, 2, 0, 1, 1, 2]
                then output = [1.414, 1.732, 1.414] = [sqrt(2), sqrtN(3), sqrtN(2)]
            """.trimIndent()
        }
    }

    Op("unsortedSegmentSum") {
        useMixin(unsortedSegmentOp)
        Doc(Language.ANY, DocScope.ALL){
            """
                Unsorted segment sum operation. As per segmentSum(String, SDVariable, SDVariable) but without
                the requirement for the indices to be sorted.
                If data =     [1, 3, 2, 6, 4, 9, 8]
                segmentIds =  [1, 0, 2, 0, 1, 1, 2]
                then output = [9, 14, 10] = [sum(3,6), sum(1,4,9), sum(2,8)]
            """.trimIndent()
        }
    }

    Op("variance") {
        javaPackage = "org.nd4j.linalg.api.ops.impl.summarystats"
        legacy = true
        Input(NUMERIC, "x") { description = "Input variable" }
        Arg(BOOL, "biasCorrected") { description = "If true: divide by (N-1) (i.e., sample variable). If false: divide by N (population variance)" }
        Arg(BOOL, "keepDims") {  description = "If true: keep the dimensions that are reduced on (as size 1). False: remove the reduction dimensions"; defaultValue=false }
        Arg(INT, "dimensions") { count=AtLeast(0); description = "Dimensions to reduce over. If dimensions are not specified, full array reduction is performed" }
        Output(NUMERIC, "output"){ description = "reduced array of rank (input rank - num dimensions)" }
        Doc(Language.ANY, DocScope.ALL){
            """
                Variance array reduction operation, optionally along specified dimensions
            """.trimIndent()
        }
        useMixin(keepDimsDoc)
    }

    Op("zerosLike") {
        javaPackage = "org.nd4j.linalg.api.ops.impl.shape"
        Input(NUMERIC, "input") { description = "Input " }
        Output(NUMERIC, "output"){ description = "A new Variable with the same (dynamic) shape as the input" }

        Doc(Language.ANY, DocScope.ALL){
            """
                Return a variable of all 0s, with the same shape as the input variable. Note that this is dynamic:
                if the input shape changes in later execution, the returned variable's shape will also be updated
            """.trimIndent()
        }
    }

    Op("any") {
        javaPackage = "org.nd4j.linalg.api.ops.impl.reduce.bool"
        legacy = true
        Input(NDARRAY, "x") { description = " Input variable" }
        Arg(INT, "dimensions"){ count = AtLeast(0); description = "Dimensions to reduce over. If dimensions are not specified, full array reduction is performed" }
        Output(BOOL, "output"){ description = "reduced array of rank (input rank - num dimensions)" }
        Doc(Language.ANY, DocScope.ALL){
            """
                Boolean or array reduction operation, optionally along specified dimensions
            """.trimIndent()
        }
    }

    Op("all") {
        javaPackage = "org.nd4j.linalg.api.ops.impl.reduce.bool"
        legacy = true
        Input(NDARRAY, "x") { description = "Input variable" }
        Arg(INT, "dimensions"){ count = AtLeast(0); description = "Dimensions to reduce over. If dimensions are not specified, full array reduction is performed" }
        Output(BOOL, "output"){ description = "reduced array of rank (input rank - num dimensions)" }
        Doc(Language.ANY, DocScope.ALL){
            """
                Boolean and array reduction operation, optionally along specified dimensions
            """.trimIndent()
        }
    }

    Op("castTo"){
        javaPackage = "org.nd4j.linalg.api.ops.impl.transforms.dtype"
        javaOpClass = "Cast"
        Input(NDARRAY, "arg") { description = "Input variable to cast"}
        Arg(DATA_TYPE, "datatype"){ description = "Datatype to cast to"}
        Output(NDARRAY, "output"){ description = "Output array (after casting)"}
        Doc(Language.ANY, DocScope.ALL){
            """
                Cast the array to a new datatype - for example, Integer -> Float
            """.trimIndent()
        }
    }


    Op("batchMmul"){
        javaPackage = "org.nd4j.linalg.api.ops.impl.reduce.custom"
        Input(NUMERIC, "inputsA"){ count = AtLeast(1); description = "First array of input matrices, all of shape (M, N) or (N, M)"}
        Input(NUMERIC, "inputsB"){ count = AtLeast(1); description = " Second array of input matrices, all of shape (N, K) or (K, N)"}
        Arg(BOOL, "transposeA"){ description = "Whether to transpose A arrays or not"; defaultValue=false}
        Arg(BOOL, "transposeB"){ description = "Whether to transpose B arrays or not"; defaultValue=false}
        Output(NUMERIC, "output1"){multiOutput=true; description = "Array of multiplied SDVariables of shape (M, K)"}

        Doc(Language.ANY, DocScope.ALL){
            """
                Matrix multiply a batch of matrices. matricesA and matricesB have to be arrays of same
                length and each pair taken from these sets has to have dimensions (M, N) and (N, K),
                respectively. If transposeA is true, matrices from matricesA will have shape (N, M) instead.
                Likewise, if transposeB is true, matrices from matricesB will have shape (K, N).
                
                The result of this operation will be a batch of multiplied matrices. The
                result has the same length as both input batches and each output matrix is of shape (M, K).
            """.trimIndent()
        }
    }

    Op("unstack"){
        javaPackage = "org.nd4j.linalg.api.ops.impl.shape"
        Input(NDARRAY, "value"){ description="Input variable to unstack"}
        Arg(INT, "axis"){description = "Axis to unstack on"}
        Arg(INT, "num"){ description = "Number of output variables"}

        Doc(Language.ANY, DocScope.ALL){
            """
                Unstack a variable of rank X into N rank X-1 variables by taking slices along the specified axis.
                If input has shape [a,b,c] then output has shape:
                axis = 0: [b,c]
                axis = 1: [a,c]
                axis = 2: [a,b]
            """.trimIndent()
        }
    }
}
