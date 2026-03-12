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

package org.nd4j.linalg.api.ops.impl.transforms.custom;

import org.nd4j.common.base.Preconditions;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.DynamicCustomOp;

import java.util.ArrayList;
import java.util.Collections;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

/**
 * Einsum (Einstein summation) operation.
 *
 * Einsum provides a powerful way to express tensor operations using Einstein summation notation.
 * The equation string specifies the subscripts for each input tensor and the output tensor.
 *
 * Examples:
 * - Matrix multiplication: "ij,jk->ik"
 * - Transpose: "ij->ji"
 * - Diagonal: "ii->i"
 * - Trace: "ii->"
 * - Batch matmul: "bij,bjk->bik"
 * - Dot product: "i,i->"
 * - Outer product: "i,j->ij"
 *
 * @author Adam Gibson
 */
public class Einsum extends DynamicCustomOp {

    private String equation;

    public Einsum() {
    }

    public Einsum(SameDiff sd, SDVariable[] inputs, String equation) {
        super(sd, inputs, false);
        this.equation = equation;
        addSArgument(equation);
    }

    public Einsum(INDArray[] inputs, String equation) {
        super(inputs, null);
        this.equation = equation;
        addSArgument(equation);
    }

    @Override
    public String opName() {
        return "einsum";
    }

    @Override
    public List<SDVariable> doDiff(List<SDVariable> gradients) {
        String eq = equation;
        if (eq == null && numSArguments() > 0) {
            eq = getSArgument(0);
        }

        Preconditions.checkState(eq != null && !eq.isEmpty(), "No einsum equation is available for backprop");

        String normalized = eq.replace(" ", "");
        int arrowIdx = normalized.indexOf("->");
        Preconditions.checkState(arrowIdx > 0 && arrowIdx < normalized.length() - 1,
                "Einsum backprop currently requires an explicit non-empty output equation, got \"%s\"", normalized);
        Preconditions.checkState(!normalized.contains("..."),
                "Einsum backprop does not yet support ellipsis equations, got \"%s\"", normalized);

        String[] inputSubs = normalized.substring(0, arrowIdx).split(",");
        String outputSub = normalized.substring(arrowIdx + 2);
        Preconditions.checkState(inputSubs.length == args().length,
                "Expected %s einsum inputs for equation \"%s\", got %s", inputSubs.length, normalized, args().length);

        validateSubscripts(normalized, outputSub, "output");
        for (int i = 0; i < inputSubs.length; i++) {
            validateSubscripts(normalized, inputSubs[i], "input " + i);
        }

        List<SDVariable> ret = new ArrayList<>(inputSubs.length);
        for (int i = 0; i < inputSubs.length; i++) {
            List<SDVariable> gradInputs = new ArrayList<>(args().length);
            List<String> gradSubs = new ArrayList<>(args().length);
            Set<Character> availableLabels = new HashSet<>();

            gradInputs.add(gradients.get(0));
            gradSubs.add(outputSub);
            addLabels(outputSub, availableLabels);

            for (int j = 0; j < inputSubs.length; j++) {
                if (j == i) {
                    continue;
                }

                gradInputs.add(arg(j));
                gradSubs.add(inputSubs[j]);
                addLabels(inputSubs[j], availableLabels);
            }

            ensureGradientSupported(normalized, inputSubs[i], availableLabels);

            String gradEquation = String.join(",", gradSubs) + "->" + inputSubs[i];
            ret.add(sameDiff.linalg().einsum(gradInputs.toArray(new SDVariable[0]), gradEquation));
        }

        return ret;
    }

    @Override
    public void configureFromArguments() {
        super.configureFromArguments();
        if (numSArguments() > 0) {
            this.equation = getSArgument(0);
        }
    }

    @Override
    public int getNumOutputs() {
        return 1;
    }

    @Override
    public List<DataType> calculateOutputDataTypes(List<DataType> dataTypes) {
        // Output type is same as first input type
        if (dataTypes != null && !dataTypes.isEmpty()) {
            return Collections.singletonList(dataTypes.get(0));
        }
        return Collections.singletonList(DataType.FLOAT);
    }

    private static void validateSubscripts(String equation, String subscripts, String operandDescription) {
        Set<Character> labels = new HashSet<>();
        for (int i = 0; i < subscripts.length(); i++) {
            char c = subscripts.charAt(i);
            Preconditions.checkState(Character.isLetter(c),
                    "Einsum backprop only supports alphabetic subscripts, got \"%s\" in %s for equation \"%s\"",
                    c, operandDescription, equation);
            Preconditions.checkState(labels.add(c),
                    "Einsum backprop does not yet support repeated labels in %s for equation \"%s\"",
                    operandDescription, equation);
        }
    }

    private static void addLabels(String subscripts, Set<Character> labels) {
        for (int i = 0; i < subscripts.length(); i++) {
            labels.add(subscripts.charAt(i));
        }
    }

    private static void ensureGradientSupported(String equation, String targetSubscripts, Set<Character> availableLabels) {
        for (int i = 0; i < targetSubscripts.length(); i++) {
            char c = targetSubscripts.charAt(i);
            Preconditions.checkState(availableLabels.contains(c),
                    "Einsum backprop does not yet support labels that only appear in the differentiated input. "
                            + "Missing label \"%s\" for equation \"%s\"",
                    c, equation);
        }
    }
}
