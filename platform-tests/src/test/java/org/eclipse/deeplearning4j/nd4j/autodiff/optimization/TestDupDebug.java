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
package org.eclipse.deeplearning4j.nd4j.autodiff.optimization;

import org.junit.jupiter.api.Tag;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.MethodSource;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.autodiff.samediff.internal.SameDiffOp;
import org.nd4j.common.tests.tags.TagNames;
import org.nd4j.linalg.BaseNd4jTestWithBackends;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.factory.Nd4jBackend;

import java.util.Collections;
import java.util.List;

import static org.junit.Assert.*;

/**
 * Debug test to understand why dup() breaks subtraction.
 */
@Tag(TagNames.DL4J_OLD_API)
public class TestDupDebug extends BaseNd4jTestWithBackends {

    @Override
    public char ordering() {
        return 'c';
    }

    @Override
    public long getTimeoutMilliseconds() {
        return 1_000_000_000L;
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testDupDebug(Nd4jBackend nd4jBackend) {
        // c = 1.0 (constant)
        // add = c + 1 = 2.0
        // v = 1.0 (variable)
        // out = v - add = 1.0 - 2.0 = -1.0
        SameDiff sd = SameDiff.create();
        SDVariable c = sd.constant("c", Nd4j.scalar(1.0));
        SDVariable c2 = c.add("add", 1);
        SDVariable v = sd.var("variable", Nd4j.scalar(1.0));
        SDVariable out = v.sub("out", c2);

        System.out.println("=== ORIGINAL GRAPH ===");
        printGraphDetails(sd);

        INDArray originalResult = sd.outputSingle(Collections.emptyMap(), "out");
        System.out.println("Original result: " + originalResult);

        System.out.println("\n=== DUPLICATING ===");
        SameDiff copy = sd.dup();

        System.out.println("\n=== DUPLICATED GRAPH ===");
        printGraphDetails(copy);

        INDArray copyResult = copy.outputSingle(Collections.emptyMap(), "out");
        System.out.println("Copy result: " + copyResult);

        assertEquals("Original should be -1.0", -1.0, originalResult.getDouble(0), 1e-5);
        assertEquals("Copy should be -1.0", -1.0, copyResult.getDouble(0), 1e-5);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testSimpleSubtractDup(Nd4jBackend nd4jBackend) {
        // Simplest possible subtraction: a - b
        SameDiff sd = SameDiff.create();
        SDVariable a = sd.constant("a", Nd4j.scalar(1.0));
        SDVariable b = sd.constant("b", Nd4j.scalar(2.0));
        SDVariable out = a.sub("out", b);

        System.out.println("=== SIMPLE SUBTRACTION - ORIGINAL ===");
        printGraphDetails(sd);

        INDArray originalResult = sd.outputSingle(Collections.emptyMap(), "out");
        System.out.println("Original result (1-2=-1): " + originalResult);

        SameDiff copy = sd.dup();

        System.out.println("\n=== SIMPLE SUBTRACTION - COPY ===");
        printGraphDetails(copy);

        INDArray copyResult = copy.outputSingle(Collections.emptyMap(), "out");
        System.out.println("Copy result (should be -1): " + copyResult);

        assertEquals("Original should be -1.0", -1.0, originalResult.getDouble(0), 1e-5);
        assertEquals("Copy should be -1.0", -1.0, copyResult.getDouble(0), 1e-5);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testSimpleAddDup(Nd4jBackend nd4jBackend) {
        // Simple addition: a + b (should be symmetric, so order doesn't matter)
        SameDiff sd = SameDiff.create();
        SDVariable a = sd.constant("a", Nd4j.scalar(1.0));
        SDVariable b = sd.constant("b", Nd4j.scalar(2.0));
        SDVariable out = a.add("out", b);

        System.out.println("=== SIMPLE ADDITION - ORIGINAL ===");
        printGraphDetails(sd);

        INDArray originalResult = sd.outputSingle(Collections.emptyMap(), "out");
        System.out.println("Original result (1+2=3): " + originalResult);

        SameDiff copy = sd.dup();

        System.out.println("\n=== SIMPLE ADDITION - COPY ===");
        printGraphDetails(copy);

        INDArray copyResult = copy.outputSingle(Collections.emptyMap(), "out");
        System.out.println("Copy result (should be 3): " + copyResult);

        assertEquals("Original should be 3.0", 3.0, originalResult.getDouble(0), 1e-5);
        assertEquals("Copy should be 3.0", 3.0, copyResult.getDouble(0), 1e-5);
    }

    private void printGraphDetails(SameDiff sd) {
        System.out.println("Variables:");
        for (SDVariable v : sd.variables()) {
            System.out.println("  " + v.name() + " [" + v.getVariableType() + "]");
            INDArray arr = null;
            try {
                arr = v.getArr();
            } catch (Exception e) {
                // Ignore
            }
            if (arr != null) {
                System.out.println("    value: " + arr);
            }
        }

        System.out.println("Ops:");
        for (SameDiffOp op : sd.getOps().values()) {
            System.out.println("  " + op.getName() + " [" + (op.getOp() != null ? op.getOp().opName() : "null") + "]");
            List<String> inputs = op.getInputsToOp();
            System.out.println("    inputs: " + inputs);
            List<String> outputs = op.getOutputsOfOp();
            System.out.println("    outputs: " + outputs);

            if (op.getOp() != null) {
                try {
                    SDVariable[] args = op.getOp().args();
                    if (args != null) {
                        System.out.print("    args() order: [");
                        for (int i = 0; i < args.length; i++) {
                            if (i > 0) System.out.print(", ");
                            System.out.print(args[i] != null ? args[i].name() : "null");
                        }
                        System.out.println("]");
                    }
                } catch (Exception e) {
                    System.out.println("    args() error: " + e.getMessage());
                }
            }
        }
    }
}
