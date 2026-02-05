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
import org.nd4j.autodiff.functions.DifferentialFunction;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.autodiff.samediff.internal.SameDiffOp;
import org.nd4j.common.tests.tags.TagNames;
import org.nd4j.linalg.BaseNd4jTestWithBackends;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.ScalarOp;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.factory.Nd4jBackend;

import java.util.Collections;
import java.util.List;

import static org.junit.Assert.*;

/**
 * Debug test to understand why dup() breaks scalar add.
 */
@Tag(TagNames.DL4J_OLD_API)
public class TestDupDebug2 extends BaseNd4jTestWithBackends {

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
    public void testScalarAddDup(Nd4jBackend nd4jBackend) {
        // Simple scalar add: c + 1
        SameDiff sd = SameDiff.create();
        SDVariable c = sd.constant("c", Nd4j.scalar(1.0));
        SDVariable add = c.add("add", 1);

        System.out.println("=== SCALAR ADD - ORIGINAL ===");
        printOpDetails(sd, "add");

        INDArray originalResult = sd.outputSingle(Collections.emptyMap(), "add");
        System.out.println("Original result (1+1=2): " + originalResult);

        SameDiff copy = sd.dup();

        System.out.println("\n=== SCALAR ADD - COPY ===");
        printOpDetails(copy, "add");

        INDArray copyResult = copy.outputSingle(Collections.emptyMap(), "add");
        System.out.println("Copy result (should be 2): " + copyResult);

        assertEquals("Original should be 2.0", 2.0, originalResult.getDouble(0), 1e-5);
        assertEquals("Copy should be 2.0", 2.0, copyResult.getDouble(0), 1e-5);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testScalarSubDup(Nd4jBackend nd4jBackend) {
        // Simple scalar sub: c - 1
        SameDiff sd = SameDiff.create();
        SDVariable c = sd.constant("c", Nd4j.scalar(5.0));
        SDVariable sub = c.sub("sub", 2);

        System.out.println("=== SCALAR SUB - ORIGINAL ===");
        printOpDetails(sd, "sub");

        INDArray originalResult = sd.outputSingle(Collections.emptyMap(), "sub");
        System.out.println("Original result (5-2=3): " + originalResult);

        SameDiff copy = sd.dup();

        System.out.println("\n=== SCALAR SUB - COPY ===");
        printOpDetails(copy, "sub");

        INDArray copyResult = copy.outputSingle(Collections.emptyMap(), "sub");
        System.out.println("Copy result (should be 3): " + copyResult);

        assertEquals("Original should be 3.0", 3.0, originalResult.getDouble(0), 1e-5);
        assertEquals("Copy should be 3.0", 3.0, copyResult.getDouble(0), 1e-5);
    }

    private void printOpDetails(SameDiff sd, String opName) {
        System.out.println("All ops in graph: " + sd.getOps().keySet());
        SameDiffOp op = sd.getOps().get(opName);
        if (op == null) {
            System.out.println("Op not found: " + opName);
            // Try to find by output name
            for (SameDiffOp sop : sd.getOps().values()) {
                if (sop.getOutputsOfOp() != null && sop.getOutputsOfOp().contains(opName)) {
                    op = sop;
                    System.out.println("Found op by output name: " + sop.getName());
                    break;
                }
            }
        }
        if (op == null) return;

        DifferentialFunction df = op.getOp();
        System.out.println("Op: " + opName);
        System.out.println("  Type: " + (df != null ? df.getClass().getSimpleName() : "null"));
        System.out.println("  opName: " + (df != null ? df.opName() : "null"));
        System.out.println("  opType: " + (df != null ? df.opType() : "null"));
        System.out.println("  inputs: " + op.getInputsToOp());
        System.out.println("  outputs: " + op.getOutputsOfOp());

        if (df instanceof ScalarOp) {
            ScalarOp scalarOp = (ScalarOp) df;
            INDArray scalarValue = scalarOp.scalar();
            System.out.println("  SCALAR VALUE: " + (scalarValue != null ? scalarValue : "NULL"));
        }
    }
}
