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
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.BaseOp;
import org.nd4j.linalg.api.ops.Op;
import org.nd4j.linalg.api.ops.OpContext;
import org.nd4j.linalg.api.ops.ScalarOp;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.factory.Nd4jBackend;

import java.util.List;

import static org.junit.Assert.*;

/**
 * Test to verify scalar values are preserved during dup() and constant folding.
 */
@Tag(TagNames.DL4J_OLD_API)
public class TestScalarOpDup extends BaseNd4jTestWithBackends {

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
    public void testScalarValuePreservedAfterDup(Nd4jBackend nd4jBackend) {
        // Create a simple graph with a scalar add
        SameDiff sd = SameDiff.create();
        SDVariable c = sd.constant("c", Nd4j.scalar(1.0));
        SDVariable add = c.add("add", 1.0); // scalar add with value 1.0

        // Print all ops to see what names they have
        System.out.println("All ops in original graph:");
        for (String opName : sd.getOps().keySet()) {
            SameDiffOp op = sd.getOps().get(opName);
            System.out.println("  Op name: " + opName + ", Op type: " + (op.getOp() != null ? op.getOp().opName() : "null"));
        }

        // Get the scalar op from the original graph - it might not be named "add"
        // Find the op that produces the "add" variable
        String opNameForAdd = sd.getVariables().get("add").getOutputOfOp();
        System.out.println("Op that produces 'add' variable: " + opNameForAdd);

        SameDiffOp originalOp = sd.getOps().get(opNameForAdd);
        assertNotNull("Original op should exist", originalOp);
        DifferentialFunction originalDf = originalOp.getOp();
        assertTrue("Op should be a ScalarOp", originalDf instanceof ScalarOp);

        ScalarOp originalScalarOp = (ScalarOp) originalDf;
        INDArray originalScalar = originalScalarOp.scalar();
        System.out.println("Original scalar value: " + (originalScalar != null ? originalScalar.getDouble(0) : "null"));

        assertNotNull("Original scalar should not be null", originalScalar);
        assertEquals("Original scalar should be 1.0", 1.0, originalScalar.getDouble(0), 1e-5);

        // Duplicate the graph
        SameDiff copy = sd.dup();

        // Get the scalar op from the duplicated graph - use the same lookup pattern
        String copyOpNameForAdd = copy.getVariables().get("add").getOutputOfOp();
        System.out.println("Op that produces 'add' variable in copy: " + copyOpNameForAdd);

        SameDiffOp copyOp = copy.getOps().get(copyOpNameForAdd);
        assertNotNull("Copy op should exist", copyOp);
        DifferentialFunction copyDf = copyOp.getOp();
        assertTrue("Copy op should be a ScalarOp", copyDf instanceof ScalarOp);

        ScalarOp copyScalarOp = (ScalarOp) copyDf;
        INDArray copyScalar = copyScalarOp.scalar();
        System.out.println("Copy scalar value: " + (copyScalar != null ? copyScalar.getDouble(0) : "null"));

        assertNotNull("Copy scalar should not be null", copyScalar);
        assertEquals("Copy scalar should be 1.0", 1.0, copyScalar.getDouble(0), 1e-5);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testConstantFoldingManual(Nd4jBackend nd4jBackend) {
        // Manually test what constant folding does
        SameDiff sd = SameDiff.create();
        SDVariable c = sd.constant("c", Nd4j.scalar(1.0));
        SDVariable add = c.add("add", 1.0);

        // Get the duplicated graph
        SameDiff copy = sd.dup();

        // Print all ops in the copy
        System.out.println("All ops in copy graph:");
        for (String opName : copy.getOps().keySet()) {
            SameDiffOp op = copy.getOps().get(opName);
            System.out.println("  Op name: " + opName + ", Op type: " + (op.getOp() != null ? op.getOp().opName() : "null"));
        }

        // Get the scalar op - find the op that produces "add"
        String opNameForAdd = copy.getVariables().get("add").getOutputOfOp();
        System.out.println("Op that produces 'add' variable: " + opNameForAdd);

        SameDiffOp op = copy.getOps().get(opNameForAdd);
        assertNotNull("Op should exist", op);
        DifferentialFunction df = op.getOp();

        System.out.println("Op class: " + df.getClass().getName());
        System.out.println("Op name: " + df.opName());

        if (df instanceof ScalarOp) {
            ScalarOp scalarOp = (ScalarOp) df;
            INDArray scalar = scalarOp.scalar();
            System.out.println("Scalar value before clearArrays: " + (scalar != null ? scalar.getDouble(0) : "null"));
        }

        // Clear arrays (what constant folding does)
        df.clearArrays();

        if (df instanceof ScalarOp) {
            ScalarOp scalarOp = (ScalarOp) df;
            INDArray scalar = scalarOp.scalar();
            System.out.println("Scalar value after clearArrays: " + (scalar != null ? scalar.getDouble(0) : "null"));
        }

        // Set x to the constant value
        INDArray inputArr = copy.getVariable("c").getArr();
        System.out.println("Input array value: " + inputArr.getDouble(0));

        Op o = (Op) df;
        o.setX(inputArr);

        // Execute the op using OpContext (like ConstantFunctionOptimizations does)
        OpContext ctx = Nd4j.getExecutioner().buildContext();
        try {
            ctx.setInputArrays(o.x());

            // Calculate output shape and allocate output array
            List<DataBuffer> outputShape = ((BaseOp) o).calculateOutputShape(ctx);
            if (outputShape != null && !outputShape.isEmpty()) {
                DataBuffer shapeDesc = outputShape.get(0);
                INDArray z = Nd4j.createFromDescriptor(shapeDesc);
                ctx.setOutputArray(0, z);
            }

            // Execute with context - this properly handles ScalarOps
            Nd4j.getExecutioner().exec(o, ctx);

            // Get the output
            INDArray output = ctx.getOutputArray(0);
            System.out.println("Output value: " + (output != null ? output.getDouble(0) : "null"));

            // The expected output is 1.0 + 1.0 = 2.0
            assertNotNull("Output should not be null", output);
            assertEquals("Output should be 2.0 (1.0 + 1.0)", 2.0, output.getDouble(0), 1e-5);
        } finally {
            try { ctx.close(); } catch (Exception ignored) {}
        }
    }
}
