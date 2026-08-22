/*
 *  ******************************************************************************
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
 *  * SPDX-License-Identifier: Apache-2.0
 *  *****************************************************************************
 */

package org.eclipse.deeplearning4j.nd4j.linalg.api;

import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.autodiff.samediff.execution.DspPlanAssertions;
import org.nd4j.autodiff.samediff.execution.GraphExecutionMode;
import org.nd4j.common.tests.tags.TagNames;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.executioner.OpExecutioner;
import org.nd4j.linalg.api.ops.CustomOp;
import org.nd4j.linalg.api.ops.DynamicCustomOp;
import org.nd4j.linalg.factory.Nd4j;

import java.util.HashMap;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

/** Numerical and strict-lowering tests for the TPU graph backend. */
@Tag(TagNames.TPU)
public class TpuGraphBackendTest {

    @Test
    public void testTpuExecutionContractsExist() {
        assertEquals("TPU", OpExecutioner.ExecutionerType.TPU.name());
        assertEquals(13, GraphExecutionMode.TPU.getNativeCode());
        assertEquals(GraphExecutionMode.TPU, GraphExecutionMode.fromNativeCode(13));
    }

    @Test
    public void testStableHloAddMultiplyExecutesAndReplays() throws Exception {
        assumeTpuBackend();

        SameDiff sameDiff = SameDiff.create();
        SDVariable x = sameDiff.placeHolder("x", DataType.FLOAT, 2, 3);
        SDVariable y = sameDiff.placeHolder("y", DataType.FLOAT, 3);
        SDVariable sum = x.add("sum", y);
        sum.mul("output", y);
        sameDiff.setOutputs("output");
        sameDiff.setGraphExecutionMode(GraphExecutionMode.TPU);

        INDArray xValue = Nd4j.linspace(1, 6, 6, DataType.FLOAT).reshape(2, 3);
        INDArray yValue = Nd4j.createFromArray(2.0f, 3.0f, 4.0f);
        INDArray expected = xValue.add(yValue).mul(yValue);
        Map<String, INDArray> inputs = new HashMap<>();
        inputs.put("x", xValue);
        inputs.put("y", yValue);

        for (int i = 0; i < 8; ++i) {
            INDArray actual = sameDiff.output(inputs, "output").get("output");
            assertNotNull(actual);
            assertTrue(expected.equalsWithEps(actual, 1e-5),
                    "StableHLO/PJRT output differs on replay " + i);
        }
        assertTrue(DspPlanAssertions.getSegmentCompiledBackend(sameDiff, 0)
                        .contains("TPU"),
                "DSP segment was not compiled by the TPU backend");
        DspPlanAssertions.assertSegmentReplayCountAtLeast(
                sameDiff, 0, 3, "TPU StableHLO repeated execution");
    }

    @Test
    public void testReluThresholdIsPreservedByStableHlo() throws Exception {
        assumeTpuBackend();

        SameDiff sameDiff = SameDiff.create();
        SDVariable input = sameDiff.placeHolder("input", DataType.FLOAT, 3);
        sameDiff.nn().relu("output", input, 1.5);
        sameDiff.setOutputs("output");
        sameDiff.setGraphExecutionMode(GraphExecutionMode.TPU);

        Map<String, INDArray> inputs = new HashMap<>();
        inputs.put("input", Nd4j.createFromArray(-1.0f, 0.0f, 2.0f));
        INDArray expected = Nd4j.createFromArray(1.5f, 1.5f, 2.0f);
        INDArray actual = sameDiff.output(inputs, "output").get("output");
        assertTrue(expected.equalsWithEps(actual, 1e-5),
                "StableHLO ReLU must preserve the ND4J threshold argument");
    }

    @Test
    public void testEagerCustomOpUsesSameStableHloPath() throws Exception {
        assumeTpuBackend();
        INDArray left = Nd4j.createFromArray(1.0f, 2.0f, 3.0f);
        INDArray right = Nd4j.createFromArray(4.0f, 5.0f, 6.0f);
        INDArray output = Nd4j.createUninitialized(DataType.FLOAT, 3);
        CustomOp op = DynamicCustomOp.builder("add")
                .addInputs(left, right)
                .addOutputs(output)
                .build();
        Nd4j.getExecutioner().exec(op);
        assertTrue(Nd4j.createFromArray(5.0f, 7.0f, 9.0f)
                        .equalsWithEps(output, 1e-5),
                "TPU eager custom op did not execute through StableHLO/PJRT");
    }

    @Test
    public void testForcedTpuRejectsUnsupportedOpWithoutHostFallback() throws Exception {
        assumeTpuBackend();

        SameDiff sameDiff = SameDiff.create();
        SDVariable input = sameDiff.placeHolder("input", DataType.FLOAT, 2, 3);
        sameDiff.argmax("output", input, 1);
        sameDiff.setOutputs("output");
        sameDiff.setGraphExecutionMode(GraphExecutionMode.TPU);

        Map<String, INDArray> inputs = new HashMap<>();
        inputs.put("input", Nd4j.ones(DataType.FLOAT, 2, 3));
        RuntimeException failure = assertThrows(RuntimeException.class,
                () -> sameDiff.output(inputs, "output"),
                "Forced TPU mode must fail closed when no complete lowering exists");
        String message = fullMessage(failure).toLowerCase();
        assertTrue(message.contains("tpu") || message.contains("lower")
                        || message.contains("compile"),
                "Failure must identify TPU compilation/lowering, not an unrelated error: "
                        + message);
    }

    private void assumeTpuBackend() throws Exception {
        Class<?> backendClass = Class.forName("org.nd4j.linalg.jtpu.JTpuBackend");
        Object backend = backendClass.getDeclaredConstructor().newInstance();
        assumeTrue((Boolean) backendClass.getMethod("canRun").invoke(backend),
                "No addressable TPU PJRT device is available");
    }

    private String fullMessage(Throwable failure) {
        StringBuilder result = new StringBuilder();
        for (Throwable current = failure; current != null; current = current.getCause()) {
            if (current.getMessage() != null) result.append(current.getMessage()).append(' ');
        }
        return result.toString();
    }
}
