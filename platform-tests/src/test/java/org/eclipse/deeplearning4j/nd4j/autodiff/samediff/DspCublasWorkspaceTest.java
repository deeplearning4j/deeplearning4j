/*
 *  ******************************************************************************
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
 *  *  SPDX-License-Identifier: Apache-2.0
 *  *****************************************************************************
 */
package org.eclipse.deeplearning4j.nd4j.autodiff.samediff;

import lombok.extern.slf4j.Slf4j;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.Timeout;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.executioner.OpExecutioner;
import org.nd4j.linalg.factory.Environment;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.ops.transforms.Transforms;
import org.nd4j.nativeblas.NativeOps;

import java.util.Collections;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

/**
 * Every DSP plan on CUDA hands cuBLAS a workspace of its own. By default it is the size cuBLAS
 * recommends for the visible GPUs, and a plan that cannot allocate it fails with an error naming
 * it rather than running its GEMMs without one (pedantic math then returns zeros for FP16).
 */
@Slf4j
public class DspCublasWorkspaceTest {

    @Test
    void defaultIsTheSizeCublasRecommends() {
        assumeCuda();
        NativeOps ops = Nd4j.getNativeOps();
        int expected = 4;
        for (int d = 0; d < Nd4j.getAffinityManager().getNumberOfDevices(); d++) {
            if (ops.getDeviceMajor(d) >= 9) {
                expected = 32;
            }
        }
        assertEquals(expected, Nd4j.getEnvironment().dspCublasWorkspaceMb(),
                "32MB from Hopper (compute capability 9) on, 4MB before");
    }

    @Test
    void workspaceMustHaveASize() {
        assumeCuda();
        assertThrows(IllegalArgumentException.class, () -> Nd4j.getEnvironment().setDspCublasWorkspaceMb(0));
    }

    /**
     * A workspace twice the device's memory cannot be allocated: the execution must fail and say
     * so, and leave nothing behind, so the same graph and a new one then warm up, capture and
     * replay (a capture would wait forever on an execution the failed entry never finished).
     */
    @Test
    @Timeout(600)
    void workspaceTheDeviceCannotHoldFailsTheExecution() {
        assumeCuda();
        Environment environment = Nd4j.getEnvironment();
        int configured = environment.dspCublasWorkspaceMb();
        long deviceMb = Nd4j.getNativeOps().getDeviceTotalMemory(0) / (1024 * 1024);
        INDArray w = Nd4j.randn(DataType.FLOAT, 64, 32).muli(0.1);
        INDArray b = Nd4j.randn(DataType.FLOAT, 32).muli(0.1);
        INDArray in = Nd4j.randn(DataType.FLOAT, 2, 64);
        INDArray expected = Transforms.softmax(in.mmul(w).addRowVector(b), true);

        try (SameDiff failing = denseSoftmax(w, b)) {
            environment.setDspCublasWorkspaceMb((int) Math.min(Integer.MAX_VALUE, deviceMb * 2));
            RuntimeException failure;
            try {
                failure = assertThrows(RuntimeException.class,
                        () -> failing.output(Collections.singletonMap("input", in), "output"));
            } finally {
                environment.setDspCublasWorkspaceMb(configured);
            }
            log.info("Execution without its cuBLAS workspace failed with: {}", messages(failure));
            assertTrue(messages(failure).contains("cuBLAS workspace"),
                    "the failure must name the workspace: " + messages(failure));

            try (SameDiff fresh = denseSoftmax(w, b)) {
                for (int i = 0; i < 4; i++) {
                    assertOutput(expected, fresh, in);
                }
            }
            for (int i = 0; i < 4; i++) {
                assertOutput(expected, failing, in);
            }
        }
    }

    private static SameDiff denseSoftmax(INDArray w, INDArray b) {
        SameDiff sd = SameDiff.create();
        SDVariable input = sd.placeHolder("input", DataType.FLOAT, -1, 64);
        SDVariable weights = sd.var("w", w.dup());
        SDVariable bias = sd.var("b", b.dup());
        sd.nn.softmax("output", input.mmul(weights).add(bias), -1);
        return sd;
    }

    private static void assertOutput(INDArray expected, SameDiff sd, INDArray in) {
        INDArray out = sd.output(Collections.singletonMap("input", in), "output").get("output");
        assertTrue(expected.equalsWithEps(out, 1e-5), "expected " + expected + " but got " + out);
        out.close();
    }

    private static String messages(Throwable failure) {
        StringBuilder all = new StringBuilder();
        for (Throwable t = failure; t != null; t = t.getCause()) {
            all.append(t.getMessage()).append(" | ");
        }
        return all.toString();
    }

    private static void assumeCuda() {
        assumeTrue(Nd4j.getExecutioner().type() == OpExecutioner.ExecutionerType.CUDA,
                "the cuBLAS workspace is the CUDA backend's");
    }
}
