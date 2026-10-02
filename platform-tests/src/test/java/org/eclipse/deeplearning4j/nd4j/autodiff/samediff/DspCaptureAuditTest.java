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
import org.nd4j.autodiff.samediff.execution.DspHandle;
import org.nd4j.autodiff.samediff.execution.GraphExecutionMode;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.executioner.OpExecutioner;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.ops.transforms.Transforms;

import java.util.Collections;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

/**
 * A CUDA graph capture audits every slot it records: how many graph nodes the slot added
 * and, for one that added none, whether that is host work the graph does not repeat. A
 * reshape that views a buffer the graph writes adds no node and needs none on replay, so
 * it is not host-only and the captured graph is complete.
 */
@Slf4j
public class DspCaptureAuditTest {

    @Test
    @Timeout(300)
    void viewReshapesBetweenKernelsAreNotHostOnly() {
        assumeTrue(Nd4j.getExecutioner().type() == OpExecutioner.ExecutionerType.CUDA,
                "the capture audit is the CUDA graph backend's");
        INDArray w1 = Nd4j.randn(DataType.FLOAT, 64, 64).muli(0.1);
        INDArray w2 = Nd4j.randn(DataType.FLOAT, 64, 64).muli(0.1);
        try (SameDiff sd = SameDiff.create()) {
            SDVariable x = sd.placeHolder("x", DataType.FLOAT, 4, 64);
            SDVariable hidden = sd.nn().relu("hidden", sd.mmul("mm1", x, sd.var("w1", w1.dup())), 0);
            SDVariable heads = sd.reshape("heads", hidden, 2, 2, 64);
            SDVariable activated = sd.math().tanh("activated", heads);
            SDVariable rows = sd.reshape("rows", activated, 4, 64);
            sd.mmul("out", rows, sd.var("w2", w2.dup()));
            sd.setGraphExecutionMode(GraphExecutionMode.CUDA_GRAPHS);
            sd.setDspAutoCompileEnabled(true);
            sd.setDspNativeAutoCompileEnabled(true);

            for (int i = 0; i < 12; i++) {
                INDArray in = Nd4j.randn(DataType.FLOAT, 4, 64);
                INDArray expected = Transforms.tanh(Transforms.relu(in.mmul(w1))).mmul(w2);
                INDArray out = sd.output(Collections.singletonMap("x", in), "out").get("out");
                assertTrue(expected.equalsWithEps(out, 1e-4),
                        "execution " + i + ": expected " + expected + " but got " + out);
            }

            DspHandle plan = sd.dsp();
            assertTrue(plan.isCompiled(), "the graph must run as a DSP plan");
            log.info("Capture audit: captured={}/{} replays={} hostOnly={} ({}) stats={}",
                    plan.numCapturedGraphSegments(), plan.numSegments(), plan.totalGraphReplays(),
                    plan.numHostOnlyOps(), plan.hostOnlyOpNames(), plan.captureStats());
            assertTrue(plan.numCapturedGraphSegments() > 0, "the plan must capture a CUDA graph");
            assertTrue(plan.totalGraphReplays() > 0, "the captured graph must replay");
            assertEquals(0, plan.numHostOnlyOps(),
                    "reshapes that view the graph's own buffers are not host-only: " + plan.hostOnlyOpNames());
            assertTrue(plan.isCaptureComplete(), "the captured graph covers every slot");
        }
    }
}
