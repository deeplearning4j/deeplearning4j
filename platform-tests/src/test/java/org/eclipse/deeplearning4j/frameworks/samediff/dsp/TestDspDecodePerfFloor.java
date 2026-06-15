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
package org.eclipse.deeplearning4j.frameworks.samediff.dsp;

import lombok.extern.slf4j.Slf4j;
import org.eclipse.deeplearning4j.frameworks.samediff.dsp.regression.DspRegressionHarness;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Environment;
import org.nd4j.linalg.factory.Nd4j;

import java.util.Collections;
import java.util.Map;

import static org.junit.jupiter.api.Assumptions.assumeTrue;

/**
 * Regression floor for end-to-end DSP decode throughput.
 *
 * <p>This is the single most important test in this package — it exists specifically because
 * the current batch-zero / cross-stream-sync perf disaster would have been caught automatically
 * if this assertion had been in place. The fixture is deliberately tiny (not full VLM) so it
 * runs in seconds in CI while still exercising the DSP compile+replay pipeline.</p>
 *
 * <p>Run a single config with:</p>
 * <pre>{@code
 *   mvn test -Dtest=TestDspDecodePerfFloor#testSyntheticDecodeTokPerSecFloor \
 *            -Ddsp.perf.minTokPerSec=70
 * }</pre>
 */
@Slf4j
@Tag("dsp")
@Tag("perf")
public class TestDspDecodePerfFloor extends DspRegressionHarness {

    private static final int HIDDEN = 128;
    private static final int VOCAB  = 1024;

    @Override
    protected SameDiff buildFixture() {
        return buildTinyDecodeFixture(HIDDEN, VOCAB);
    }

    @BeforeEach
    public void enableDspAutoCompile() {
        // DSP auto-compile is off by default — we must opt in or the fixture runs on the
        // standard path and the test does not actually measure DSP throughput.
        Environment env = Nd4j.getEnvironment();
        env.setDspBatchZero(true);
        env.setDspBatchedGemm(true);
        env.setTritonGraphCapture(true);
        env.setTritonConsolidatedArgTable(true);
        env.setTritonArgDirtyTracking(true);
    }

    /**
     * Run a short synthetic decode loop and assert tok/s is above the configured floor.
     * Floor defaults to 70 tok/s but can be overridden with {@code -Ddsp.perf.minTokPerSec=...}.
     */
    @Test
    public void testSyntheticDecodeTokPerSecFloor() {
        assumeTrue(isCudaAvailable(), "CUDA backend required for DSP perf floor test");

        SameDiff sd = buildFixture();
        INDArray x = Nd4j.randn(DataType.FLOAT, 1, HIDDEN);
        Map<String, INDArray> inputs = Collections.singletonMap("x", x);

        int warmup     = 10;
        int iterations = 100;
        double floor   = defaultTokPerSecFloor();

        log.info("TestDspDecodePerfFloor: warmup={}, iterations={}, floor={} tok/s",
                warmup, iterations, floor);

        assertTokPerSecFloor(sd, inputs, "logits", warmup, iterations, floor);
    }
}
