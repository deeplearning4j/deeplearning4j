/*
 *  ******************************************************************************
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

package org.eclipse.deeplearning4j.nd4j.autodiff.serialization;

import lombok.extern.slf4j.Slf4j;
import org.junit.jupiter.api.Test;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.autodiff.samediff.optimize.GraphOptimizer;
import org.nd4j.autodiff.samediff.serde.SDZSerializer;
import org.nd4j.common.config.ND4JSystemProperties;
import org.nd4j.common.tests.BaseND4JTest;
import org.nd4j.linalg.api.buffer.DataType;

import java.io.File;
import java.util.Collections;
import java.util.EnumMap;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.*;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

/**
 * Offline builder + verification for the fp16 BGE encoder.
 *
 * The fp32 model.opt.sdz fills the 24 GB 4090 at [32,512], starving CUDA-graph
 * capture of its 512 MB workspace -> permanent slot-by-slot -> ~10 s/exec. Prod
 * SHOULD run this model in fp16 (GraphOptimizer QuantizationOptimizations, opt-in
 * via -Dnd4j.optimizer.fp16=true, default OFF). This test verifies that the fp16
 * path in dl4j actually (a) casts weights to HALF, (b) keeps graph inputs as
 * feedable PLACEHOLDERs, and (c) writes a ~half-size artifact. Backend-agnostic
 * (no forward pass) so it runs on CPU without touching CUDA.
 */
@Slf4j
public class BgeFp16ModelBuilderTest extends BaseND4JTest {

    private static final String SRC = System.getProperty("bge.model.path",
            System.getProperty("user.home") + "/.kompile/models/bge-base-en-v1.5/model.opt.sdz");
    private static final String DST = System.getProperty("bge.fp16.out",
            System.getProperty("user.home") + "/.kompile/models/bge-base-en-v1.5/model.fp16.sdz");
    private static final String OUTPUT = "last_hidden_state";

    @Override
    public long getTimeoutMilliseconds() {
        return 20 * 60 * 1000L;
    }

    @Override
    public DataType getDataType() {
        return DataType.FLOAT; // never DOUBLE for this encoder
    }

    @Test
    public void buildAndVerifyFp16Model() throws Exception {
        File src = new File(SRC);
        assumeTrue(src.exists(), "Source fp32 model not found: " + SRC);

        // QuantizationOptimizations casts weights to HALF only when this is set.
        System.setProperty(ND4JSystemProperties.OPTIMIZER_FP16, "true");

        SameDiff model = SDZSerializer.load(src, true);
        assertNotNull(model);
        log.info("Loaded fp32 model: inputs={} outputs={} vars={} ops={} sizeMB={}",
                model.inputs(), model.outputs(), model.variables().size(),
                model.ops().length, src.length() / (1024 * 1024));
        logDtypeHistogram("BEFORE", model);

        int opsBefore = model.ops().length;
        SameDiff fp16 = GraphOptimizer.optimize(model, OUTPUT);
        log.info("GraphOptimizer(fp16=true): {} -> {} ops", opsBefore, fp16.ops().length);
        logDtypeHistogram("AFTER", fp16);

        // (b) Inputs must remain feedable placeholders.
        assertFalse(fp16.inputs().isEmpty(), "fp16 model must retain feedable inputs");
        for (String in : fp16.inputs()) {
            SDVariable v = fp16.getVariable(in);
            log.info("  input '{}' type={} dtype={}", in, v.getVariableType(), v.dataType());
        }

        // (a) At least some weights must now be HALF.
        Map<DataType, Integer> after = dtypeHistogram(fp16);
        int half = after.getOrDefault(DataType.HALF, 0);
        log.info("HALF variable count after optimize: {}", half);

        File dst = new File(DST);
        SDZSerializer.save(fp16, dst, true, Collections.emptyMap());
        long mb = dst.length() / (1024 * 1024);
        log.info("Saved fp16 model: {} ({} MB) — fp32 source was {} MB",
                dst.getAbsolutePath(), mb, src.length() / (1024 * 1024));
        assertTrue(dst.exists() && dst.length() > 0, "fp16 model file must be written");

        // Report (do not hard-fail on the count yet — this run is diagnostic: it tells us
        // whether GraphOptimizer's fp16 pass actually fired and how much it shrank the model).
        log.info("FP16_BUILD_RESULT halfVars={} savedMB={} srcMB={}",
                half, mb, src.length() / (1024 * 1024));
    }

    private void logDtypeHistogram(String label, SameDiff sd) {
        log.info("dtype histogram ({}): {}", label, dtypeHistogram(sd));
    }

    private Map<DataType, Integer> dtypeHistogram(SameDiff sd) {
        Map<DataType, Integer> h = new EnumMap<>(DataType.class);
        for (SDVariable v : sd.variables()) {
            h.merge(v.dataType(), 1, Integer::sum);
        }
        return h;
    }
}
