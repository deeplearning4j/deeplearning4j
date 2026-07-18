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

package org.eclipse.deeplearning4j.ggml;

import lombok.extern.slf4j.Slf4j;
import org.eclipse.deeplearning4j.llm.data.LLMModelDownloader;
import org.junit.jupiter.api.Test;
import org.nd4j.ggml.format.GGMLMetadata;
import org.nd4j.ggml.format.GGMLTensorInfo;
import org.nd4j.ggml.format.GGUFReader;

import java.io.File;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Verifies the standardized llama.cpp GGUF contract used by Qwen3.5 native
 * multi-token prediction. This test reads only the GGUF header and tensor
 * directory; it does not allocate model weights or initialize a backend.
 */
@Slf4j
public class TestQwen35MtpImport {

    static final String MODEL_FILE = "Qwen3.5-0.8B-MTP-Q4_K_M.gguf";
    static final String MODEL_URL =
            "https://huggingface.co/unsloth/Qwen3.5-0.8B-MTP-GGUF/resolve/main/"
                    + "Qwen3.5-0.8B-Q4_K_M.gguf";

    @Test
    public void testCachedQwen35ContainsNativeMtpPayload() throws Exception {
        File model = LLMModelDownloader.downloadCustom(MODEL_URL, MODEL_FILE);
        assertTrue(model.isFile(), "MTP-enabled Qwen3.5 model was not downloaded: " + model);

        try (GGUFReader reader = new GGUFReader(model)) {
            GGMLMetadata metadata = reader.getMetadata();
            Map<String, Object> raw = metadata.getRawMetadata();
            String architecture = String.valueOf(raw.get("general.architecture"));
            String nextnKey = architecture + ".nextn_predict_layers";
            Object nextnValue = raw.get(nextnKey);

            List<String> mtpTensorNames = reader.getTensorInfos().stream()
                    .map(GGMLTensorInfo::getName)
                    .filter(name -> name.contains(".nextn."))
                    .sorted()
                    .collect(Collectors.toList());

            log.info("QWEN35_MTP_CONTRACT file={} architecture={} blockCount={} nextnLayers={} "
                            + "totalTensors={} mtpTensors={} names={}",
                    model, architecture, raw.get(architecture + ".block_count"), nextnValue,
                    reader.getTensorInfos().size(), mtpTensorNames.size(), mtpTensorNames);

            assertEquals("qwen35", architecture, "Unexpected GGUF architecture");
            assertTrue(nextnValue instanceof Number,
                    "Missing numeric " + nextnKey + " metadata");
            assertEquals(1, ((Number) nextnValue).intValue(),
                    "Qwen3.5-0.8B should contain one NextN predictor layer");
            assertFalse(mtpTensorNames.isEmpty(), "GGUF contains no blk.*.nextn.* tensors");

            int blockCount = ((Number) raw.get(architecture + ".block_count")).intValue();
            int mtpLayer = blockCount - ((Number) nextnValue).intValue();
            String prefix = "blk." + mtpLayer + ".nextn.";
            assertTrue(mtpTensorNames.contains(prefix + "eh_proj.weight"),
                    "Missing MTP embedding-hidden projection");
            assertTrue(mtpTensorNames.contains(prefix + "enorm.weight"),
                    "Missing MTP embedding norm");
            assertTrue(mtpTensorNames.contains(prefix + "hnorm.weight"),
                    "Missing MTP hidden-state norm");
        }
    }
}
