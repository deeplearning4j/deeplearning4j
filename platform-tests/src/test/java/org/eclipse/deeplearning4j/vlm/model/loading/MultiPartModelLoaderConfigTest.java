/*
 *  ******************************************************************************
 *  * SPDX-License-Identifier: Apache-2.0
 *  *****************************************************************************/
package org.eclipse.deeplearning4j.vlm.model.loading;

import org.eclipse.deeplearning4j.llm.config.ModelConfig;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.nio.file.Files;
import java.nio.file.Path;

import static org.junit.jupiter.api.Assertions.assertEquals;

public class MultiPartModelLoaderConfigTest {
    @Test
    public void tokenizerModelMaxLengthFillsMissingModelContext(@TempDir Path modelDir)
            throws Exception {
        Files.writeString(modelDir.resolve("generation_config.json"),
                "{\"eos_token_id\":49279}");
        Files.writeString(modelDir.resolve("tokenizer_config.json"),
                "{\"model_max_length\":8192}");

        ModelConfig config = MultiPartModelLoader.loadConfig(modelDir.toFile());

        assertEquals(8192, config.getMaxPositionEmbeddings());
        assertEquals(49279, config.getEosTokenIdSingle());
    }

    @Test
    public void explicitModelContextWinsOverTokenizerMetadata(@TempDir Path modelDir)
            throws Exception {
        Files.writeString(modelDir.resolve("config.json"),
                "{\"max_position_embeddings\":4096}");
        Files.writeString(modelDir.resolve("tokenizer_config.json"),
                "{\"model_max_length\":8192}");

        assertEquals(4096,
                MultiPartModelLoader.loadConfig(modelDir.toFile()).getMaxPositionEmbeddings());
    }
}
