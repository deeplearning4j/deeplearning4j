/*
 *  ******************************************************************************
 *  * SPDX-License-Identifier: Apache-2.0
 *  *****************************************************************************/
package org.eclipse.deeplearning4j.vlm.model;

import org.eclipse.deeplearning4j.llm.config.ModelConfig;
import org.junit.jupiter.api.Test;

import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

public class VisionLanguageModelContextBudgetTest {
    @Test
    public void autoBudgetUsesActualPromptAndDeclaredContext() {
        ModelConfig config = new ModelConfig();
        config.setMaxPositionEmbeddings(8192);
        VisionLanguageModel model = VisionLanguageModel.builder()
                .config(config)
                .maxKvLen(8192)
                .build();

        assertEquals(7043, model.resolveGenerationBudget(1149, 0));
        assertEquals(512, model.resolveGenerationBudget(1149, 512));
        assertEquals(43, model.resolveGenerationBudget(8149, 512));
        assertThrows(IllegalStateException.class,
                () -> model.resolveGenerationBudget(8192, 0));
    }

    @Test
    public void preservesEveryDeclaredModelEosToken() {
        ModelConfig config = new ModelConfig();
        config.setEosTokenId(List.of(49279, 49280, 49281));

        VisionLanguageModel model = VisionLanguageModel.builder().config(config).build();

        assertEquals(List.of(49279, 49280, 49281), model.modelEosTokenIds());
    }

    @Test
    public void directImagePromptCountsImageAndTextTokensAgainstContext() {
        int[] promptIds = VisionLanguageModel.combinedPromptIds(
                3, 49190, new int[]{10, 11});

        assertArrayEquals(new int[]{49190, 49190, 49190, 10, 11}, promptIds);

        ModelConfig config = new ModelConfig();
        config.setMaxPositionEmbeddings(8);
        VisionLanguageModel model = VisionLanguageModel.builder().config(config).build();
        assertEquals(3, model.resolveGenerationBudget(promptIds.length, 0));
    }

    @Test
    public void legacyPackageWithoutContextUsesCompatibilityBudget() {
        VisionLanguageModel model = VisionLanguageModel.builder().config(new ModelConfig()).build();
        assertEquals(512, model.resolveGenerationBudget(20, 0));
        assertEquals(73, model.resolveGenerationBudget(20, 73));
    }
}
