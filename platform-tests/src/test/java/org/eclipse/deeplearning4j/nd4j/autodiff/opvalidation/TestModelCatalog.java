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

package org.eclipse.deeplearning4j.nd4j.autodiff.opvalidation;

import lombok.extern.slf4j.Slf4j;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.nd4j.common.tests.tags.NativeTag;
import org.nd4j.common.tests.tags.TagNames;
import org.eclipse.deeplearning4j.omnihub.api.FinetuneRecipe;
import org.eclipse.deeplearning4j.omnihub.catalog.LLMCatalog;

import java.util.List;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Tests for FinetuneRecipe and LLMCatalog model catalog.
 *
 * @author Adam Gibson
 */
@Slf4j
@NativeTag
@Tag(TagNames.SAMEDIFF)
@DisplayName("Model Catalog Tests")
public class TestModelCatalog {

    @Test
    @DisplayName("FinetuneRecipe - Llama 3 Target Modules")
    public void testFinetuneRecipeLlama3() {
        FinetuneRecipe recipe = FinetuneRecipe.llama3();

        assertNotNull(recipe);
        List<String> modules = recipe.getLoraTargetModules();
        assertNotNull(modules);
        assertTrue(modules.contains("q_proj"));
        assertTrue(modules.contains("k_proj"));
        assertTrue(modules.contains("v_proj"));
        assertTrue(modules.contains("o_proj"));
        assertTrue(modules.contains("gate_proj"));
        assertTrue(modules.contains("up_proj"));
        assertTrue(modules.contains("down_proj"));
        assertEquals(7, modules.size());

        assertEquals(16, recipe.getRecommendedRank());
        assertEquals(32, recipe.getRecommendedAlpha());
        assertEquals(2e-4, recipe.getRecommendedLr(), 1e-10);
        assertEquals(8192, recipe.getMaxSeqLength());
    }

    @Test
    @DisplayName("FinetuneRecipe - Gemma Alpha")
    public void testFinetuneRecipeGemma() {
        FinetuneRecipe recipe = FinetuneRecipe.gemma();

        assertNotNull(recipe);
        assertEquals(16, recipe.getRecommendedAlpha());
        assertEquals(16, recipe.getRecommendedRank());
        assertEquals(2048, recipe.getMaxSeqLength());
        assertEquals(7, recipe.getLoraTargetModules().size());
    }

    @Test
    @DisplayName("LLMCatalog - List Models")
    public void testLLMCatalogListModels() {
        List<String> models = LLMCatalog.listModels();

        assertNotNull(models);
        assertFalse(models.isEmpty());
        assertTrue(models.contains("llama-3-8b"), "Should contain llama-3-8b");
        assertTrue(models.contains("mistral-7b-v0.3"), "Should contain mistral-7b-v0.3");
        assertTrue(models.contains("qwen2.5-7b"), "Should contain qwen2.5-7b");
        assertTrue(models.contains("gemma-2-9b"), "Should contain gemma-2-9b");
        log.info("Available models: {}", models);
    }

    @Test
    @DisplayName("LLMCatalog - Get Recipe")
    public void testLLMCatalogGetRecipe() {
        FinetuneRecipe recipe = LLMCatalog.getRecipe("llama-3-8b");

        assertNotNull(recipe, "Recipe for llama-3-8b should not be null");
        assertTrue(recipe.getLoraTargetModules().contains("q_proj"));
        assertEquals(16, recipe.getRecommendedRank());

        // Non-existent model should return null
        FinetuneRecipe missing = LLMCatalog.getRecipe("nonexistent-model");
        assertNull(missing, "Recipe for nonexistent model should be null");
    }

    @Test
    @DisplayName("LLMCatalog - Get HuggingFace ID")
    public void testLLMCatalogGetHuggingFaceId() {
        String hfId = LLMCatalog.getHuggingFaceId("llama-3-8b");
        assertNotNull(hfId);
        assertEquals("meta-llama/Meta-Llama-3-8B", hfId);

        String mistralId = LLMCatalog.getHuggingFaceId("mistral-7b-v0.3");
        assertNotNull(mistralId);
        assertEquals("mistralai/Mistral-7B-v0.3", mistralId);

        // Non-existent model should return null
        String missingId = LLMCatalog.getHuggingFaceId("nonexistent-model");
        assertNull(missingId);
    }
}
