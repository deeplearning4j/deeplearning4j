/* SPDX-License-Identifier: Apache-2.0 */
package org.eclipse.deeplearning4j.llm.generation;

import org.eclipse.deeplearning4j.llm.tokenizer.Encoding;
import org.eclipse.deeplearning4j.llm.tokenizer.Tokenizer;
import org.junit.jupiter.api.Test;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.lang.reflect.Field;
import java.lang.reflect.Constructor;
import java.lang.reflect.Proxy;
import java.util.Collections;
import org.nd4j.autodiff.samediff.SameDiff;

import static org.junit.jupiter.api.Assertions.*;

/** Pure admission tests: no model, device allocation, or native decode is needed. */
class TestGenerationPipelinePrefillCapacity {
    private static GenerationPipelineConfig fixedConfig() {
        return GenerationPipelineConfig.builder().maxPrefillLength(512).chatTemplate("template").build();
    }

    @Test
    void acceptsWithinAndEqualBoundary() {
        GenerationPipelineConfig config = fixedConfig();
        assertDoesNotThrow(() -> GenerationPipeline.requirePrefillCapacity(config, 1));
        assertDoesNotThrow(() -> GenerationPipeline.requirePrefillCapacity(config, 511));
        assertDoesNotThrow(() -> GenerationPipeline.requirePrefillCapacity(config, 512));
        assertEquals(512, config.getMaxPrefillLength());
    }

    @Test
    void rejectsOversizedInsteadOfTruncatingOrGrowingCapacity() {
        GenerationPipelineConfig config = fixedConfig();
        for (int length : new int[]{513, 1264}) {
            IllegalArgumentException error = assertThrows(IllegalArgumentException.class,
                    () -> GenerationPipeline.requirePrefillCapacity(config, length));
            assertTrue(error.getMessage().contains(Integer.toString(length)));
            assertTrue(error.getMessage().contains("maxPrefillLength=512"));
        }
        assertEquals(512, config.getMaxPrefillLength());
    }

    @Test
    void dynamicPrefillHasNoFixedBufferLimit() {
        GenerationPipelineConfig config = GenerationPipelineConfig.builder().maxPrefillLength(0).build();
        assertDoesNotThrow(() -> GenerationPipeline.requirePrefillCapacity(config, 1264));
    }

    @Test
    void textAndSessionRejectBeforeDeviceOrStateAccess() throws Exception {
        GenerationPipeline pipeline = admissionOnlyPipeline();
        int[] ids = new int[1264];
        ids[0] = 17;
        Tokenizer tokenizer = tokenizer(ids);
        setField(pipeline, "tokenizer", tokenizer);
        // Decoder deliberately absent: admission must happen before device access.
        assertThrows(IllegalArgumentException.class, () -> pipeline.generate("prompt", 1));
        assertThrows(IllegalArgumentException.class, () -> pipeline.startSession("prompt", 1));
        assertThrows(IllegalArgumentException.class, () -> pipeline.generateStream("prompt", 1, s -> fail()));
        assertEquals(1264, ids.length);
        assertEquals(17, ids[0]);
    }

    @Test
    void embeddingPromptRejectsBeforeReadingOrSlicingEmbeddings() throws Exception {
        GenerationPipeline pipeline = admissionOnlyPipeline();
        INDArray embeddings = (INDArray) Proxy.newProxyInstance(INDArray.class.getClassLoader(),
                new Class<?>[]{INDArray.class}, (proxy, method, args) -> {
                    throw new AssertionError("Unexpected embedding access: " + method.getName());
                });
        assertThrows(IllegalArgumentException.class, () -> pipeline.generate(embeddings, new int[1264], 1));
    }

    @Test
    void oversizedEmbeddingSequenceAlsoRejectsBeforeNativeWork() throws Exception {
        GenerationPipeline pipeline = admissionOnlyPipeline();
        INDArray embeddings = (INDArray) Proxy.newProxyInstance(INDArray.class.getClassLoader(),
                new Class<?>[]{INDArray.class}, (proxy, method, args) -> {
                    assertEquals("size", method.getName());
                    assertEquals(1, args[0]);
                    return 1264L;
                });
        assertThrows(IllegalArgumentException.class, () -> pipeline.generate(embeddings, new int[512], 1));
    }

    private static GenerationPipeline admissionOnlyPipeline() throws Exception {
        Constructor<?> constructor = GenerationPipeline.class.getDeclaredConstructors()[0];
        constructor.setAccessible(true);
        GenerationPipeline pipeline = (GenerationPipeline) constructor.newInstance(
                SameDiff.create(), false, null, false, tokenizer(new int[]{1}), null,
                ModelIOConfig.builder().build(), null, 0L, null, null, null, false, fixedConfig(), null);
        setField(pipeline, "decoder", null);
        return pipeline;
    }

    private static Tokenizer tokenizer(int[] ids) {
        return (Tokenizer) Proxy.newProxyInstance(Tokenizer.class.getClassLoader(),
                new Class<?>[]{Tokenizer.class}, (proxy, method, args) -> {
                    switch (method.getName()) {
                        case "getSpecialTokenIds": return Collections.emptySet();
                        case "getAddedTokens": return Collections.emptyMap();
                        case "encodePrompt": return Encoding.builder().ids(ids).build();
                        default: throw new AssertionError("Unexpected tokenizer access: " + method.getName());
                    }
                });
    }

    private static void setField(GenerationPipeline pipeline, String name, Object value) throws Exception {
        Field field = GenerationPipeline.class.getDeclaredField(name);
        field.setAccessible(true);
        field.set(pipeline, value);
    }
}
