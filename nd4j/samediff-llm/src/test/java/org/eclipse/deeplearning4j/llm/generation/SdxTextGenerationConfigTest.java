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
package org.eclipse.deeplearning4j.llm.generation;

import org.nd4j.shade.jackson.databind.JsonNode;
import org.nd4j.shade.jackson.databind.node.ObjectNode;
import org.junit.jupiter.api.Test;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ops.impl.transforms.custom.CausalConv1d;
import org.nd4j.linalg.factory.Nd4j;

import java.io.IOException;
import java.util.Arrays;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

class SdxTextGenerationConfigTest {

    @Test
    void hybridGraphEmitsV2RecurrentStateContract() throws Exception {
        SameDiff graph = SameDiff.create();
        graph.placeHolder("input_ids", DataType.INT64, 1, 8);
        graph.placeHolder("_causal_mask", DataType.FLOAT, 1, 1, 8, 16);
        graph.placeHolder("position_offset", DataType.INT64, 1);
        graph.placeHolder("cache_position", DataType.INT64, 1);
        SDVariable actualLength =
                graph.placeHolder("actual_sequence_length", DataType.INT64, 1);

        SDVariable key =
                graph.placeHolder("past_key_values.0.key", DataType.FLOAT, 1, 16, 2, 4);
        SDVariable value =
                graph.placeHolder("past_key_values.0.value", DataType.FLOAT, 1, 16, 2, 4);
        graph.identity("k_rope_0", key);
        graph.identity("v_heads_0", value);

        SDVariable convInput = graph.placeHolder("conv_input", DataType.FLOAT, 1, 8, 4);
        SDVariable convState =
                graph.placeHolder("past_conv_state.1", DataType.FLOAT, 1, 4, 2);
        SDVariable convWeight =
                graph.constant("conv_weight", Nd4j.ones(DataType.FLOAT, 4, 3));
        SDVariable[] conv = new CausalConv1d(
                graph, convInput, convWeight, null, convState, actualLength, 0, 0)
                .outputVariables();
        graph.updateVariableNameAndReference(conv[1], "conv_state_out_1");

        graph.identity("lm_logits", conv[0]);
        graph.identity("lm_logits_last", conv[0]);
        graph.setOutputs(Arrays.asList(
                "k_rope_0",
                "v_heads_0",
                "conv_state_out_1",
                "lm_logits",
                "lm_logits_last"));

        SdxTextGenerationConfig.Options options =
                SdxTextGenerationConfig.Options.builder()
                        .contextLength(16)
                        .maxPrefillLength(8)
                        .bosId(1)
                        .padId(0)
                        .eosIds(Arrays.asList(2, 3))
                        .maxNewTokens(4)
                        .build();

        ObjectNode config = SdxTextGenerationConfig.derive(graph, options);

        assertEquals(2, config.path("formatVersion").asInt());
        assertEquals("causal-lm-in-graph-state-v2", config.path("profile").asText());
        assertEquals("lm_logits_last", config.path("io").path("logits").asText());
        assertEquals("lm_logits", config.path("io").path("prefillLogits").asText());
        assertEquals("k_rope_0",
                config.path("io").path("prefillKeyOutputs").get(0).asText());
        assertEquals("v_heads_0",
                config.path("io").path("prefillValueOutputs").get(0).asText());

        JsonNode states = config.path("io").path("recurrentStates");
        assertEquals(1, states.size());
        assertEquals("past_conv_state.1", states.get(0).path("input").asText());
        assertEquals("conv_state_out_1", states.get(0).path("output").asText());
        assertEquals("CONV", states.get(0).path("kind").asText());
        assertEquals("FLOAT32", states.get(0).path("dataType").asText());
        assertEquals("FLOAT32", config.path("execution").path("kvDtype").asText());
        assertEquals(Arrays.asList(1L, 4L, 2L),
                Arrays.asList(
                        states.get(0).path("shape").get(0).asLong(),
                        states.get(0).path("shape").get(1).asLong(),
                        states.get(0).path("shape").get(2).asLong()));
        assertTrue(config.path("execution").path("planOwnsKvScatter").asBoolean());
    }

    @Test
    void rejectsPrefillKvOutputDtypeThatDiffersFromDecodeCache() {
        SameDiff graph = SameDiff.create();
        graph.placeHolder("input_ids", DataType.INT64, 1, 8);
        graph.placeHolder("_causal_mask", DataType.FLOAT, 1, 1, 8, 16);
        graph.placeHolder("position_offset", DataType.INT64, 1);
        graph.placeHolder("cache_position", DataType.INT64, 1);
        graph.placeHolder("actual_sequence_length", DataType.INT64, 1);

        SDVariable key =
                graph.placeHolder("past_key_values.0.key", DataType.HALF, 1, 16, 2, 4);
        SDVariable value =
                graph.placeHolder("past_key_values.0.value", DataType.HALF, 1, 16, 2, 4);
        key.castTo("k_rope_0", DataType.FLOAT);
        graph.identity("v_heads_0", value);
        graph.identity("lm_logits", key);
        graph.setOutputs(Arrays.asList("k_rope_0", "v_heads_0", "lm_logits"));

        SdxTextGenerationConfig.Options options =
                SdxTextGenerationConfig.Options.builder()
                        .contextLength(16)
                        .maxPrefillLength(8)
                        .padId(0)
                        .eosIds(Arrays.asList(2))
                        .maxNewTokens(4)
                        .build();

        IOException failure = assertThrows(
                IOException.class, () -> SdxTextGenerationConfig.derive(graph, options));

        assertTrue(failure.getMessage().contains("KV dtype contract mismatch"));
        assertTrue(failure.getMessage().contains("k_rope_0"));
        assertTrue(failure.getMessage().contains("FLOAT32"));
    }
}
