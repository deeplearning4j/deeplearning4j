/*
 * SPDX-License-Identifier: Apache-2.0
 */

package org.nd4j.dsp.runtime.litertlm;

import org.junit.jupiter.api.Test;

import java.nio.file.Paths;
import java.util.Arrays;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;

class SdxLiteRtLmChatSessionTest {

    @Test
    void formatsConversationMessagesWithStrictJsonEscaping() {
        assertEquals(
                "{\"role\":\"user\",\"content\":[{\"type\":\"text\","
                        + "\"text\":\"say \\\"hi\\\"\\n\\t\\u0001\"}]}",
                SdxLiteRtLmChatSession.messageJson(
                        "user", "say \"hi\"\n\t\u0001"));
    }

    @Test
    void normalizesOfficialSocSpellings() {
        assertEquals("tensorg5",
                SdxLiteRtLmChatSession.normalizedSoc("Tensor_G5"));
        assertEquals("tensorg5",
                SdxLiteRtLmChatSession.normalizedSoc("Tensor G5"));
        assertEquals("tensorg5",
                SdxLiteRtLmChatSession.normalizedSoc("tensor-g5"));
    }

    @Test
    void exposesNoConfigurableCpuOrGpuBackend() {
        assertEquals("npu", SdxLiteRtLmChatSession.BACKEND);
        assertFalse(Arrays.stream(SdxLiteRtLmChatSession.Builder.class
                        .getDeclaredMethods())
                .anyMatch(method -> method.getName().equals("backend")));
    }

    @Test
    void validatesSamplingBeforeNativeLoading() {
        SdxLiteRtLmChatSession.Builder builder =
                SdxLiteRtLmChatSession.builder(
                        Paths.get("model.litertlm"),
                        Paths.get("dispatch"));

        assertThrows(IllegalArgumentException.class,
                () -> builder.sampler(0, 0.9f, 0.8f, 1));
        assertThrows(IllegalArgumentException.class,
                () -> builder.sampler(40, 0.0f, 0.8f, 1));
        assertThrows(IllegalArgumentException.class,
                () -> builder.sampler(40, 0.9f, -0.1f, 1));
        assertThrows(IllegalArgumentException.class,
                () -> builder.contextTokens(0));
        assertThrows(IllegalArgumentException.class,
                () -> builder.maxOutputTokens(0));
    }
}
