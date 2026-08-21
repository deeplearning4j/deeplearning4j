/*
 *  ******************************************************************************
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
 *  * See the NOTICE file distributed with this work for additional
 *  * information regarding copyright ownership.
 *  ******************************************************************************
 */

package org.eclipse.deeplearning4j.tokenizers;

import org.junit.jupiter.api.Test;

import java.util.List;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

public class NativeTokenizerTest {

    private static final String TOKENIZER_JSON =
            "{"
                    + "\"version\":\"1.0\","
                    + "\"truncation\":null,"
                    + "\"padding\":null,"
                    + "\"added_tokens\":[],"
                    + "\"normalizer\":null,"
                    + "\"pre_tokenizer\":{\"type\":\"Whitespace\"},"
                    + "\"post_processor\":null,"
                    + "\"decoder\":null,"
                    + "\"model\":{"
                    + "\"type\":\"WordLevel\","
                    + "\"vocab\":{\"[UNK]\":0,\"hello\":1,\"world\":2},"
                    + "\"unk_token\":\"[UNK]\""
                    + "}"
                    + "}";

    private static final String BYTE_LEVEL_TOKENIZER_JSON =
            "{"
                    + "\"version\":\"1.0\","
                    + "\"truncation\":null,"
                    + "\"padding\":null,"
                    + "\"added_tokens\":[],"
                    + "\"normalizer\":null,"
                    + "\"pre_tokenizer\":{\"type\":\"ByteLevel\","
                    + "\"add_prefix_space\":false,\"trim_offsets\":false,\"use_regex\":false},"
                    + "\"post_processor\":null,"
                    + "\"decoder\":{\"type\":\"ByteLevel\","
                    + "\"add_prefix_space\":false,\"trim_offsets\":false,\"use_regex\":false},"
                    + "\"model\":{"
                    + "\"type\":\"BPE\","
                    + "\"dropout\":null,\"unk_token\":null,\"continuing_subword_prefix\":null,"
                    + "\"end_of_word_suffix\":null,\"fuse_unk\":false,"
                    + "\"vocab\":{\"ĠHello\":0,\"Ã©\":1},\"merges\":[]"
                    + "}"
                    + "}";

    @Test
    public void encodesDecodesAndOwnsNativeResources() {
        NativeTokenizer tokenizer = NativeTokenizer.fromJson(TOKENIZER_JSON);
        assertTrue(tokenizer.isValid());
        assertEquals(3L, tokenizer.vocabSize());
        assertEquals(0, tokenizer.tokenToId("[UNK]"));
        assertEquals(1, tokenizer.tokenToId("hello"));
        assertEquals(-1, tokenizer.tokenToId("missing"));
        assertArrayEquals(new int[] {1, 2},
                tokenizer.encode("hello world", false));
        NativeTokenizer.EncodedText detailed =
                tokenizer.encodeWithTokens("hello world", false);
        assertArrayEquals(new int[] {1, 2}, detailed.ids());
        assertArrayEquals(new String[] {"hello", "world"}, detailed.tokens());
        assertArrayEquals(new long[] {1L, 2L},
                tokenizer.encodeLong("hello world", false));
        assertEquals("hello world",
                tokenizer.decode(new int[] {1, 2}, false));
        assertEquals("hello world",
                tokenizer.decode(new long[] {1L, 2L}, false));
        assertFalse(tokenizer.version().isEmpty());

        tokenizer.close();
        tokenizer.close();
        assertFalse(tokenizer.isValid());
        assertThrows(IllegalStateException.class,
                () -> tokenizer.encode("hello", false));
    }

    @Test
    public void rejectsOutOfRangeSdxTokenIds() {
        try (NativeTokenizer tokenizer = NativeTokenizer.fromJson(TOKENIZER_JSON)) {
            assertThrows(IllegalArgumentException.class,
                    () -> tokenizer.decode(new long[] {-1L}, false));
            assertThrows(IllegalArgumentException.class,
                    () -> tokenizer.decode(new long[] {0x1_0000_0000L}, false));
        }
    }

    @Test
    public void incrementallyDecodesWithNativeStream() {
        NativeTokenizer tokenizer = NativeTokenizer.fromJson(TOKENIZER_JSON);
        NativeTokenizer.DecodeStream stream =
                tokenizer.newDecodeStream(false);
        assertEquals("hello", stream.step(1L));

        tokenizer.close();
        assertEquals(" world", stream.step(2L));
        assertThrows(IllegalArgumentException.class,
                () -> stream.step(0x1_0000_0000L));

        stream.close();
        stream.close();
        assertThrows(IllegalStateException.class, () -> stream.step(1L));
    }

    @Test
    public void decodesByteLevelTokensToUnicodeThroughNativeFfi() {
        try (NativeTokenizer tokenizer = NativeTokenizer.fromJson(BYTE_LEVEL_TOKENIZER_JSON)) {
            assertEquals(" Helloé",
                    tokenizer.decode(new int[] {0, 1}, true));
            NativeTokenizer.DecodeStream stream = tokenizer.newDecodeStream(true);
            assertEquals(" Helloé", stream.step(0L) + stream.step(1L));
            stream.close();
        }
    }

    @Test
    public void rejectsInvalidTokenizerJsonWithNativeError() {
        IllegalStateException error = assertThrows(
                IllegalStateException.class,
                () -> NativeTokenizer.fromJson("{not-json}"));
        assertTrue(error.getMessage().contains("load tokenizer JSON failed"));
    }

    @Test
    public void rendersModelOwnedHuggingFaceChatTemplate() {
        String config = "{"
                + "\"bos_token\":{\"content\":\"<s>\",\"special\":true},"
                + "\"chat_template\":\"{{ bos_token }}{% for message in messages %}"
                + "{{ message['role'] }}:{{ message['content'] }}\\n{% endfor %}"
                + "{% if add_generation_prompt %}assistant:{% endif %}\""
                + "}";
        try (NativeTokenizer tokenizer = NativeTokenizer.fromJson(TOKENIZER_JSON)) {
            String rendered = tokenizer.applyChatTemplate(
                    config,
                    List.of(
                            NativeTokenizer.ChatMessage.system("Be concise."),
                            NativeTokenizer.ChatMessage.user("quoted \"value\" — café 日本語")),
                    true);
            assertEquals(
                    "<s>system:Be concise.\nuser:quoted \"value\" — café 日本語\nassistant:",
                    rendered);
        }
    }

    @Test
    public void rendersAndroidSmokeChatContextWithoutLosingMessages() {
        String config = "{"
                + "\"bos_token\":\"<s>\","
                + "\"chat_template\":\"{{ bos_token }}{% for message in messages %}"
                + "{{ message['role'] }}:{{ message['content'] }}\\n{% endfor %}"
                + "tools={{ tools | length }};choice={{ tool_choice }};"
                + "{% if add_generation_prompt %}assistant:{% endif %}\""
                + "}";
        String androidContext = "{"
                + "\"messages\":[{\"role\":\"user\",\"content\":\"Reply café 日本語.\"}],"
                + "\"tools\":[],"
                + "\"tool_choice\":\"none\","
                + "\"add_generation_prompt\":true"
                + "}";

        assertEquals(
                "<s>user:Reply café 日本語.\ntools=0;choice=none;assistant:",
                NativeTokenizer.renderChatTemplateContext(config, androidContext));
    }

    @Test
    public void rendersChatTemplateWithoutAllocatingTokenizerHandle() {
        String config = "{"
                + "\"bos_token\":\"<s>\","
                + "\"chat_template\":\"{{ bos_token }}{% for message in messages %}"
                + "{% if message['role'] == 'system' %}<SYS>{% endif %}"
                + "{{ message['content'] }}{% if message['role'] == 'system' %}</SYS>{% endif %}"
                + "{% endfor %}{% if add_generation_prompt %}<A>{% endif %}\""
                + "}";

        assertEquals(
                "<s><SYS>rules</SYS>hello<A>",
                NativeTokenizer.renderChatTemplate(
                        config,
                        List.of(
                                NativeTokenizer.ChatMessage.system("rules"),
                                NativeTokenizer.ChatMessage.user("hello")),
                        true));
    }

    @Test
    public void rejectsIncompleteTokenizerChatConfiguration() {
        try (NativeTokenizer tokenizer = NativeTokenizer.fromJson(TOKENIZER_JSON)) {
            IllegalStateException error = assertThrows(
                    IllegalStateException.class,
                    () -> tokenizer.applyChatTemplate(
                            "{}",
                            List.of(NativeTokenizer.ChatMessage.user("hello")),
                            true));
            assertTrue(error.getMessage().contains("chat_template"));
        }
    }
}
