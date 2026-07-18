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

    @Test
    public void encodesDecodesAndOwnsNativeResources() {
        NativeTokenizer tokenizer = NativeTokenizer.fromJson(TOKENIZER_JSON);
        assertTrue(tokenizer.isValid());
        assertEquals(3L, tokenizer.vocabSize());
        assertArrayEquals(new int[] {1, 2},
                tokenizer.encode("hello world", false));
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
    public void rejectsInvalidTokenizerJsonWithNativeError() {
        IllegalStateException error = assertThrows(
                IllegalStateException.class,
                () -> NativeTokenizer.fromJson("{not-json}"));
        assertTrue(error.getMessage().contains("load tokenizer JSON failed"));
    }
}
