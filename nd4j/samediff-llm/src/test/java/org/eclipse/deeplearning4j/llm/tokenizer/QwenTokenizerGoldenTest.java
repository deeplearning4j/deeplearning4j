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
package org.eclipse.deeplearning4j.llm.tokenizer;

import org.junit.jupiter.api.Test;

import java.io.File;
import java.util.Arrays;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assertions.assertEquals;

class QwenTokenizerGoldenTest {
    private static final String TOKENIZER_DIR_PROPERTY = "kompile.qwen.tokenizer.dir";

    @Test
    void realQwenTokenizerProducesUtf8RoundTrip() {
        String dir = System.getProperty(TOKENIZER_DIR_PROPERTY);
        assertNotNull(dir, "Set -D" + TOKENIZER_DIR_PROPERTY
                + " to the cached HuggingFace Qwen tokenizer directory");
        File tokenizerFile = new File(dir, "tokenizer.json");
        assertTrue(tokenizerFile.isFile(), "Missing real Qwen tokenizer.json: " + tokenizerFile);
        assertTrue(new File(dir, "tokenizer_config.json").isFile(),
                "Missing real Qwen tokenizer_config.json: " + dir);

        try (HuggingFaceTokenizer tokenizer = HuggingFaceTokenizer.fromFile(tokenizerFile)) {
            assertTrue(tokenizer.isValid());
            assertTrue(tokenizer.getVocabSize() > 100_000);

            String input = "Hello é 日本語";
            Encoding encoding = tokenizer.encode(input, false);
            // Golden IDs and token strings come from the standalone HuggingFace
            // tokenizers reference implementation for this exact tokenizer.json.
            assertArrayEquals(new int[] {9419, 3825, 220, 247359}, encoding.getIds());
            assertArrayEquals(new String[] {"Hello", "ĠÃ©", "Ġ", "æĹ¥æľ¬èªŀ"}, encoding.getTokens());
            assertEquals(encoding.getIds().length, encoding.getTokens().length);
            assertFalse(Arrays.stream(encoding.getTokens()).anyMatch(token -> token == null || token.indexOf('�') >= 0));
            for (int i = 0; i < encoding.getIds().length; i++) {
                assertEquals(encoding.getTokens()[i], tokenizer.getToken(encoding.getIds()[i]));
            }

            String decoded = tokenizer.decode(encoding.getIds(), false);
            assertEquals(input, decoded);

            StringBuilder streamed = new StringBuilder();
            try (HuggingFaceTokenizer.DecodeStream decoder =
                         tokenizer.newDecodeStream(false)) {
                for (int id : encoding.getIds()) {
                    streamed.append(decoder.step(Integer.toUnsignedLong(id)));
                }
            }
            assertEquals(decoded, streamed.toString());
        }
    }
}
