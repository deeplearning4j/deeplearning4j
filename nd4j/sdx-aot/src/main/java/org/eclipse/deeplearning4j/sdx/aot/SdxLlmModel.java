/*
 * Copyright (c) Eclipse Deeplearning4j
 * SPDX-License-Identifier: Apache-2.0
 */
package org.eclipse.deeplearning4j.sdx.aot;

import java.io.IOException;
import java.util.function.BooleanSupplier;
import java.util.function.Consumer;

/** Shared model-handle contract behind both legacy SameDiff and compiled SDX execution. */
interface SdxLlmModel extends AutoCloseable {
    String generateText(String prompt, String optionsJson) throws IOException;

    default String generateStreaming(String prompt, String optionsJson,
                                     Consumer<String> onChunk,
                                     BooleanSupplier shouldCancel) throws IOException {
        if (shouldCancel != null && shouldCancel.getAsBoolean()) {
            return "";
        }
        String text = generateText(prompt, optionsJson);
        if (onChunk != null && !text.isEmpty()) onChunk.accept(text);
        return text;
    }

    String generateChat(String requestJson, String optionsJson) throws IOException;

    String parseChatResult(String requestJson, String rawText) throws IOException;

    String renderChatPrompt(String messagesOrContextJson,
                            boolean addGenerationPrompt) throws IOException;

    int[] tokenize(String text, boolean addSpecialTokens);

    String detokenize(int[] ids, boolean skipSpecialTokens);

    String lastResultJson();

    String infoJson();

    @Override
    void close();
}
