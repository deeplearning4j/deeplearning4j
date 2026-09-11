/*
 * SPDX-License-Identifier: Apache-2.0
 * This program is made available under the Apache License, Version 2.0.
 * See https://www.apache.org/licenses/LICENSE-2.0 and the NOTICE file.
 */
package org.eclipse.deeplearning4j.llm.generation.constraint;

import org.eclipse.deeplearning4j.llm.tokenizer.ChatTemplate;

import java.util.ArrayList;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/** Model-owned Gemma call envelopes with request-owned tool names and argument schemas. */
public final class GemmaToolCallConstraint implements TextConstraint {
    public static final String TYPE = "gemma_tool_call";
    private final Map<String, ChatTemplate.Tool> tools;

    public GemmaToolCallConstraint(List<String> toolNames,
                                   Map<String, List<String>> requiredArguments,
                                   Map<String, Map<String, Object>> parameterSchemas) {
        if (toolNames == null || toolNames.isEmpty()) {
            throw new IllegalArgumentException("GemmaToolCallConstraint requires declared tools");
        }
        Map<String, ChatTemplate.Tool> declarations = new LinkedHashMap<>();
        for (String name : toolNames) {
            if (!GemmaToolCallCodec.validName(name)) {
                throw new IllegalArgumentException("Unsupported Gemma tool name: " + name);
            }
            Map<String, Object> schema = new LinkedHashMap<>();
            if (parameterSchemas != null && parameterSchemas.get(name) != null) {
                schema.putAll(parameterSchemas.get(name));
            }
            schema.putIfAbsent("type", "object");
            // The complete request schema is authoritative. Required-name-only configs
            // still require those keys, but do not invent value types or enum values.
            if (!schema.containsKey("required") && requiredArguments != null
                    && requiredArguments.get(name) != null) {
                schema.put("required", new ArrayList<>(requiredArguments.get(name)));
            }
            declarations.put(name, new ChatTemplate.Tool(name, "", schema));
        }
        tools = Collections.unmodifiableMap(declarations);
    }

    @Override
    public boolean canExtend(String currentText, String piece) {
        if (piece == null || piece.isEmpty()) return false;
        String current = currentText == null ? "" : currentText;
        if (current.length() > GemmaToolCallCodec.MAX_CHARS - piece.length()) return false;
        // One structural whitespace token is enough; repeated whitespace must make
        // progress unless it is literal string data. Use the codec's framing state,
        // not JSON quote/escape rules (Gemma backslashes are literal characters).
        if (!current.isEmpty()
                && JsonObjectConstraint.isJsonWhitespace(current.charAt(current.length() - 1))
                && JsonObjectConstraint.isOnlyJsonWhitespace(piece)
                && !GemmaToolCallCodec.scan(current, tools).insideString) {
            return false;
        }
        return GemmaToolCallCodec.scan(current + piece, tools).valid;
    }

    @Override
    public boolean allowsSpecialToken(String currentText, String piece) {
        return (ChatTemplate.GEMMA_TOOL_CALL_START.equals(piece)
                || ChatTemplate.GEMMA_TOOL_CALL_END.equals(piece)
                || ChatTemplate.GEMMA_STRING.equals(piece)) && canExtend(currentText, piece);
    }

    @Override
    public boolean isAccepting(String currentText) {
        return currentText != null && GemmaToolCallCodec.scan(currentText, tools).complete;
    }

    @Override
    public TextConstraint reset() { return this; }

    @Override
    public String type() { return TYPE; }
}
