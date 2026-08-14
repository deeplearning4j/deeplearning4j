/*
 *  ******************************************************************************
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
 *  * SPDX-License-Identifier: Apache-2.0
 *  ******************************************************************************
 */
package org.eclipse.deeplearning4j.llm.generation.constraint;

import org.eclipse.deeplearning4j.llm.generation.ToolSchemaValidator;
import org.eclipse.deeplearning4j.llm.tokenizer.ChatTemplate;
import org.nd4j.shade.jackson.databind.DeserializationFeature;
import org.nd4j.shade.jackson.databind.JsonNode;
import org.nd4j.shade.jackson.databind.ObjectMapper;

import java.util.ArrayList;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * Constrains generation to the XML function protocol declared by a model chat
 * template:
 *
 * <pre>{@code
 * <tool_call>
 * <function=tool_name>
 * <parameter=argument_name>
 * value
 * </parameter>
 * </function>
 * </tool_call>
 * }</pre>
 *
 * <p>The protocol is selected from the imported template, never from a model
 * name. Tool names, ordered required parameter names, value schemas, and the
 * complete envelope are enforced while decoding.</p>
 */
public final class XmlToolCallConstraint implements TextConstraint {

    public static final String TYPE = "xml_tool_call";
    static final String CALL_START = ChatTemplate.XML_TOOL_CALL_START + "\n";
    static final String FUNCTION_CLOSE = "</function>\n";
    static final String CALL_END = ChatTemplate.XML_TOOL_CALL_END;
    static final String PARAMETER_CLOSE = "\n</parameter>\n";

    private static final ObjectMapper MAPPER = new ObjectMapper()
            .enable(DeserializationFeature.FAIL_ON_TRAILING_TOKENS);

    private final List<String> toolNames;
    private final Map<String, List<String>> argumentNamesByTool;
    private final Map<String, Map<String, Object>> parameterSchemasByTool;

    public XmlToolCallConstraint(
            List<String> toolNames,
            Map<String, List<String>> argumentNamesByTool,
            Map<String, Map<String, Object>> parameterSchemasByTool) {
        if (toolNames == null || toolNames.isEmpty()) {
            throw new IllegalArgumentException(
                    "XmlToolCallConstraint requires at least one tool name");
        }
        List<String> copiedNames = new ArrayList<>(toolNames.size());
        for (String name : toolNames) {
            if (name == null || name.isBlank()) {
                throw new IllegalArgumentException("XML tool names must not be blank");
            }
            copiedNames.add(name);
        }
        this.toolNames = Collections.unmodifiableList(copiedNames);

        Map<String, List<String>> copiedArguments = new LinkedHashMap<>();
        if (argumentNamesByTool != null) {
            argumentNamesByTool.forEach((name, arguments) -> {
                if (this.toolNames.contains(name) && arguments != null) {
                    copiedArguments.put(name, List.copyOf(arguments));
                }
            });
        }
        this.argumentNamesByTool = Collections.unmodifiableMap(copiedArguments);

        Map<String, Map<String, Object>> copiedSchemas = new LinkedHashMap<>();
        if (parameterSchemasByTool != null) {
            parameterSchemasByTool.forEach((name, schema) -> {
                if (this.toolNames.contains(name) && schema != null) {
                    copiedSchemas.put(name,
                            Collections.unmodifiableMap(new LinkedHashMap<>(schema)));
                }
            });
        }
        this.parameterSchemasByTool = Collections.unmodifiableMap(copiedSchemas);
    }

    @Override
    public boolean canExtend(String currentText, String piece) {
        String current = currentText == null ? "" : currentText;
        String extension = piece == null ? "" : piece;
        return !extension.isEmpty() && !isAccepting(current)
                && validPrefix(current + extension, false);
    }

    @Override
    public boolean allowsSpecialToken(String currentText, String piece) {
        String candidate = piece == null ? "" : piece;
        boolean protocolMarker = ChatTemplate.XML_TOOL_CALL_START.equals(candidate)
                || ChatTemplate.XML_TOOL_CALL_END.equals(candidate);
        return protocolMarker
                && validPrefix((currentText == null ? "" : currentText) + candidate, false);
    }

    @Override
    public boolean isAccepting(String currentText) {
        return currentText != null && validPrefix(currentText, true);
    }

    private boolean validPrefix(String text, boolean requireComplete) {
        for (String toolName : toolNames) {
            if (validToolPrefix(text, toolName, requireComplete)) {
                return true;
            }
        }
        return false;
    }

    private boolean validToolPrefix(String text, String toolName, boolean requireComplete) {
        String prefix = CALL_START + ChatTemplate.XML_FUNCTION_START + toolName + ">\n";
        if (prefix.startsWith(text)) {
            return !requireComplete;
        }
        if (!text.startsWith(prefix)) {
            return false;
        }

        int cursor = prefix.length();
        Map<String, Object> arguments = new LinkedHashMap<>();
        List<String> argumentNames =
                argumentNamesByTool.getOrDefault(toolName, List.of());
        Map<String, Map<String, Object>> argumentSchemas =
                argumentSchemas(parameterSchemasByTool.get(toolName));
        for (String argumentName : argumentNames) {
            String opening = ChatTemplate.XML_PARAMETER_START + argumentName + ">\n";
            String remainder = text.substring(cursor);
            if (opening.startsWith(remainder)) {
                return !requireComplete;
            }
            if (!remainder.startsWith(opening)) {
                return false;
            }
            cursor += opening.length();
            int close = text.indexOf(PARAMETER_CLOSE, cursor);
            if (close < 0) {
                return !requireComplete
                        && validOpenParameterValue(
                                text.substring(cursor),
                                argumentSchemas.getOrDefault(argumentName, Map.of()));
            }
            Object value = parseValue(text.substring(cursor, close).trim());
            if (!ToolSchemaValidator.isValidValue(
                    value, argumentSchemas.getOrDefault(argumentName, Map.of()))) {
                return false;
            }
            arguments.put(argumentName, value);
            cursor = close + PARAMETER_CLOSE.length();
        }

        String suffix = FUNCTION_CLOSE + CALL_END;
        String remainder = text.substring(cursor);
        if (suffix.startsWith(remainder)) {
            if (remainder.length() < suffix.length()) {
                return !requireComplete;
            }
            ChatTemplate.Tool declaration = new ChatTemplate.Tool(
                    toolName, "", parameterSchemasByTool.getOrDefault(toolName, Map.of()));
            return ToolSchemaValidator.validateArguments(declaration, arguments).isEmpty();
        }
        return false;
    }

    private static boolean validOpenParameterValue(
            String valueAndPossibleClose, Map<String, Object> schema) {
        int structuralStart = valueAndPossibleClose.length();
        boolean closingParameter = false;
        int maximumSuffix = Math.min(
                valueAndPossibleClose.length(), PARAMETER_CLOSE.length() - 1);
        for (int length = maximumSuffix; length > 0; length--) {
            String suffix = valueAndPossibleClose.substring(
                    valueAndPossibleClose.length() - length);
            if (PARAMETER_CLOSE.startsWith(suffix)) {
                structuralStart = valueAndPossibleClose.length() - length;
                closingParameter = true;
                break;
            }
        }
        String value = valueAndPossibleClose.substring(0, structuralStart);
        if (value.indexOf('<') >= 0) {
            return false;
        }
        return closingParameter
                ? NativeToolCallConstraint.validCompleteValue(value, schema)
                : NativeToolCallConstraint.validValuePrefix(value, schema);
    }

    @SuppressWarnings("unchecked")
    private static Map<String, Map<String, Object>> argumentSchemas(
            Map<String, Object> parameterSchema) {
        if (parameterSchema == null
                || !(parameterSchema.get("properties") instanceof Map<?, ?>)) {
            return Map.of();
        }
        Map<String, Map<String, Object>> result = new LinkedHashMap<>();
        ((Map<?, ?>) parameterSchema.get("properties")).forEach((name, schema) -> {
            if (name != null && schema instanceof Map<?, ?>) {
                Map<String, Object> copied = new LinkedHashMap<>();
                ((Map<?, ?>) schema).forEach(
                        (key, value) -> copied.put(String.valueOf(key), value));
                result.put(String.valueOf(name), Collections.unmodifiableMap(copied));
            }
        });
        return Collections.unmodifiableMap(result);
    }

    private static Object parseValue(String value) {
        try {
            JsonNode json = MAPPER.readTree(value);
            if (json != null) {
                return MAPPER.convertValue(json, Object.class);
            }
        } catch (Exception ignored) {
            // XML string parameters are intentionally unquoted in this protocol.
        }
        return value;
    }

    @Override
    public TextConstraint reset() {
        return this;
    }

    @Override
    public String type() {
        return TYPE;
    }
}
