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

package org.eclipse.deeplearning4j.llm.generation.constraint;

import org.eclipse.deeplearning4j.llm.generation.ToolSchemaValidator;
import org.eclipse.deeplearning4j.llm.tokenizer.ChatTemplate;
import org.nd4j.shade.jackson.databind.DeserializationFeature;
import org.nd4j.shade.jackson.databind.JsonNode;
import org.nd4j.shade.jackson.databind.ObjectMapper;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.regex.Pattern;
import java.util.stream.Collectors;

/**
 * Constrains generation to one declared native function call:
 *
 * <pre>{@code
 * <|tool_call_start|>[tool_name(key=value, ...)]<|tool_call_end|>
 * }</pre>
 *
 * <p>The constraint owns the complete native envelope, including the model-declared end
 * sentinel. Function names and the opening envelope are constrained exactly. Argument values
 * remain model-selected, but quotes and nested collection delimiters must be balanced and the
 * final argument list must use named {@code key=value} entries.</p>
 */
public final class NativeToolCallConstraint implements TextConstraint {

    public static final String TYPE = "native_tool_call";
    static final String CALL_START = ChatTemplate.NATIVE_TOOL_CALL_START + "[";
    static final String CALL_CLOSE = ")]";
    static final String CALL_END = ChatTemplate.NATIVE_TOOL_CALL_END;

    private static final Pattern ARGUMENT_NAME =
            Pattern.compile("[A-Za-z_][A-Za-z0-9_]*");
    private static final ObjectMapper MAPPER = new ObjectMapper()
            .enable(DeserializationFeature.FAIL_ON_TRAILING_TOKENS);

    private final List<String> toolNames;
    private final List<String> callPrefixes;
    private final Map<String, List<String>> argumentNamesByTool;
    private final Map<String, Map<String, List<String>>> argumentValuesByTool;
    private final Map<String, Map<String, Object>> parameterSchemasByTool;

    public NativeToolCallConstraint(List<String> toolNames) {
        this(toolNames, Collections.emptyMap(), Collections.emptyMap(),
                Collections.emptyMap());
    }

    public NativeToolCallConstraint(
            List<String> toolNames,
            Map<String, List<String>> argumentNamesByTool) {
        this(toolNames, argumentNamesByTool, Collections.emptyMap(),
                Collections.emptyMap());
    }

    public NativeToolCallConstraint(
            List<String> toolNames,
            Map<String, List<String>> argumentNamesByTool,
            Map<String, Map<String, List<String>>> argumentValuesByTool) {
        this(toolNames, argumentNamesByTool, argumentValuesByTool,
                Collections.emptyMap());
    }

    public NativeToolCallConstraint(
            List<String> toolNames,
            Map<String, List<String>> argumentNamesByTool,
            Map<String, Map<String, List<String>>> argumentValuesByTool,
            Map<String, Map<String, Object>> parameterSchemasByTool) {
        if (toolNames == null || toolNames.isEmpty()) {
            throw new IllegalArgumentException(
                    "NativeToolCallConstraint requires at least one tool name");
        }
        for (String name : toolNames) {
            if (name == null || name.isBlank()) {
                throw new IllegalArgumentException("Native tool names must not be blank");
            }
        }
        this.toolNames = Collections.unmodifiableList(List.copyOf(toolNames));
        this.callPrefixes = this.toolNames.stream()
                .map(name -> CALL_START + name + "(")
                .collect(Collectors.toList());

        Map<String, List<String>> copied = new LinkedHashMap<>();
        if (argumentNamesByTool != null) {
            argumentNamesByTool.forEach((name, arguments) -> {
                if (this.toolNames.contains(name) && arguments != null) {
                    copied.put(name, List.copyOf(arguments));
                }
            });
        }
        this.argumentNamesByTool = Collections.unmodifiableMap(copied);

        Map<String, Map<String, List<String>>> copiedValues = new LinkedHashMap<>();
        if (argumentValuesByTool != null) {
            argumentValuesByTool.forEach((toolName, valuesByArgument) -> {
                if (!this.toolNames.contains(toolName) || valuesByArgument == null) {
                    return;
                }
                Map<String, List<String>> values = new LinkedHashMap<>();
                valuesByArgument.forEach((argumentName, allowedValues) -> {
                    if (argumentName != null
                            && allowedValues != null
                            && !allowedValues.isEmpty()) {
                        values.put(argumentName, List.copyOf(allowedValues));
                    }
                });
                if (!values.isEmpty()) {
                    copiedValues.put(toolName, Collections.unmodifiableMap(values));
                }
            });
        }
        this.argumentValuesByTool = Collections.unmodifiableMap(copiedValues);

        Map<String, Map<String, Object>> copiedSchemas = new LinkedHashMap<>();
        if (parameterSchemasByTool != null) {
            parameterSchemasByTool.forEach((toolName, schema) -> {
                if (this.toolNames.contains(toolName) && schema != null) {
                    copiedSchemas.put(toolName,
                            Collections.unmodifiableMap(new LinkedHashMap<>(schema)));
                }
            });
        }
        this.parameterSchemasByTool = Collections.unmodifiableMap(copiedSchemas);
    }

    public NativeToolCallConstraint(String... toolNames) {
        this(Arrays.asList(toolNames));
    }

    @Override
    public boolean canExtend(String currentText, String piece) {
        String current = currentText == null ? "" : currentText;
        String extension = piece == null ? "" : piece;
        if (isAccepting(current)) {
            return false;
        }
        if (isOnlyWhitespace(extension)
                && endsWithWhitespace(current)
                && !isInsideQuotedLiteral(current)) {
            return false;
        }
        return validPrefix(current + extension);
    }

    /**
     * Structural whitespace is optional in the native function grammar. Allowing it to extend an
     * unchanged parser state forever lets a model exhaust its entire token budget without ever
     * reaching a call. One whitespace token remains legal between structural tokens; subsequent
     * whitespace-only tokens are masked until the model emits syntax. Whitespace inside a quoted
     * value remains data and is governed by that value's schema bounds.
     */
    private static boolean isOnlyWhitespace(String text) {
        return text != null
                && !text.isEmpty()
                && text.codePoints().allMatch(
                        codePoint -> Character.isWhitespace(codePoint)
                                || Character.isSpaceChar(codePoint));
    }

    private static boolean endsWithWhitespace(String text) {
        if (text == null || text.isEmpty()) {
            return false;
        }
        int codePoint = text.codePointBefore(text.length());
        return Character.isWhitespace(codePoint) || Character.isSpaceChar(codePoint);
    }

    static boolean repeatsStructuredValueWhitespace(
            String currentValue, String extension, Map<String, Object> schema) {
        String type = schema != null && schema.get("type") instanceof String
                ? (String) schema.get("type") : "";
        if (type.isEmpty() || "string".equals(type)
                || !JsonObjectConstraint.isOnlyJsonWhitespace(extension)
                || isInsideQuotedLiteral(currentValue)) {
            return false;
        }
        return currentValue != null
                && !currentValue.isEmpty()
                && JsonObjectConstraint.isJsonWhitespace(
                        currentValue.charAt(currentValue.length() - 1));
    }

    private static boolean isInsideQuotedLiteral(String text) {
        boolean quoted = false;
        boolean escaped = false;
        char quote = 0;
        for (int index = 0; index < text.length(); index++) {
            char current = text.charAt(index);
            if (quoted) {
                if (escaped) {
                    escaped = false;
                } else if (current == '\\') {
                    escaped = true;
                } else if (current == quote) {
                    quoted = false;
                }
            } else if (current == '\'' || current == '"') {
                quoted = true;
                quote = current;
            }
        }
        return quoted;
    }

    @Override
    public boolean allowsSpecialToken(String currentText, String piece) {
        String current = currentText == null ? "" : currentText;
        String candidate = piece == null ? "" : piece;
        if (candidate.isEmpty()) {
            return false;
        }
        if (current.isEmpty() && CALL_START.startsWith(candidate)) {
            return true;
        }
        return CALL_END.equals(candidate) && isCompleteCore(current);
    }

    @Override
    public boolean isAccepting(String currentText) {
        if (currentText == null || !currentText.endsWith(CALL_END)) {
            return false;
        }
        return isCompleteCore(
                currentText.substring(0, currentText.length() - CALL_END.length()));
    }

    private boolean isCompleteCore(String text) {
        for (int index = 0; index < callPrefixes.size(); index++) {
            if (isCompleteCore(index, text)) {
                return true;
            }
        }
        return false;
    }

    private boolean isCompleteCore(int index, String text) {
        String prefix = callPrefixes.get(index);
        if (!text.startsWith(prefix) || !text.endsWith(CALL_CLOSE)) {
            return false;
        }
        String body = text.substring(
                prefix.length(), text.length() - CALL_CLOSE.length());
        String toolName = toolNames.get(index);
        List<String> expectedArguments =
                argumentNamesByTool.getOrDefault(toolName, List.of());
        Map<String, List<String>> allowedValues =
                argumentValuesByTool.getOrDefault(toolName, Map.of());
        Map<String, Map<String, Object>> argumentSchemas =
                argumentSchemas(parameterSchemasByTool.get(toolName));
        return validBody(body, false)
                && validCompleteArgumentList(body)
                && validCompleteSchema(
                        body, expectedArguments, allowedValues, argumentSchemas);
    }

    private boolean validPrefix(String text) {
        for (int index = 0; index < callPrefixes.size(); index++) {
            String prefix = callPrefixes.get(index);
            if (prefix.startsWith(text)) {
                return true;
            }
            if (!text.startsWith(prefix)) {
                continue;
            }
            String body = text.substring(prefix.length());
            int close = body.lastIndexOf(CALL_CLOSE);
            if (close >= 0) {
                String coreBody = body.substring(0, close + CALL_CLOSE.length());
                String endSuffix = body.substring(close + CALL_CLOSE.length());
                if (CALL_END.startsWith(endSuffix)
                        && isCompleteCore(index, prefix + coreBody)) {
                    return true;
                }
            }
            String toolName = toolNames.get(index);
            List<String> expectedArguments =
                    argumentNamesByTool.getOrDefault(toolName, List.of());
            Map<String, List<String>> allowedValues =
                    argumentValuesByTool.getOrDefault(toolName, Map.of());
            Map<String, Map<String, Object>> argumentSchemas =
                    argumentSchemas(parameterSchemasByTool.get(toolName));
            if (validBody(body, true)
                    && validSchemaPrefix(
                            body, expectedArguments, allowedValues, argumentSchemas)) {
                return true;
            }
        }
        return false;
    }

    /**
     * Checks balanced native-argument text. A top-level ')' may only be the first
     * character of the terminal ")]" sequence.
     */
    private static boolean validBody(String body, boolean allowPartialClose) {
        int braces = 0;
        int brackets = 0;
        int parentheses = 0;
        boolean quoted = false;
        boolean escaped = false;
        char quote = 0;

        for (int i = 0; i < body.length(); i++) {
            char c = body.charAt(i);
            if (quoted) {
                if (escaped) {
                    escaped = false;
                } else if (c == '\\') {
                    escaped = true;
                } else if (c == quote) {
                    quoted = false;
                }
                continue;
            }

            if (c == '\'' || c == '"') {
                quoted = true;
                quote = c;
                continue;
            }

            switch (c) {
                case '{':
                    braces++;
                    break;
                case '}':
                    if (--braces < 0) return false;
                    break;
                case '[':
                    brackets++;
                    break;
                case ']':
                    if (braces == 0 && brackets == 0 && parentheses == 0
                            && i > 0 && body.charAt(i - 1) == ')') {
                        if (i != body.length() - 1) return false;
                        break;
                    }
                    if (--brackets < 0) return false;
                    break;
                case '(':
                    parentheses++;
                    break;
                case ')':
                    if (parentheses > 0) {
                        parentheses--;
                        break;
                    }
                    if (braces != 0 || brackets != 0) return false;
                    if (i == body.length() - 1) {
                        return allowPartialClose;
                    }
                    if (body.charAt(i + 1) != ']' || i + 2 != body.length()) {
                        return false;
                    }
                    break;
                default:
                    break;
            }
        }

        if (braces < 0 || brackets < 0 || parentheses < 0) {
            return false;
        }
        if (body.endsWith(CALL_CLOSE)) {
            return !quoted && braces == 0 && brackets == 0 && parentheses == 0;
        }
        return true;
    }

    private static boolean validSchemaPrefix(
            String body,
            List<String> expectedArguments,
            Map<String, List<String>> allowedValues,
            Map<String, Map<String, Object>> argumentSchemas) {
        if (!argumentSchemas.isEmpty()) {
            return validParameterSchemaPrefix(
                    body, expectedArguments, allowedValues, argumentSchemas);
        }
        if (expectedArguments == null || expectedArguments.isEmpty()) {
            return true;
        }

        if (body.endsWith(")") && !body.endsWith(CALL_CLOSE)) {
            return validCompleteSchema(
                    body.substring(0, body.length() - 1),
                    expectedArguments, allowedValues, argumentSchemas);
        }
        if (body.endsWith(CALL_CLOSE)) {
            return validCompleteSchema(
                    body.substring(0, body.length() - CALL_CLOSE.length()),
                    expectedArguments, allowedValues, argumentSchemas);
        }

        List<String> arguments = splitTopLevelArguments(body);
        if (arguments.size() > expectedArguments.size()) {
            return false;
        }
        for (int index = 0; index < arguments.size() - 1; index++) {
            String expectedName = expectedArguments.get(index);
            if (!validNamedArgument(
                    arguments.get(index), expectedName,
                    allowedValues.getOrDefault(expectedName, List.of()),
                    argumentSchemas.getOrDefault(expectedName, Map.of()))) {
                return false;
            }
        }
        int currentIndex = arguments.size() - 1;
        if (currentIndex < 0 || currentIndex >= expectedArguments.size()) {
            return false;
        }
        String expectedName = expectedArguments.get(currentIndex);
        return validPartialNamedArgument(
                arguments.get(currentIndex), expectedName,
                allowedValues.getOrDefault(expectedName, List.of()),
                argumentSchemas.getOrDefault(expectedName, Map.of()));
    }

    private static boolean validCompleteSchema(
            String body,
            List<String> expectedArguments,
            Map<String, List<String>> allowedValues,
            Map<String, Map<String, Object>> argumentSchemas) {
        if (!argumentSchemas.isEmpty()) {
            return validCompleteParameterSchema(
                    body, expectedArguments, allowedValues, argumentSchemas);
        }
        if (expectedArguments == null || expectedArguments.isEmpty()) {
            return true;
        }
        List<String> arguments = splitTopLevelArguments(body);
        if (arguments.size() != expectedArguments.size()) {
            return false;
        }
        for (int index = 0; index < arguments.size(); index++) {
            String expectedName = expectedArguments.get(index);
            if (!validNamedArgument(
                    arguments.get(index), expectedName,
                    allowedValues.getOrDefault(expectedName, List.of()),
                    argumentSchemas.getOrDefault(expectedName, Map.of()))) {
                return false;
            }
        }
        return true;
    }

    private static boolean validParameterSchemaPrefix(
            String body,
            List<String> requiredArguments,
            Map<String, List<String>> allowedValues,
            Map<String, Map<String, Object>> argumentSchemas) {
        if (body.endsWith(")") && !body.endsWith(CALL_CLOSE)) {
            return validCompleteParameterSchema(
                    body.substring(0, body.length() - 1),
                    requiredArguments, allowedValues, argumentSchemas);
        }
        if (body.endsWith(CALL_CLOSE)) {
            return validCompleteParameterSchema(
                    body.substring(0, body.length() - CALL_CLOSE.length()),
                    requiredArguments, allowedValues, argumentSchemas);
        }

        List<String> arguments = splitTopLevelArguments(body);
        List<String> seen = new ArrayList<>();
        for (int index = 0; index < arguments.size() - 1; index++) {
            String argument = arguments.get(index);
            String name = completeArgumentName(argument);
            if (name == null || seen.contains(name)
                    || !validRequiredArgumentOrder(name, requiredArguments, seen)
                    || !argumentSchemas.containsKey(name)
                    || !validNamedArgument(
                            argument, name,
                            allowedValues.getOrDefault(name, List.of()),
                            argumentSchemas.get(name))) {
                return false;
            }
            seen.add(name);
        }

        String current = arguments.isEmpty() ? "" : arguments.get(arguments.size() - 1);
        return validPartialDeclaredArgument(
                current, seen, requiredArguments, allowedValues, argumentSchemas);
    }

    private static boolean validCompleteParameterSchema(
            String body,
            List<String> requiredArguments,
            Map<String, List<String>> allowedValues,
            Map<String, Map<String, Object>> argumentSchemas) {
        List<String> seen = new ArrayList<>();
        if (!body.isBlank()) {
            for (String argument : splitTopLevelArguments(body)) {
                String name = completeArgumentName(argument);
                if (name == null || seen.contains(name)
                        || !validRequiredArgumentOrder(name, requiredArguments, seen)
                        || !argumentSchemas.containsKey(name)
                        || !validNamedArgument(
                                argument, name,
                                allowedValues.getOrDefault(name, List.of()),
                                argumentSchemas.get(name))) {
                    return false;
                }
                seen.add(name);
            }
        }
        return requiredArguments == null || seen.containsAll(requiredArguments);
    }

    private static boolean validPartialDeclaredArgument(
            String argument,
            List<String> seen,
            List<String> requiredArguments,
            Map<String, List<String>> allowedValues,
            Map<String, Map<String, Object>> argumentSchemas) {
        String part = argument.stripLeading();
        if (part.isBlank()) {
            return seen.size() < argumentSchemas.size();
        }
        int equals = topLevelEquals(part);
        if (equals < 0) {
            for (String name : argumentSchemas.keySet()) {
                if (!seen.contains(name)
                        && validRequiredArgumentOrder(name, requiredArguments, seen)
                        && (name.startsWith(part)
                                || part.startsWith(name)
                                && part.substring(name.length()).isBlank())) {
                    return true;
                }
            }
            return false;
        }

        String name = part.substring(0, equals).trim();
        if (seen.contains(name)
                || !validRequiredArgumentOrder(name, requiredArguments, seen)
                || !argumentSchemas.containsKey(name)) {
            return false;
        }
        String value = part.substring(equals + 1).stripLeading();
        List<String> allowed = allowedValues.getOrDefault(name, List.of());
        if (!allowed.isEmpty()) {
            boolean matchingValue = false;
            for (String candidate : allowed) {
                if (nativeStringLiteral(candidate).startsWith(value)) {
                    matchingValue = true;
                    break;
                }
            }
            if (!matchingValue) {
                return false;
            }
        }
        return validValuePrefix(value, argumentSchemas.get(name));
    }

    /**
     * Required native arguments are emitted in schema order so a model cannot start dependent
     * fields before their prerequisites. Optional arguments may still appear at any point.
     */
    private static boolean validRequiredArgumentOrder(
            String name, List<String> requiredArguments, List<String> seen) {
        if (requiredArguments == null || !requiredArguments.contains(name)) {
            return true;
        }
        for (String required : requiredArguments) {
            if (!seen.contains(required)) {
                return required.equals(name);
            }
        }
        return false;
    }

    private static String completeArgumentName(String argument) {
        String part = argument == null ? "" : argument.trim();
        int equals = topLevelEquals(part);
        if (equals <= 0) {
            return null;
        }
        String name = part.substring(0, equals).trim();
        return ARGUMENT_NAME.matcher(name).matches() ? name : null;
    }

    private static boolean validNamedArgument(
            String argument,
            String expectedName,
            List<String> allowedValues,
            Map<String, Object> argumentSchema) {
        if (!validArgument(argument)) {
            return false;
        }
        String part = argument.trim();
        int equals = topLevelEquals(part);
        if (equals <= 0 || !expectedName.equals(part.substring(0, equals).trim())) {
            return false;
        }
        String value = part.substring(equals + 1).trim();
        if (allowedValues != null && !allowedValues.isEmpty()
                && allowedValues.stream()
                        .map(NativeToolCallConstraint::nativeStringLiteral)
                        .noneMatch(value::equals)) {
            return false;
        }
        return argumentSchema == null || argumentSchema.isEmpty()
                || validCompleteValue(value, argumentSchema);
    }

    private static boolean validPartialNamedArgument(
            String argument,
            String expectedName,
            List<String> allowedValues,
            Map<String, Object> argumentSchema) {
        String part = argument.stripLeading();
        if (part.isBlank()) {
            return true;
        }
        int equals = topLevelEquals(part);
        if (equals < 0) {
            if (expectedName.startsWith(part)) {
                return true;
            }
            return part.startsWith(expectedName)
                    && part.substring(expectedName.length()).isBlank();
        }
        if (!expectedName.equals(part.substring(0, equals).trim())) {
            return false;
        }
        String value = part.substring(equals + 1).stripLeading();
        if (allowedValues != null && !allowedValues.isEmpty()) {
            boolean allowedPrefix = false;
            for (String allowedValue : allowedValues) {
                if (nativeStringLiteral(allowedValue).startsWith(value)) {
                    allowedPrefix = true;
                    break;
                }
            }
            if (!allowedPrefix) {
                return false;
            }
        }
        return argumentSchema == null || argumentSchema.isEmpty()
                || validValuePrefix(value, argumentSchema);
    }

    private static String nativeStringLiteral(String value) {
        String text = value == null ? "" : value;
        return "\"" + text
                .replace("\\", "\\\\")
                .replace("\"", "\\\"")
                .replace("\n", "\\n")
                .replace("\r", "\\r")
                .replace("\t", "\\t") + "\"";
    }

    private static Map<String, Map<String, Object>> argumentSchemas(
            Map<String, Object> parameterSchema) {
        if (parameterSchema == null
                || !(parameterSchema.get("properties") instanceof Map<?, ?>)) {
            return Map.of();
        }
        Map<String, Map<String, Object>> result = new LinkedHashMap<>();
        ((Map<?, ?>) parameterSchema.get("properties")).forEach((name, schema) -> {
            if (name instanceof String && schema instanceof Map<?, ?>) {
                Map<String, Object> copied = new LinkedHashMap<>();
                ((Map<?, ?>) schema).forEach(
                        (key, value) -> copied.put(String.valueOf(key), value));
                result.put((String) name, Collections.unmodifiableMap(copied));
            }
        });
        return Collections.unmodifiableMap(result);
    }

    static boolean validCompleteValue(
            String rawValue, Map<String, Object> schema) {
        String value = JsonObjectConstraint.stripJsonWhitespace(rawValue);
        if (value.isEmpty()) {
            return false;
        }
        if ("string".equals(schema.get("type"))) {
            NativeStringState string = parseNativeString(value);
            return string.valid && string.closed
                    && ToolSchemaValidator.isValidValue(string.value, schema);
        }
        try {
            JsonNode parsed = MAPPER.readTree(value);
            if (parsed == null || parsed.isMissingNode()) {
                return false;
            }
            Object converted = MAPPER.convertValue(parsed, Object.class);
            return ToolSchemaValidator.isValidValue(converted, schema);
        } catch (Exception ignored) {
            return false;
        }
    }

    static boolean validValuePrefix(
            String rawValue, Map<String, Object> schema) {
        String value = JsonObjectConstraint.stripLeadingJsonWhitespace(rawValue);
        if (value.isEmpty()) {
            return true;
        }
        String type = schema.get("type") instanceof String
                ? (String) schema.get("type") : "";
        switch (type) {
            case "string":
                return validStringPrefix(value, schema);
            case "array":
                return validArrayPrefix(value, schema);
            case "object":
                return validJsonCompositePrefix(value, schema, '{', '}');
            case "integer":
            case "number":
                return validNumberPrefix(value, schema);
            case "boolean":
                return "true".startsWith(value) || "false".startsWith(value);
            case "null":
                return "null".startsWith(value);
            default:
                return true;
        }
    }

    private static boolean validStringPrefix(
            String value, Map<String, Object> schema) {
        NativeStringState state = parseNativeString(value);
        if (!state.valid) {
            return false;
        }
        if (!validStringValuePrefix(state.value, schema)
                || !validIncompleteStringEscapePrefix(value, state.value, schema)) {
            return false;
        }
        return !state.closed || ToolSchemaValidator.isValidValue(state.value, schema);
    }

    private static boolean validStringValuePrefix(
            String value, Map<String, Object> schema) {
        int length = value.codePointCount(0, value.length());
        Object maxLength = schema.get("maxLength");
        if (maxLength instanceof Number
                && length > ((Number) maxLength).intValue()) {
            return false;
        }

        if (!validPatternPrefix(value, schema)) {
            return false;
        }
        List<String> allowed = allowedStringValues(schema);
        return allowed.isEmpty() || allowed.stream().anyMatch(item -> item.startsWith(value));
    }

    private static boolean validPatternPrefix(String value, Map<String, Object> schema) {
        Object configuredPattern = schema.get("pattern");
        if (!(configuredPattern instanceof String) || ((String) configuredPattern).isEmpty()) {
            return true;
        }
        try {
            var matcher = Pattern.compile((String) configuredPattern).matcher(value);
            return matcher.find() || matcher.hitEnd();
        } catch (java.util.regex.PatternSyntaxException invalidSchema) {
            return false;
        }
    }

    private static boolean validIncompleteStringEscapePrefix(
            String rawValue, String decodedValue, Map<String, Object> schema) {
        boolean trailingEscape = hasTrailingIncompleteEscape(rawValue);
        String digits = incompleteUnicodeEscapeDigits(rawValue);
        int length = decodedValue.codePointCount(0, decodedValue.length());
        Object maxLength = schema.get("maxLength");
        if ((trailingEscape || digits != null)
                && maxLength instanceof Number
                && length >= ((Number) maxLength).intValue()) {
            return false;
        }
        List<String> allowed = allowedStringValues(schema);
        if (trailingEscape && !allowed.isEmpty()) {
            return allowed.stream().anyMatch(candidate -> candidate.startsWith(decodedValue)
                    && candidate.length() > decodedValue.length());
        }
        if (digits == null) {
            return true;
        }

        if (allowed.isEmpty()) {
            return true;
        }
        String normalizedDigits = digits.toLowerCase(java.util.Locale.ROOT);
        for (String candidate : allowed) {
            if (!candidate.startsWith(decodedValue)
                    || candidate.length() <= decodedValue.length()) {
                continue;
            }
            String hex = Integer.toHexString(candidate.charAt(decodedValue.length()));
            String padded = "0000".substring(hex.length()) + hex;
            if (padded.startsWith(normalizedDigits)) {
                return true;
            }
        }
        return false;
    }

    private static boolean hasTrailingIncompleteEscape(String rawValue) {
        if (rawValue == null || rawValue.isEmpty()) {
            return false;
        }
        int slashCount = 0;
        for (int index = rawValue.length() - 1;
                index >= 0 && rawValue.charAt(index) == '\\'; index--) {
            slashCount++;
        }
        return (slashCount & 1) == 1;
    }

    private static String incompleteUnicodeEscapeDigits(String rawValue) {
        if (rawValue == null || rawValue.isEmpty()) {
            return null;
        }
        int marker = rawValue.lastIndexOf("\\u");
        if (marker < 0) {
            return null;
        }
        int slashCount = 0;
        for (int index = marker; index >= 0 && rawValue.charAt(index) == '\\'; index--) {
            slashCount++;
        }
        if ((slashCount & 1) == 0) {
            return null;
        }
        String digits = rawValue.substring(marker + 2);
        if (digits.length() >= 4) {
            return null;
        }
        for (int index = 0; index < digits.length(); index++) {
            if (Character.digit(digits.charAt(index), 16) < 0) {
                return null;
            }
        }
        return digits;
    }

    private static List<String> allowedStringValues(Map<String, Object> schema) {
        if (schema.get("const") instanceof String) {
            return List.of((String) schema.get("const"));
        }
        List<String> allowed = new ArrayList<>();
        Object enumObject = schema.get("enum");
        if (enumObject instanceof java.util.Collection<?>) {
            for (Object candidate : (java.util.Collection<?>) enumObject) {
                if (candidate instanceof String) {
                    allowed.add((String) candidate);
                }
            }
        }
        return allowed;
    }

    private static NativeStringState parseNativeString(String raw) {
        String value = JsonObjectConstraint.stripLeadingJsonWhitespace(raw);
        if (value.isEmpty() || value.charAt(0) != '"') {
            return NativeStringState.invalid();
        }
        StringBuilder decoded = new StringBuilder();
        boolean escaped = false;
        for (int index = 1; index < value.length(); index++) {
            char current = value.charAt(index);
            if (escaped) {
                switch (current) {
                    case 'n':
                        decoded.append('\n');
                        break;
                    case 'r':
                        decoded.append('\r');
                        break;
                    case 't':
                        decoded.append('\t');
                        break;
                    case 'b':
                        decoded.append('\b');
                        break;
                    case 'f':
                        decoded.append('\f');
                        break;
                    case '"':
                    case '\\':
                    case '/':
                        decoded.append(current);
                        break;
                    case 'u':
                        int available = Math.min(4, value.length() - index - 1);
                        for (int digit = 1; digit <= available; digit++) {
                            if (Character.digit(value.charAt(index + digit), 16) < 0) {
                                return NativeStringState.invalid();
                            }
                        }
                        if (available < 4) {
                            return new NativeStringState(
                                    true, false, decoded.toString());
                        }
                        decoded.append((char) Integer.parseInt(
                                value.substring(index + 1, index + 5), 16));
                        index += 4;
                        break;
                    default:
                        return NativeStringState.invalid();
                }
                escaped = false;
                continue;
            }
            if (current == '\\') {
                escaped = true;
                continue;
            }
            if (current == '"') {
                if (!value.substring(index + 1).isEmpty()
                        && !JsonObjectConstraint.isOnlyJsonWhitespace(
                                value.substring(index + 1))) {
                    return NativeStringState.invalid();
                }
                return new NativeStringState(true, true, decoded.toString());
            }
            if (Character.isISOControl(current)) {
                return NativeStringState.invalid();
            }
            decoded.append(current);
        }
        return new NativeStringState(true, false, decoded.toString());
    }

    private static boolean validArrayPrefix(
            String value, Map<String, Object> schema) {
        ArrayPrefixState state = parseArrayPrefix(value);
        if (!state.valid) {
            return false;
        }
        if (state.closed) {
            return validCompleteValue(value, schema);
        }

        int maxItems = schema.get("maxItems") instanceof Number
                ? Math.max(0, ((Number) schema.get("maxItems")).intValue())
                : Integer.MAX_VALUE;

        List<Object> completeValues = new ArrayList<>();
        int lastIndex = state.items.size() - 1;
        for (int index = 0; index < lastIndex; index++) {
            String item = JsonObjectConstraint.stripJsonWhitespace(state.items.get(index));
            Map<String, Object> itemSchema = arrayItemSchema(schema, index);
            if (itemSchema == null) {
                return false;
            }
            if (item.isEmpty() || !validCompleteValue(item, itemSchema)) {
                return false;
            }
            Object parsed = parseJsonValue(item);
            if (parsed == INVALID_VALUE) {
                return false;
            }
            // A tokenizer token may span the end of one item, the comma, and the start of the
            // next item. In that case a duplicate can move directly into completeValues without
            // ever being observed as the current item below. Enforce uniqueItems while collecting
            // every completed value as well as for the currently open value.
            if (Boolean.TRUE.equals(schema.get("uniqueItems"))
                    && completeValues.contains(parsed)) {
                return false;
            }
            completeValues.add(parsed);
        }
        String current = lastIndex < 0 ? ""
                : JsonObjectConstraint.stripLeadingJsonWhitespace(state.items.get(lastIndex));
        if (completeValues.size() >= maxItems) {
            return lastIndex == 0 && current.isEmpty();
        }
        if (current.isEmpty()) {
            return lastIndex == 0 || arrayItemSchema(schema, lastIndex) != null;
        }
        Map<String, Object> itemSchema = arrayItemSchema(schema, lastIndex);
        if (itemSchema == null) {
            return false;
        }
        if (!validValuePrefix(current, itemSchema)) {
            return false;
        }
        if (Boolean.TRUE.equals(schema.get("uniqueItems"))) {
            Boolean viable = uniqueObjectCompletionViability(
                    current, itemSchema, completeValues);
            if (Boolean.FALSE.equals(viable)) {
                return false;
            }
        }
        if (Boolean.TRUE.equals(schema.get("uniqueItems"))
                && validCompleteValue(current, itemSchema)) {
            Object parsed = parseJsonValue(current);
            return parsed != INVALID_VALUE && !completeValues.contains(parsed);
        }
        if (Boolean.TRUE.equals(schema.get("uniqueItems"))) {
            String forcedCompletion = forcedCompleteObjectValue(current, itemSchema);
            if (forcedCompletion != null) {
                Object parsed = parseJsonValue(forcedCompletion);
                if (parsed != INVALID_VALUE && completeValues.contains(parsed)) {
                    return false;
                }
            }
        }
        return true;
    }

    /**
     * Determines whether an incomplete final string property still has an enum/const completion
     * that makes the enclosing object distinct. A null result means the prefix cannot be decided
     * locally and ordinary incremental validation remains authoritative.
     */
    private static Boolean uniqueObjectCompletionViability(
            String value,
            Map<String, Object> schema,
            List<Object> completeValues) {
        if (!"object".equals(schema.get("type"))) {
            return null;
        }
        ObjectPrefixState state = parseObjectPrefix(value);
        if (!state.valid || state.closed || state.members.isEmpty()) {
            return null;
        }
        Map<String, Map<String, Object>> properties = objectPropertySchemas(schema);
        Object additionalProperties = schema.get("additionalProperties");
        List<String> seen = new ArrayList<>();
        int lastIndex = state.members.size() - 1;
        for (int index = 0; index < lastIndex; index++) {
            ObjectMember member = completeObjectMember(
                    state.members.get(index), properties, additionalProperties, seen);
            if (member == null) {
                return null;
            }
            seen.add(member.name);
        }

        String lastMember = state.members.get(lastIndex);
        int colon = topLevelColon(lastMember);
        if (colon <= 0) {
            return null;
        }
        NativeStringState name = parseNativeString(lastMember.substring(0, colon).trim());
        if (!name.valid || !name.closed || seen.contains(name.value)) {
            return null;
        }
        Map<String, Object> valueSchema = objectPropertySchema(
                name.value, properties, additionalProperties);
        if (valueSchema == null || !"string".equals(valueSchema.get("type"))) {
            return null;
        }
        NativeStringState partial = parseNativeString(
                JsonObjectConstraint.stripLeadingJsonWhitespace(
                        lastMember.substring(colon + 1)));
        if (!partial.valid || partial.closed) {
            return null;
        }
        List<String> allowed = allowedStringValues(valueSchema);
        if (allowed.isEmpty()) {
            return null;
        }

        seen.add(name.value);
        if (hasAvailableObjectProperty(properties, additionalProperties, seen)) {
            return null;
        }
        Object required = schema.get("required");
        if (required instanceof java.util.Collection<?>
                && ((java.util.Collection<?>) required).stream()
                        .map(String::valueOf)
                        .anyMatch(requiredName -> !seen.contains(requiredName))) {
            return null;
        }

        for (String candidateValue : allowed) {
            if (!candidateValue.startsWith(partial.value)) {
                continue;
            }
            StringBuilder completed = new StringBuilder("{");
            for (int index = 0; index < lastIndex; index++) {
                if (index > 0) {
                    completed.append(',');
                }
                completed.append(state.members.get(index));
            }
            if (lastIndex > 0) {
                completed.append(',');
            }
            completed.append(lastMember, 0, colon + 1)
                    .append(nativeStringLiteral(candidateValue))
                    .append('}');
            String candidate = completed.toString();
            if (validCompleteValue(candidate, schema)) {
                Object parsed = parseJsonValue(candidate);
                if (parsed != INVALID_VALUE && !completeValues.contains(parsed)) {
                    return true;
                }
            }
        }
        return false;
    }

    /**
     * Completes an open object only when its current member is complete and the schema permits no
     * further member. This lets an enclosing {@code uniqueItems} array reject the token that makes a
     * duplicate inevitable, rather than accepting it and discovering a dead end at the closing brace.
     */
    private static String forcedCompleteObjectValue(
            String value, Map<String, Object> schema) {
        if (!"object".equals(schema.get("type"))) {
            return null;
        }
        ObjectPrefixState state = parseObjectPrefix(value);
        if (!state.valid || state.closed || state.members.isEmpty()) {
            return null;
        }
        Map<String, Map<String, Object>> properties = objectPropertySchemas(schema);
        Object additionalProperties = schema.get("additionalProperties");
        List<String> seen = new ArrayList<>();
        for (String rawMember : state.members) {
            ObjectMember member = completeObjectMember(
                    rawMember, properties, additionalProperties, seen);
            if (member == null) {
                return null;
            }
            seen.add(member.name);
        }
        if (hasAvailableObjectProperty(properties, additionalProperties, seen)) {
            return null;
        }
        Object required = schema.get("required");
        if (required instanceof java.util.Collection<?>
                && ((java.util.Collection<?>) required).stream()
                        .map(String::valueOf)
                        .anyMatch(name -> !seen.contains(name))) {
            return null;
        }
        String candidate = value + "}";
        return validCompleteValue(candidate, schema) ? candidate : null;
    }

    /**
     * Resolves the JSON Schema item contract for one array position. Draft 2020-12 prefixItems
     * takes precedence for its declared slots; items governs only the remaining tail.
     */
    private static Map<String, Object> arrayItemSchema(
            Map<String, Object> schema, int index) {
        Object prefixItems = schema.get("prefixItems");
        if (prefixItems instanceof List<?> && index >= 0) {
            List<?> positional = (List<?>) prefixItems;
            if (index < positional.size()) {
                Object positionalSchema = positional.get(index);
                if (!(positionalSchema instanceof Map<?, ?>)) {
                    return Map.of();
                }
                Map<String, Object> copied = new LinkedHashMap<>();
                ((Map<?, ?>) positionalSchema).forEach(
                        (key, item) -> copied.put(String.valueOf(key), item));
                return copied;
            }
        }
        if (Boolean.FALSE.equals(schema.get("items"))) {
            return null;
        }
        if (schema.get("items") instanceof Map<?, ?>) {
            Map<String, Object> copied = new LinkedHashMap<>();
            ((Map<?, ?>) schema.get("items")).forEach(
                    (key, item) -> copied.put(String.valueOf(key), item));
            return copied;
        }
        return Map.of();
    }

    private static boolean validJsonCompositePrefix(
            String value,
            Map<String, Object> schema,
            char open,
            char close) {
        if (open != '{' || close != '}') {
            return false;
        }
        ObjectPrefixState state = parseObjectPrefix(value);
        if (!state.valid) {
            return false;
        }
        if (state.closed) {
            return validCompleteValue(value, schema);
        }

        Map<String, Map<String, Object>> properties = objectPropertySchemas(schema);
        Object additionalProperties = schema.get("additionalProperties");
        List<String> seen = new ArrayList<>();
        int lastIndex = state.members.size() - 1;
        for (int index = 0; index < lastIndex; index++) {
            ObjectMember member = completeObjectMember(
                    state.members.get(index), properties, additionalProperties, seen);
            if (member == null) {
                return false;
            }
            seen.add(member.name);
        }

        String current = lastIndex < 0 ? "" : state.members.get(lastIndex);
        return validPartialObjectMember(
                current, properties, additionalProperties, seen);
    }

    private static Map<String, Map<String, Object>> objectPropertySchemas(
            Map<String, Object> schema) {
        if (schema == null || !(schema.get("properties") instanceof Map<?, ?>)) {
            return Map.of();
        }
        Map<String, Map<String, Object>> result = new LinkedHashMap<>();
        ((Map<?, ?>) schema.get("properties")).forEach((name, propertySchema) -> {
            if (name != null && propertySchema instanceof Map<?, ?>) {
                Map<String, Object> copied = new LinkedHashMap<>();
                ((Map<?, ?>) propertySchema).forEach(
                        (key, item) -> copied.put(String.valueOf(key), item));
                result.put(String.valueOf(name), Collections.unmodifiableMap(copied));
            }
        });
        return Collections.unmodifiableMap(result);
    }

    private static ObjectMember completeObjectMember(
            String rawMember,
            Map<String, Map<String, Object>> properties,
            Object additionalProperties,
            List<String> seen) {
        String member = JsonObjectConstraint.stripJsonWhitespace(rawMember);
        int colon = topLevelColon(member);
        if (colon <= 0) {
            return null;
        }
        NativeStringState name = parseNativeString(
                JsonObjectConstraint.stripJsonWhitespace(member.substring(0, colon)));
        if (!name.valid || !name.closed || seen.contains(name.value)) {
            return null;
        }
        Map<String, Object> valueSchema = objectPropertySchema(
                name.value, properties, additionalProperties);
        String value = JsonObjectConstraint.stripJsonWhitespace(
                member.substring(colon + 1));
        if (valueSchema == null || !validCompleteValue(value, valueSchema)) {
            return null;
        }
        return new ObjectMember(name.value);
    }

    private static boolean validPartialObjectMember(
            String rawMember,
            Map<String, Map<String, Object>> properties,
            Object additionalProperties,
            List<String> seen) {
        String member = JsonObjectConstraint.stripLeadingJsonWhitespace(rawMember);
        if (member.isEmpty()) {
            return hasAvailableObjectProperty(properties, additionalProperties, seen);
        }

        int colon = topLevelColon(member);
        if (colon < 0) {
            NativeStringState name = parseNativeString(member);
            if (!name.valid) {
                return false;
            }
            if (name.closed) {
                return !seen.contains(name.value)
                        && objectPropertySchema(
                                name.value, properties, additionalProperties) != null;
            }
            if (allowsAdditionalObjectProperty(additionalProperties)) {
                return true;
            }
            List<String> availableProperties = new ArrayList<>();
            for (String candidate : properties.keySet()) {
                if (!seen.contains(candidate)) {
                    availableProperties.add(candidate);
                }
            }
            if (availableProperties.isEmpty()) {
                return false;
            }
            Map<String, Object> propertyNameSchema = Map.of(
                    "type", "string",
                    "enum", availableProperties);
            return validStringValuePrefix(name.value, propertyNameSchema)
                    && validIncompleteStringEscapePrefix(
                    member, name.value, propertyNameSchema);
        }

        NativeStringState name = parseNativeString(
                JsonObjectConstraint.stripJsonWhitespace(member.substring(0, colon)));
        if (!name.valid || !name.closed || seen.contains(name.value)) {
            return false;
        }
        Map<String, Object> valueSchema = objectPropertySchema(
                name.value, properties, additionalProperties);
        if (valueSchema == null) {
            return false;
        }
        String value = JsonObjectConstraint.stripLeadingJsonWhitespace(
                member.substring(colon + 1));
        return value.isEmpty() || validValuePrefix(value, valueSchema);
    }

    private static boolean hasAvailableObjectProperty(
            Map<String, Map<String, Object>> properties,
            Object additionalProperties,
            List<String> seen) {
        return allowsAdditionalObjectProperty(additionalProperties)
                || properties.keySet().stream().anyMatch(name -> !seen.contains(name));
    }

    private static boolean allowsAdditionalObjectProperty(Object additionalProperties) {
        return !(additionalProperties instanceof Boolean)
                || Boolean.TRUE.equals(additionalProperties)
                || additionalProperties instanceof Map<?, ?>;
    }

    private static Map<String, Object> objectPropertySchema(
            String name,
            Map<String, Map<String, Object>> properties,
            Object additionalProperties) {
        Map<String, Object> declared = properties.get(name);
        if (declared != null) {
            return declared;
        }
        if (Boolean.FALSE.equals(additionalProperties)) {
            return null;
        }
        if (additionalProperties instanceof Map<?, ?>) {
            Map<String, Object> copied = new LinkedHashMap<>();
            ((Map<?, ?>) additionalProperties).forEach(
                    (key, value) -> copied.put(String.valueOf(key), value));
            return copied;
        }
        return Map.of();
    }

    private static int topLevelColon(String text) {
        int braces = 0;
        int brackets = 0;
        boolean quoted = false;
        boolean escaped = false;
        for (int index = 0; index < text.length(); index++) {
            char current = text.charAt(index);
            if (quoted) {
                if (escaped) {
                    escaped = false;
                } else if (current == '\\') {
                    escaped = true;
                } else if (current == '"') {
                    quoted = false;
                }
                continue;
            }
            if (current == '"') {
                quoted = true;
            } else if (current == '{') {
                braces++;
            } else if (current == '}') {
                braces--;
            } else if (current == '[') {
                brackets++;
            } else if (current == ']') {
                brackets--;
            } else if (current == ':' && braces == 0 && brackets == 0) {
                return index;
            }
        }
        return -1;
    }

    private static ObjectPrefixState parseObjectPrefix(String raw) {
        String value = JsonObjectConstraint.stripLeadingJsonWhitespace(raw);
        if (value.isEmpty() || value.charAt(0) != '{') {
            return ObjectPrefixState.invalid();
        }
        List<String> members = new ArrayList<>();
        List<Character> stack = new ArrayList<>();
        boolean quoted = false;
        boolean escaped = false;
        int start = 1;
        for (int index = 1; index < value.length(); index++) {
            char current = value.charAt(index);
            if (quoted) {
                if (escaped) {
                    escaped = false;
                } else if (current == '\\') {
                    escaped = true;
                } else if (current == '"') {
                    quoted = false;
                } else if (Character.isISOControl(current)) {
                    return ObjectPrefixState.invalid();
                }
                continue;
            }
            if (current == '"') {
                quoted = true;
                continue;
            }
            if (current == '{' || current == '[') {
                stack.add(current);
                continue;
            }
            if (current == '}' || current == ']') {
                if (current == '}' && stack.isEmpty()) {
                    members.add(value.substring(start, index));
                    if (!value.substring(index + 1).isEmpty()
                            && !JsonObjectConstraint.isOnlyJsonWhitespace(
                                    value.substring(index + 1))) {
                        return ObjectPrefixState.invalid();
                    }
                    return new ObjectPrefixState(true, true, members);
                }
                if (stack.isEmpty() || !matching(stack.remove(stack.size() - 1), current)) {
                    return ObjectPrefixState.invalid();
                }
                continue;
            }
            if (current == ',' && stack.isEmpty()) {
                members.add(value.substring(start, index));
                start = index + 1;
            }
        }
        members.add(value.substring(start));
        return new ObjectPrefixState(true, false, members);
    }

    private static boolean validNumberPrefix(
            String value, Map<String, Object> schema) {
        if (value == null
                || !value.matches("[ \\t\\r\\n]*-?[0-9]*(\\.[0-9]*)?([eE][+-]?[0-9]*)?[ \\t\\r\\n]*")) {
            return false;
        }
        String trimmed = JsonObjectConstraint.stripJsonWhitespace(value);
        boolean trailingJsonWhitespace = !value.isEmpty()
                && isJsonWhitespace(value.charAt(value.length() - 1));
        if ("-".equals(trimmed)
                && schema.get("minimum") instanceof Number
                && ((Number) schema.get("minimum")).doubleValue() >= 0.0d) {
            return false;
        }
        if (trimmed.isEmpty() || "-".equals(trimmed)
                || trimmed.endsWith(".") || trimmed.endsWith("e")
                || trimmed.endsWith("E") || trimmed.endsWith("+")
                || trimmed.endsWith("-")) {
            return !trailingJsonWhitespace;
        }
        return validCompleteValue(trimmed, schema);
    }

    private static boolean isJsonWhitespace(char value) {
        return JsonObjectConstraint.isJsonWhitespace(value);
    }

    private static ArrayPrefixState parseArrayPrefix(String raw) {
        String value = JsonObjectConstraint.stripLeadingJsonWhitespace(raw);
        if (value.isEmpty() || value.charAt(0) != '[') {
            return ArrayPrefixState.invalid();
        }
        List<String> items = new ArrayList<>();
        List<Character> stack = new ArrayList<>();
        boolean quoted = false;
        boolean escaped = false;
        int start = 1;
        for (int index = 1; index < value.length(); index++) {
            char current = value.charAt(index);
            if (quoted) {
                if (escaped) {
                    escaped = false;
                } else if (current == '\\') {
                    escaped = true;
                } else if (current == '"') {
                    quoted = false;
                } else if (current <= 0x1f) {
                    return ArrayPrefixState.invalid();
                }
                continue;
            }
            if (current == '"') {
                quoted = true;
                continue;
            }
            if (current == '{' || current == '[' || current == '(') {
                stack.add(current);
                continue;
            }
            if (current == '}' || current == ']' || current == ')') {
                if (current == ']' && stack.isEmpty()) {
                    items.add(value.substring(start, index));
                    if (!value.substring(index + 1).isEmpty()
                            && !JsonObjectConstraint.isOnlyJsonWhitespace(
                                    value.substring(index + 1))) {
                        return ArrayPrefixState.invalid();
                    }
                    return new ArrayPrefixState(true, true, items);
                }
                if (stack.isEmpty() || !matching(stack.remove(stack.size() - 1), current)) {
                    return ArrayPrefixState.invalid();
                }
                continue;
            }
            if (current == ',' && stack.isEmpty()) {
                items.add(value.substring(start, index));
                start = index + 1;
            }
        }
        items.add(value.substring(start));
        return new ArrayPrefixState(true, false, items);
    }

    private static boolean matching(char open, char close) {
        return open == '{' && close == '}'
                || open == '[' && close == ']'
                || open == '(' && close == ')';
    }

    private static final Object INVALID_VALUE = new Object();

    private static Object parseJsonValue(String raw) {
        try {
            JsonNode parsed = MAPPER.readTree(raw);
            return parsed == null ? INVALID_VALUE
                    : MAPPER.convertValue(parsed, Object.class);
        } catch (Exception ignored) {
            return INVALID_VALUE;
        }
    }

    private static final class ObjectMember {
        private final String name;

        private ObjectMember(String name) {
            this.name = name;
        }
    }

    private static final class ObjectPrefixState {
        private final boolean valid;
        private final boolean closed;
        private final List<String> members;

        private ObjectPrefixState(boolean valid, boolean closed, List<String> members) {
            this.valid = valid;
            this.closed = closed;
            this.members = members;
        }

        private static ObjectPrefixState invalid() {
            return new ObjectPrefixState(false, false, List.of());
        }
    }

    private static final class NativeStringState {
        private final boolean valid;
        private final boolean closed;
        private final String value;

        private NativeStringState(boolean valid, boolean closed, String value) {
            this.valid = valid;
            this.closed = closed;
            this.value = value;
        }

        private static NativeStringState invalid() {
            return new NativeStringState(false, false, "");
        }
    }

    private static final class ArrayPrefixState {
        private final boolean valid;
        private final boolean closed;
        private final List<String> items;

        private ArrayPrefixState(boolean valid, boolean closed, List<String> items) {
            this.valid = valid;
            this.closed = closed;
            this.items = items;
        }

        private static ArrayPrefixState invalid() {
            return new ArrayPrefixState(false, false, List.of());
        }
    }

    private static List<String> splitTopLevelArguments(String body) {
        List<String> arguments = new ArrayList<>();
        int start = 0;
        int braces = 0;
        int brackets = 0;
        int parentheses = 0;
        boolean quoted = false;
        boolean escaped = false;
        char quote = 0;

        for (int index = 0; index < body.length(); index++) {
            char current = body.charAt(index);
            if (quoted) {
                if (escaped) {
                    escaped = false;
                } else if (current == '\\') {
                    escaped = true;
                } else if (current == quote) {
                    quoted = false;
                }
                continue;
            }
            if (current == '\'' || current == '"') {
                quoted = true;
                quote = current;
            } else if (current == '{') {
                braces++;
            } else if (current == '}') {
                braces--;
            } else if (current == '[') {
                brackets++;
            } else if (current == ']') {
                brackets--;
            } else if (current == '(') {
                parentheses++;
            } else if (current == ')') {
                parentheses--;
            } else if (current == ','
                    && braces == 0 && brackets == 0 && parentheses == 0) {
                arguments.add(body.substring(start, index));
                start = index + 1;
            }
        }
        arguments.add(body.substring(start));
        return arguments;
    }

    private static boolean validCompleteArgumentList(String body) {
        if (body.isBlank()) {
            return true;
        }
        int start = 0;
        int braces = 0;
        int brackets = 0;
        int parentheses = 0;
        boolean quoted = false;
        boolean escaped = false;
        char quote = 0;

        for (int i = 0; i <= body.length(); i++) {
            char c = i == body.length() ? ',' : body.charAt(i);
            if (quoted) {
                if (escaped) {
                    escaped = false;
                } else if (c == '\\') {
                    escaped = true;
                } else if (c == quote) {
                    quoted = false;
                }
                continue;
            }
            if (c == '\'' || c == '"') {
                quoted = true;
                quote = c;
                continue;
            }
            if (c == '{') braces++;
            else if (c == '}') braces--;
            else if (c == '[') brackets++;
            else if (c == ']') brackets--;
            else if (c == '(') parentheses++;
            else if (c == ')') parentheses--;
            else if (c == ',' && braces == 0 && brackets == 0 && parentheses == 0) {
                if (!validArgument(body.substring(start, i))) {
                    return false;
                }
                start = i + 1;
            }
        }
        return !quoted && braces == 0 && brackets == 0 && parentheses == 0;
    }

    private static boolean validArgument(String argument) {
        String part = argument.trim();
        int equals = topLevelEquals(part);
        if (equals <= 0 || equals == part.length() - 1) {
            return false;
        }
        String name = part.substring(0, equals).trim();
        String value = part.substring(equals + 1).trim();
        return ARGUMENT_NAME.matcher(name).matches() && !value.isEmpty();
    }

    private static int topLevelEquals(String text) {
        int braces = 0;
        int brackets = 0;
        int parentheses = 0;
        boolean quoted = false;
        boolean escaped = false;
        char quote = 0;
        for (int i = 0; i < text.length(); i++) {
            char c = text.charAt(i);
            if (quoted) {
                if (escaped) escaped = false;
                else if (c == '\\') escaped = true;
                else if (c == quote) quoted = false;
                continue;
            }
            if (c == '\'' || c == '"') {
                quoted = true;
                quote = c;
            } else if (c == '{') braces++;
            else if (c == '}') braces--;
            else if (c == '[') brackets++;
            else if (c == ']') brackets--;
            else if (c == '(') parentheses++;
            else if (c == ')') parentheses--;
            else if (c == '=' && braces == 0 && brackets == 0 && parentheses == 0) {
                return i;
            }
        }
        return -1;
    }

    @Override
    public TextConstraint reset() {
        return new NativeToolCallConstraint(
                toolNames, argumentNamesByTool, argumentValuesByTool,
                parameterSchemasByTool);
    }

    @Override
    public String type() {
        return TYPE;
    }

    public List<String> getToolNames() {
        return toolNames;
    }
}
