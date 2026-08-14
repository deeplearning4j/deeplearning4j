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
package org.eclipse.deeplearning4j.llm.generation;

import org.nd4j.shade.jackson.databind.DeserializationFeature;
import org.nd4j.shade.jackson.databind.JsonNode;
import org.nd4j.shade.jackson.databind.ObjectMapper;
import org.eclipse.deeplearning4j.llm.tokenizer.ChatTemplate;

import java.util.*;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * Template-aware parser for model-generated tool-call protocols.
 *
 * <p>The parser is deliberately schema-aware and fail-closed. It accepts complete
 * JSON envelopes and explicitly framed native envelopes, but does not turn arbitrary
 * JSON-looking prose or function declarations into executable calls.</p>
 */
public final class ToolCallParser {
    private static final ObjectMapper MAPPER = new ObjectMapper()
            .enable(DeserializationFeature.FAIL_ON_TRAILING_TOKENS);
    private static final Pattern PYTHON_CALL = Pattern.compile(
            "(?s)([A-Za-z_][A-Za-z0-9_.-]*)\\s*\\((.*)\\)");
    private static final Pattern ARGUMENT_NAME = Pattern.compile("[A-Za-z_][A-Za-z0-9_]*");
    private static final Pattern XML_FUNCTION = Pattern.compile(
            "(?s)<tool_call>\\s*<function=([A-Za-z_][A-Za-z0-9_.-]*)>\\s*"
                    + "(.*?)\\s*</function>\\s*</tool_call>");
    private static final Pattern XML_PARAMETER = Pattern.compile(
            "(?s)\\s*<parameter=([A-Za-z_][A-Za-z0-9_]*)>\\s*"
                    + "(.*?)\\s*</parameter>");

    private ToolCallParser() {
    }

    public enum Protocol {
        CONTENT_ONLY,
        JSON,
        LFM_NATIVE,
        XML
    }

    public static final class ParseResult {
        private final String rawText;
        private final String content;
        private final List<ChatTemplate.ToolCall> toolCalls;
        private final List<String> errors;

        public ParseResult(String rawText, String content,
                           List<ChatTemplate.ToolCall> toolCalls,
                           List<String> errors) {
            this.rawText = rawText == null ? "" : rawText;
            this.content = content == null ? "" : content;
            this.toolCalls = toolCalls == null ? List.of() : List.copyOf(toolCalls);
            this.errors = errors == null ? List.of() : List.copyOf(errors);
        }

        public String getRawText() { return rawText; }
        public String getContent() { return content; }
        public List<ChatTemplate.ToolCall> getToolCalls() { return toolCalls; }
        public List<ChatTemplate.ToolCall> getCalls() { return toolCalls; }
        public List<String> getErrors() { return errors; }
        public List<String> getParseErrors() { return errors; }
        public boolean hasToolCalls() { return !toolCalls.isEmpty(); }
        public boolean isClean() { return errors.isEmpty(); }
    }

    public static ParseResult parse(String rawText) {
        return parse(rawText, List.of(), Protocol.JSON);
    }

    public static ParseResult parse(String rawText, List<ChatTemplate.Tool> tools) {
        return parse(rawText, tools, Protocol.JSON);
    }

    public static ParseResult parse(String rawText, List<ChatTemplate.Tool> tools,
                                    ChatTemplate.ToolCallFormat format) {
        return parse(rawText, tools, format, ChatTemplate.ToolChoice.AUTO);
    }

    public static ParseResult parse(String rawText, List<ChatTemplate.Tool> tools,
                                    ChatTemplate.ToolCallFormat format,
                                    ChatTemplate.ToolChoice toolChoice) {
        if (format == null) {
            throw new IllegalArgumentException("Tool-call format must be explicit");
        }
        Protocol protocol;
        switch (format) {
            case NATIVE:
                protocol = Protocol.LFM_NATIVE;
                break;
            case XML:
                protocol = Protocol.XML;
                break;
            case JSON:
                protocol = Protocol.JSON;
                break;
            default:
                throw new IllegalArgumentException("Unsupported tool-call format: " + format);
        }
        return parse(rawText, tools, protocol, toolChoice);
    }

    public static ParseResult parse(String rawText, List<ChatTemplate.Tool> tools,
                                    Protocol protocol) {
        return parse(rawText, tools, protocol, ChatTemplate.ToolChoice.AUTO);
    }

    public static ParseResult parse(String rawText, List<ChatTemplate.Tool> tools,
                                    Protocol protocol, ChatTemplate.ToolChoice toolChoice) {
        String raw = rawText == null ? "" : rawText;
        List<ChatTemplate.Tool> declared = tools == null ? List.of() : tools;
        Map<String, ChatTemplate.Tool> byName = new LinkedHashMap<>();
        for (ChatTemplate.Tool tool : declared) {
            if (tool != null && tool.getName() != null && !tool.getName().isBlank()) {
                byName.put(tool.getName(), tool);
            }
        }
        if (raw.isBlank()) return new ParseResult(raw, "", List.of(), List.of());
        if (protocol == null) {
            throw new IllegalArgumentException("Tool-call protocol must be explicit");
        }
        boolean required = toolChoice == ChatTemplate.ToolChoice.REQUIRED;
        switch (protocol) {
            case CONTENT_ONLY:
                return new ParseResult(raw, raw.trim(), List.of(), List.of());
            case JSON:
                return parseJsonEnvelope(raw, byName, required);
            case LFM_NATIVE:
                if (containsNativeToolMarker(raw)) {
                    return parseNativeEnvelope(raw, byName);
                }
                if (required) {
                    return new ParseResult(raw, raw.trim(), List.of(),
                            List.of("required native tool call was missing or invalid"));
                }
                return new ParseResult(raw, raw.trim(), List.of(), List.of());
            case XML:
                if (raw.contains(ChatTemplate.XML_TOOL_CALL_START)
                        || raw.contains(ChatTemplate.XML_TOOL_CALL_END)) {
                    return parseXmlEnvelope(raw, byName);
                }
                if (required) {
                    return new ParseResult(raw, raw.trim(), List.of(),
                            List.of("required XML tool call was missing or invalid"));
                }
                return new ParseResult(raw, raw.trim(), List.of(), List.of());
            default:
                throw new IllegalArgumentException("Unsupported tool-call protocol: " + protocol);
        }
    }

    private static ParseResult parseJsonEnvelope(
            String raw, Map<String, ChatTemplate.Tool> declared, boolean required) {
        List<String> errors = new ArrayList<>();
        List<ChatTemplate.ToolCall> calls = new ArrayList<>();
        JsonNode whole = readJson(raw.trim());
        if (whole == null) {
            String trimmed = raw.trim();
            if (required || trimmed.startsWith("{") || trimmed.startsWith("[")) {
                errors.add("expected one complete JSON tool-call envelope");
            }
            return new ParseResult(raw, raw.trim(), List.of(), errors);
        }
        extractJsonCalls(whole, declared, calls, errors);
        if (calls.isEmpty()) {
            if (errors.isEmpty() && required) {
                errors.add("required JSON tool call was missing or invalid");
            }
            return new ParseResult(raw, raw.trim(), List.of(), errors);
        }
        return new ParseResult(raw, "", unique(calls), errors);
    }

    private static boolean containsNativeToolMarker(String raw) {
        return raw.contains(ChatTemplate.NATIVE_TOOL_CALL_START)
                || raw.contains(ChatTemplate.NATIVE_TOOL_CALL_END);
    }

    private static ParseResult parseNativeEnvelope(
            String raw, Map<String, ChatTemplate.Tool> declared) {
        String trimmed = raw.trim();
        List<String> errors = new ArrayList<>();
        if (!trimmed.startsWith(ChatTemplate.NATIVE_TOOL_CALL_START)
                || !trimmed.endsWith(ChatTemplate.NATIVE_TOOL_CALL_END)) {
            errors.add("incomplete native tool-call envelope");
            return new ParseResult(raw, raw.trim(), List.of(), errors);
        }

        String payload = trimmed.substring(
                ChatTemplate.NATIVE_TOOL_CALL_START.length(),
                trimmed.length() - ChatTemplate.NATIVE_TOOL_CALL_END.length()).trim();
        if (payload.length() < 2 || payload.charAt(0) != '['
                || payload.charAt(payload.length() - 1) != ']') {
            errors.add("invalid native tool-call envelope");
            return new ParseResult(raw, raw.trim(), List.of(), errors);
        }

        String invocation = payload.substring(1, payload.length() - 1).trim();
        Matcher call = PYTHON_CALL.matcher(invocation);
        if (!call.matches()) {
            errors.add("native tool-call envelope did not contain exactly one complete call");
            return new ParseResult(raw, raw.trim(), List.of(), errors);
        }

        String name = call.group(1);
        if (!declared.containsKey(name)) {
            errors.add("undeclared tool " + name);
            return new ParseResult(raw, raw.trim(), List.of(), errors);
        }
        Map<String, Object> arguments = parsePythonArguments(call.group(2));
        if (arguments == null) {
            errors.add("invalid arguments for tool " + name);
            return new ParseResult(raw, raw.trim(), List.of(), errors);
        }

        List<String> schemaErrors =
                ToolSchemaValidator.validateArguments(declared.get(name), arguments);
        if (!schemaErrors.isEmpty()) {
            errors.add("arguments for tool " + name + " violate its schema: "
                    + String.join("; ", schemaErrors));
            return new ParseResult(raw, raw.trim(), List.of(), errors);
        }

        ChatTemplate.ToolCall parsed = new ChatTemplate.ToolCall(null, name, arguments);
        return new ParseResult(raw, "", List.of(parsed), List.of());
    }

    private static ParseResult parseXmlEnvelope(
            String raw, Map<String, ChatTemplate.Tool> declared) {
        String trimmed = raw.trim();
        int envelopeStart = trimmed.indexOf(ChatTemplate.XML_TOOL_CALL_START);
        if (envelopeStart < 0) {
            return new ParseResult(raw, trimmed, List.of(),
                    List.of("required XML tool call was missing or invalid"));
        }
        String content = trimmed.substring(0, envelopeStart).trim();
        String envelope = trimmed.substring(envelopeStart);
        Matcher function = XML_FUNCTION.matcher(envelope);
        if (!function.matches()) {
            return new ParseResult(raw, content, List.of(),
                    List.of("incomplete XML tool-call envelope"));
        }

        String name = function.group(1);
        ChatTemplate.Tool tool = declared.get(name);
        if (tool == null) {
            return new ParseResult(raw, content, List.of(),
                    List.of("undeclared tool " + name));
        }

        String parameters = function.group(2);
        Matcher parameter = XML_PARAMETER.matcher(parameters);
        Map<String, Object> arguments = new LinkedHashMap<>();
        int cursor = 0;
        while (cursor < parameters.length()) {
            parameter.region(cursor, parameters.length());
            if (!parameter.lookingAt()) {
                if (parameters.substring(cursor).isBlank()) {
                    cursor = parameters.length();
                    break;
                }
                return new ParseResult(raw, content, List.of(),
                        List.of("invalid XML parameters for tool " + name));
            }
            String argumentName = parameter.group(1);
            if (arguments.containsKey(argumentName)) {
                return new ParseResult(raw, content, List.of(),
                        List.of("duplicate XML parameter " + argumentName
                                + " for tool " + name));
            }
            arguments.put(argumentName, parseScalar(parameter.group(2).trim()));
            cursor = parameter.end();
        }

        List<String> schemaErrors = ToolSchemaValidator.validateArguments(tool, arguments);
        if (!schemaErrors.isEmpty()) {
            return new ParseResult(raw, content, List.of(),
                    List.of("arguments for tool " + name + " violate its schema: "
                            + String.join("; ", schemaErrors)));
        }
        return new ParseResult(raw, content,
                List.of(new ChatTemplate.ToolCall(null, name, arguments)), List.of());
    }

    private static void extractJsonCalls(JsonNode root, Map<String, ChatTemplate.Tool> declared,
                                         List<ChatTemplate.ToolCall> out, List<String> errors) {
        if (root == null || !root.isObject()) return;

        JsonNode toolCalls = root.get("tool_calls");
        if (toolCalls != null && toolCalls.isArray()) {
            for (JsonNode call : toolCalls) extractOpenAiCall(call, declared, out, errors);
            return;
        }
        JsonNode tool = root.has("tool") ? root.get("tool") : root.get("toolName");
        if (tool != null && tool.isTextual()) {
            JsonNode args = root.has("args") ? root.get("args") : root.get("arguments");
            if (args != null && args.isObject()) add(tool.asText(), root.get("id"), args, declared, out, errors);
            return;
        }
        JsonNode function = root.get("function");
        if (function != null && function.isObject()) {
            JsonNode name = function.get("name");
            JsonNode args = function.has("arguments") ? function.get("arguments") : function.get("args");
            if (name != null && name.isTextual() && args != null && !isSchemaDeclaration(function)) {
                add(name.asText(), root.get("id"), parseArgumentsNode(args), declared, out, errors);
            }
            return;
        }
        JsonNode name = root.get("name");
        JsonNode args = root.has("arguments") ? root.get("arguments") : root.get("args");
        if (name != null && name.isTextual() && args != null && args.isObject()) {
            add(name.asText(), root.get("id"), args, declared, out, errors);
        }
    }

    private static void extractOpenAiCall(JsonNode call, Map<String, ChatTemplate.Tool> declared,
                                          List<ChatTemplate.ToolCall> out, List<String> errors) {
        if (call == null || !call.isObject()) return;
        JsonNode fn = call.get("function");
        if (fn != null && fn.isObject()) {
            JsonNode name = fn.get("name");
            JsonNode args = fn.has("arguments") ? fn.get("arguments") : fn.get("args");
            if (name != null && name.isTextual() && args != null) {
                add(name.asText(), call.get("id"), parseArgumentsNode(args), declared, out, errors);
            }
        }
    }

    private static boolean isSchemaDeclaration(JsonNode function) {
        JsonNode parameters = function.get("parameters");
        return function.has("type") && function.get("type").asText().equals("function")
                || (parameters != null && parameters.isObject()
                && parameters.has("properties") && !function.has("arguments")
                && !function.has("args"));
    }

    private static JsonNode parseArgumentsNode(JsonNode node) {
        if (node == null) return null;
        if (node.isTextual()) return readJson(node.asText());
        return node;
    }

    private static void add(String name, JsonNode id, JsonNode args,
                            Map<String, ChatTemplate.Tool> declared,
                            List<ChatTemplate.ToolCall> out, List<String> errors) {
        if (name == null || name.isBlank() || args == null || !args.isObject()) {
            errors.add("invalid arguments for tool " + name);
            return;
        }
        if (!declared.containsKey(name)) {
            errors.add("undeclared tool " + name);
            return;
        }
        Map<String, Object> values = MAPPER.convertValue(args, Map.class);
        List<String> schemaErrors =
                ToolSchemaValidator.validateArguments(declared.get(name), values);
        if (!schemaErrors.isEmpty()) {
            errors.add("arguments for tool " + name + " violate its schema: "
                    + String.join("; ", schemaErrors));
            return;
        }
        out.add(new ChatTemplate.ToolCall(id == null ? null : id.asText(), name, values));
    }

    private static Map<String, Object> parsePythonArguments(String text) {
        if (text == null || text.isBlank()) return new LinkedHashMap<>();
        List<String> parts = splitArguments(text);
        if (parts == null) return null;
        Map<String, Object> result = new LinkedHashMap<>();
        for (String part : parts) {
            int eq = part.indexOf('=');
            if (eq <= 0) return null;
            String key = part.substring(0, eq).trim();
            String value = part.substring(eq + 1).trim();
            if (!ARGUMENT_NAME.matcher(key).matches() || value.isEmpty()) return null;
            if ((value.startsWith("\"") || value.endsWith("\""))
                    && !(value.startsWith("\"") && value.endsWith("\""))) {
                return null;
            }
            if ((value.startsWith("'") || value.endsWith("'"))
                    && !(value.startsWith("'") && value.endsWith("'"))) {
                return null;
            }
            result.put(key, parseScalar(value));
        }
        return result;
    }

    private static List<String> splitArguments(String text) {
        List<String> result = new ArrayList<>();
        int depth = 0; boolean quoted = false; char quote = 0; int start = 0;
        for (int i = 0; i < text.length(); i++) {
            char c = text.charAt(i);
            if (quoted) {
                if (c == quote && (i == 0 || text.charAt(i - 1) != '\\')) quoted = false;
            } else if (c == '\'' || c == '"') {
                quoted = true; quote = c;
            } else if (c == '{' || c == '[' || c == '(') {
                depth++;
            } else if (c == '}' || c == ']' || c == ')') {
                depth--;
                if (depth < 0) return null;
            } else if (c == ',' && depth == 0) {
                result.add(text.substring(start, i)); start = i + 1;
            }
        }
        if (quoted || depth != 0) return null;
        result.add(text.substring(start));
        return result;
    }

    private static Object parseScalar(String value) {
        if (value.startsWith("\"") && value.endsWith("\"")) {
            JsonNode quoted = readJson(value);
            if (quoted != null && quoted.isTextual()) {
                return quoted.asText();
            }
            return value.substring(1, value.length() - 1);
        }
        if (value.startsWith("'") && value.endsWith("'")) {
            return unescapeSingleQuoted(value.substring(1, value.length() - 1));
        }
        JsonNode json = readJson(value);
        if (json != null) return MAPPER.convertValue(json, Object.class);
        return value;
    }

    private static String unescapeSingleQuoted(String value) {
        StringBuilder result = new StringBuilder(value.length());
        for (int i = 0; i < value.length(); i++) {
            char current = value.charAt(i);
            if (current != '\\' || i + 1 >= value.length()) {
                result.append(current);
                continue;
            }
            char escaped = value.charAt(++i);
            switch (escaped) {
                case 'n':
                    result.append('\n');
                    break;
                case 'r':
                    result.append('\r');
                    break;
                case 't':
                    result.append('\t');
                    break;
                case '\\':
                    result.append('\\');
                    break;
                case '\'':
                    result.append('\'');
                    break;
                case '\"':
                    result.append('\"');
                    break;
                default:
                    result.append(escaped);
                    break;
            }
        }
        return result.toString();
    }

    private static JsonNode readJson(String value) {
        try { return MAPPER.readTree(value); }
        catch (Exception ignored) { return null; }
    }

    private static List<ChatTemplate.ToolCall> unique(List<ChatTemplate.ToolCall> calls) {
        LinkedHashMap<String, ChatTemplate.ToolCall> result = new LinkedHashMap<>();
        for (ChatTemplate.ToolCall call : calls) {
            String key = String.valueOf(call.getId()) + "|" + call.getName() + "|" + call.getArguments();
            result.putIfAbsent(key, call);
        }
        return List.copyOf(result.values());
    }
}
