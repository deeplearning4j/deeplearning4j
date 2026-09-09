/*
 * SPDX-License-Identifier: Apache-2.0
 * This program is made available under the Apache License, Version 2.0.
 * See https://www.apache.org/licenses/LICENSE-2.0 and the NOTICE file.
 */
package org.eclipse.deeplearning4j.llm.generation.constraint;

import org.eclipse.deeplearning4j.llm.generation.ToolSchemaValidator;
import org.eclipse.deeplearning4j.llm.tokenizer.ChatTemplate;
import org.nd4j.shade.jackson.databind.ObjectMapper;

import java.math.BigDecimal;
import java.util.ArrayList;
import java.util.Collection;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.regex.Pattern;

/**
 * Bounded Gemma wire codec shared by decoding and final tool parsing. Strings between
 * {@code <|"|>} sentinels are literal data, NOT JSON escapes. Only the scalar-prefix
 * schema check converts them to JSON, using Jackson to escape quotes, slashes and controls.
 * No model names or tool names are built into this grammar.
 *
 * <p>Bounds: 64 Ki UTF-16 characters, 64 nested collections, 32 adjacent calls.
 * Unquoted names use ASCII identifiers with optional dots/hyphens. Schema semantics
 * are those of {@link ToolSchemaValidator}, not arbitrary JSON Schema evaluation.</p>
 */
public final class GemmaToolCallCodec {
    public static final int MAX_CHARS = 65536;
    public static final int MAX_DEPTH = 64;
    public static final int MAX_CALLS = 32;
    private static final ObjectMapper JSON = new ObjectMapper();
    private static final Pattern NAME = Pattern.compile("[A-Za-z_][A-Za-z0-9_.-]*");
    private static final Pattern NUMBER = Pattern.compile(
            "-?(0|[1-9][0-9]*)(\\.[0-9]+)?([eE][+-]?[0-9]+)?");
    private static final Pattern NUMBER_PREFIX = Pattern.compile(
            "-?((0|[1-9][0-9]*)(\\.[0-9]*|(?:\\.[0-9]+)?[eE][+-]?[0-9]*)?)?");

    private GemmaToolCallCodec() { }

    public static boolean validName(String name) {
        return name != null && NAME.matcher(name).matches();
    }

    public static final class Result {
        public final boolean valid;
        public final boolean complete;
        public final List<ChatTemplate.ToolCall> calls;
        public final String error;

        private Result(boolean valid, boolean complete, List<ChatTemplate.ToolCall> calls,
                       String error) {
            this.valid = valid;
            this.complete = complete;
            this.calls = complete ? List.copyOf(calls) : List.of();
            this.error = error;
        }
    }

    /** Scan the entire candidate, including token pieces straddling any delimiter. */
    public static Result scan(String text, Map<String, ChatTemplate.Tool> tools) {
        if (text == null || text.length() > MAX_CHARS) {
            return new Result(false, false, List.of(), "Gemma tool text exceeds codec bounds");
        }
        Reader reader = new Reader(text);
        List<ChatTemplate.ToolCall> calls = new ArrayList<>();
        try {
            reader.space();
            do {
                if (calls.size() >= MAX_CALLS) throw new Invalid("too many Gemma calls");
                reader.literal(ChatTemplate.GEMMA_TOOL_CALL_START + "call:");
                String name = reader.name(tools.keySet());
                ChatTemplate.Tool tool = tools.get(name);
                if (tool == null) throw new Invalid("undeclared tool " + name);
                reader.space();
                if (reader.peek() != '{') throw new Invalid("Gemma arguments must be an object");
                Map<String, Object> schema = new LinkedHashMap<>(tool.getParameters());
                schema.putIfAbsent("type", "object");
                Object value = reader.value(schema, 0);
                @SuppressWarnings("unchecked")
                Map<String, Object> arguments = (Map<String, Object>) value;
                List<String> errors = ToolSchemaValidator.validateArguments(tool, arguments);
                if (!errors.isEmpty()) throw new Invalid(String.join("; ", errors));
                reader.space();
                reader.literal(ChatTemplate.GEMMA_TOOL_CALL_END);
                calls.add(new ChatTemplate.ToolCall(null, name, arguments));
                reader.space();
            } while (reader.position < text.length());
            return new Result(true, true, calls, "");
        } catch (Incomplete ignored) {
            return new Result(text.length() < MAX_CHARS, false, List.of(),
                    "incomplete Gemma tool-call envelope");
        } catch (Invalid invalid) {
            return new Result(false, false, List.of(), invalid.getMessage());
        }
    }

    /** Encode literal protocol values. Schema-type capitalization belongs to the template renderer. */
    public static String encode(Object value) {
        return encode(value, 0);
    }

    private static String encode(Object value, int depth) {
        if (depth > MAX_DEPTH) throw new IllegalArgumentException("Gemma value is too deeply nested");
        if (value instanceof String) {
            String string = (String) value;
            if (hasControlMarker(string)) {
                throw new IllegalArgumentException("Gemma literal string contains a reserved control marker");
            }
            return ChatTemplate.GEMMA_STRING + string + ChatTemplate.GEMMA_STRING;
        }
        if (value == null || value instanceof Boolean || value instanceof Number) {
            String scalar = String.valueOf(value);
            if (value instanceof Number && !NUMBER.matcher(scalar).matches()) {
                throw new IllegalArgumentException("Gemma number must be finite JSON numeric data");
            }
            return scalar;
        }
        if (value instanceof Map<?, ?>) {
            List<String> members = new ArrayList<>();
            ((Map<?, ?>) value).forEach((key, item) -> {
                if (!(key instanceof String) || !validName((String) key)) {
                    throw new IllegalArgumentException("Unsupported Gemma unquoted key: " + key);
                }
                members.add(key + ":" + encode(item, depth + 1));
            });
            return "{" + String.join(",", members) + "}";
        }
        if (value instanceof Collection<?>) {
            List<String> items = new ArrayList<>();
            for (Object item : (Collection<?>) value) items.add(encode(item, depth + 1));
            return "[" + String.join(",", items) + "]";
        }
        throw new IllegalArgumentException("Unsupported Gemma value: " + value.getClass());
    }

    private static boolean hasControlMarker(String value) {
        return value.contains("<|") || value.contains("<tool|")
                || value.contains("<tool_call|") || value.contains("<channel|");
    }

    private static Map<String, Object> schema(Object value) {
        Map<String, Object> result = new LinkedHashMap<>();
        if (value instanceof Map<?, ?>) {
            ((Map<?, ?>) value).forEach((key, item) -> result.put(String.valueOf(key), item));
        }
        return result;
    }

    private static final class Reader {
        private final String text;
        private int position;

        private Reader(String text) { this.text = text; }

        private char peek() {
            if (position == text.length()) throw Incomplete.INSTANCE;
            return text.charAt(position);
        }

        private void space() {
            while (position < text.length() && " \t\r\n".indexOf(text.charAt(position)) >= 0) position++;
        }

        private void literal(String expected) {
            for (int index = 0; index < expected.length(); index++) {
                if (peek() != expected.charAt(index)) throw new Invalid("invalid Gemma delimiter");
                position++;
            }
        }

        private String name(Collection<String> allowed) {
            int start = position;
            char first = peek();
            if (!(first >= 'A' && first <= 'Z' || first >= 'a' && first <= 'z' || first == '_')) {
                throw new Invalid("invalid Gemma identifier");
            }
            position++;
            while (position < text.length()) {
                char c = text.charAt(position);
                if (!(c >= 'A' && c <= 'Z' || c >= 'a' && c <= 'z'
                        || c >= '0' && c <= '9' || c == '_' || c == '.' || c == '-')) break;
                position++;
            }
            String value = text.substring(start, position);
            if (position == text.length()) {
                if (allowed != null && allowed.stream().noneMatch(n -> n.startsWith(value))) {
                    throw new Invalid("undeclared Gemma identifier " + value);
                }
                throw Incomplete.INSTANCE;
            }
            if (allowed != null && !allowed.contains(value)) {
                throw new Invalid("undeclared or duplicate Gemma identifier " + value);
            }
            return value;
        }

        private Object value(Map<String, Object> schema, int depth) {
            if (depth > MAX_DEPTH) throw new Invalid("Gemma value is too deeply nested");
            space();
            char c = peek();
            String type = schema.get("type") instanceof String ? (String) schema.get("type") : "";
            Object value;
            if (c == '{') {
                requireType(type, "object");
                value = object(schema, depth + 1);
            } else if (c == '[') {
                requireType(type, "array");
                value = array(schema, depth + 1);
            } else if (c == '<') {
                requireType(type, "string");
                value = string(schema);
            } else {
                int start = position;
                while (position < text.length() && ",]} \t\r\n".indexOf(text.charAt(position)) < 0) position++;
                String token = text.substring(start, position);
                boolean numeric = !token.isEmpty() && (token.charAt(0) == '-' || Character.isDigit(token.charAt(0)));
                if (numeric) {
                    if (!type.isEmpty() && !type.equals("number") && !type.equals("integer")) {
                        throw new Invalid("wrong Gemma scalar type");
                    }
                    if (!NUMBER_PREFIX.matcher(token).matches()) throw new Invalid("invalid Gemma number");
                } else if (!("true".startsWith(token) || "false".startsWith(token) || "null".startsWith(token))) {
                    throw new Invalid("invalid Gemma scalar");
                } else {
                    requireType(type, "null".startsWith(token) ? "null" : "boolean");
                }
                if (position == text.length()) {
                    // A numeric prefix is not a finished value: 1 may become 10 or
                    // 1e-3. Apply numeric bounds/enum only once a delimiter commits it.
                    throw Incomplete.INSTANCE;
                }
                if (numeric && NUMBER.matcher(token).matches()) {
                    try { value = new BigDecimal(token); }
                    catch (NumberFormatException e) { throw new Invalid("invalid Gemma number"); }
                } else if (token.equals("true") || token.equals("false")) {
                    value = Boolean.valueOf(token);
                } else if (token.equals("null")) {
                    value = null;
                } else throw new Invalid("incomplete Gemma scalar");
            }
            if (!ToolSchemaValidator.isValidValue(value, schema)) {
                throw new Invalid("Gemma value violates its schema");
            }
            return value;
        }

        private Map<String, Object> object(Map<String, Object> schema, int depth) {
            literal("{");
            space();
            Map<String, Object> result = new LinkedHashMap<>();
            Map<String, Object> properties = schema(schema.get("properties"));
            if (peek() == '}') { position++; return result; }
            while (true) {
                Collection<String> allowed = null;
                if (Boolean.FALSE.equals(schema.get("additionalProperties"))) {
                    List<String> names = new ArrayList<>(properties.keySet());
                    names.removeAll(result.keySet());
                    allowed = names;
                }
                String key = name(allowed);
                if (result.containsKey(key)) throw new Invalid("duplicate Gemma key " + key);
                space();
                literal(":");
                Map<String, Object> child = schema(properties.containsKey(key)
                        ? properties.get(key) : schema.get("additionalProperties"));
                result.put(key, value(child, depth));
                space();
                char delimiter = peek();
                if (delimiter == '}') { position++; return result; }
                literal(",");
                space();
            }
        }

        private List<Object> array(Map<String, Object> schema, int depth) {
            literal("[");
            space();
            List<Object> result = new ArrayList<>();
            if (peek() == ']') { position++; return result; }
            List<?> prefix = schema.get("prefixItems") instanceof Collection<?>
                    ? new ArrayList<>((Collection<?>) schema.get("prefixItems")) : List.of();
            while (true) {
                Object max = schema.get("maxItems");
                if (max instanceof Number && result.size() >= ((Number) max).intValue()) {
                    throw new Invalid("too many Gemma array items");
                }
                Object itemSchema = result.size() < prefix.size()
                        ? prefix.get(result.size()) : schema.get("items");
                if (Boolean.FALSE.equals(itemSchema)) throw new Invalid("Gemma array item is not allowed");
                Object item = value(schema(itemSchema), depth);
                if (Boolean.TRUE.equals(schema.get("uniqueItems")) && result.contains(item)) {
                    throw new Invalid("duplicate Gemma array item");
                }
                result.add(item);
                space();
                char delimiter = peek();
                if (delimiter == ']') { position++; return result; }
                literal(",");
                space();
            }
        }

        private String string(Map<String, Object> schema) {
            literal(ChatTemplate.GEMMA_STRING);
            int start = position;
            while (position < text.length()) {
                if (text.startsWith(ChatTemplate.GEMMA_STRING, position)) {
                    String value = text.substring(start, position);
                    position += ChatTemplate.GEMMA_STRING.length();
                    return value;
                }
                // A partial closing sentinel is framing, not part of the string prefix.
                if (text.length() - position < ChatTemplate.GEMMA_STRING.length()
                        && ChatTemplate.GEMMA_STRING.startsWith(text.substring(position))) break;
                if (text.startsWith("<|", position) || text.startsWith("<tool|", position)
                        || text.startsWith("<tool_call|", position) || text.startsWith("<channel|", position)) {
                    throw new Invalid("unexpected control marker inside Gemma string");
                }
                position++;
            }
            String value = text.substring(start, position);
            try {
                String json = JSON.writeValueAsString(value);
                if (!NativeToolCallConstraint.validValuePrefix(json.substring(0, json.length() - 1), schema)) {
                    throw new Invalid("Gemma string prefix violates schema");
                }
            } catch (java.io.IOException e) {
                throw new Invalid("invalid Gemma literal string");
            }
            throw Incomplete.INSTANCE;
        }

        private void requireType(String actual, String expected) {
            if (!actual.isEmpty() && !actual.equals(expected)) throw new Invalid("wrong Gemma value type");
        }
    }

    private static final class Incomplete extends RuntimeException {
        private static final Incomplete INSTANCE = new Incomplete();
        private Incomplete() { super(null, null, false, false); }
    }

    private static final class Invalid extends RuntimeException {
        private Invalid(String message) { super(message, null, false, false); }
    }
}
