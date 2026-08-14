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

import org.eclipse.deeplearning4j.llm.tokenizer.ChatTemplate;

import java.lang.reflect.Array;
import java.math.BigDecimal;
import java.util.ArrayList;
import java.util.Collection;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Set;
import java.util.regex.Pattern;
import java.util.regex.PatternSyntaxException;

/**
 * Validates parsed function arguments against the JSON schema supplied with a
 * {@link ChatTemplate.Tool}. The validator intentionally lives at the shared
 * model/tool boundary so every caller gets the same fail-closed behavior.
 *
 * <p>The portable subset covers the schema vocabulary emitted by the chat and
 * crawl stacks: object properties/required/additionalProperties, arrays and
 * item schemas (including positional prefixItems), scalar types, enum/const,
 * size bounds, uniqueness, patterns,
 * and numeric bounds. Unsupported annotation keywords are ignored.</p>
 */
public final class ToolSchemaValidator {

    private ToolSchemaValidator() {
    }

    /**
     * Validates one parsed tool invocation.
     *
     * @return an empty list when valid, otherwise stable path-oriented errors
     */
    public static List<String> validateArguments(
            ChatTemplate.Tool tool, Map<String, Object> arguments) {
        if (tool == null) {
            return List.of("tool declaration is missing");
        }
        List<String> errors = new ArrayList<>();
        validateValue(arguments == null ? Map.of() : arguments,
                tool.getParameters(), "$", errors);
        return List.copyOf(errors);
    }

    /**
     * Validates a value against a schema without allocating diagnostic text for
     * the common success case.
     */
    public static boolean isValidValue(Object value, Map<String, Object> schema) {
        List<String> errors = new ArrayList<>();
        validateValue(value, schema, "$", errors);
        return errors.isEmpty();
    }

    private static void validateValue(
            Object value,
            Map<String, Object> schema,
            String path,
            List<String> errors) {
        if (schema == null || schema.isEmpty()) {
            return;
        }
        if (!matchesConstAndEnum(value, schema)) {
            errors.add(path + " is not one of the allowed values");
            return;
        }

        String type = schema.get("type") instanceof String
                ? (String) schema.get("type") : null;
        if (type == null || type.isBlank()) {
            return;
        }
        switch (type) {
            case "object":
                validateObject(value, schema, path, errors);
                break;
            case "array":
                validateArray(value, schema, path, errors);
                break;
            case "string":
                validateString(value, schema, path, errors);
                break;
            case "integer":
                validateNumber(value, schema, path, errors, true);
                break;
            case "number":
                validateNumber(value, schema, path, errors, false);
                break;
            case "boolean":
                if (!(value instanceof Boolean)) {
                    errors.add(path + " must be a boolean");
                }
                break;
            case "null":
                if (value != null) {
                    errors.add(path + " must be null");
                }
                break;
            default:
                errors.add(path + " uses unsupported schema type " + type);
                break;
        }
    }

    private static void validateObject(
            Object value,
            Map<String, Object> schema,
            String path,
            List<String> errors) {
        if (!(value instanceof Map<?, ?>)) {
            errors.add(path + " must be an object");
            return;
        }
        Map<?, ?> object = (Map<?, ?>) value;
        Object requiredObject = schema.get("required");
        if (requiredObject instanceof Collection<?>) {
            for (Object required : (Collection<?>) requiredObject) {
                if (required instanceof String && !object.containsKey(required)) {
                    errors.add(path + "." + required + " is required");
                }
            }
        }

        Map<?, ?> properties = schema.get("properties") instanceof Map<?, ?>
                ? (Map<?, ?>) schema.get("properties") : Map.of();
        for (Map.Entry<?, ?> entry : object.entrySet()) {
            String key = String.valueOf(entry.getKey());
            Object propertySchema = properties.get(key);
            if (propertySchema instanceof Map<?, ?>) {
                validateValue(entry.getValue(), stringKeyMap((Map<?, ?>) propertySchema),
                        path + "." + key, errors);
                continue;
            }
            Object additional = schema.get("additionalProperties");
            if (Boolean.FALSE.equals(additional)) {
                errors.add(path + "." + key + " is not declared");
            } else if (additional instanceof Map<?, ?>) {
                validateValue(entry.getValue(), stringKeyMap((Map<?, ?>) additional),
                        path + "." + key, errors);
            }
        }
    }

    private static void validateArray(
            Object value,
            Map<String, Object> schema,
            String path,
            List<String> errors) {
        List<?> values = asList(value);
        if (values == null) {
            errors.add(path + " must be an array");
            return;
        }
        int size = values.size();
        Integer minimum = integerKeyword(schema, "minItems");
        Integer maximum = integerKeyword(schema, "maxItems");
        if (minimum != null && size < minimum) {
            errors.add(path + " must contain at least " + minimum + " items");
        }
        if (maximum != null && size > maximum) {
            errors.add(path + " must contain at most " + maximum + " items");
        }
        if (Boolean.TRUE.equals(schema.get("uniqueItems"))) {
            Set<Object> unique = new HashSet<>();
            for (Object item : values) {
                if (!unique.add(item)) {
                    errors.add(path + " must contain unique items");
                    break;
                }
            }
        }

        int prefixCount = 0;
        if (schema.get("prefixItems") instanceof Collection<?>) {
            Collection<?> prefixItems = (Collection<?>) schema.get("prefixItems");
            for (Object prefixItem : prefixItems) {
                if (prefixCount >= values.size()) {
                    break;
                }
                Map<String, Object> prefixSchema = prefixItem instanceof Map<?, ?>
                        ? stringKeyMap((Map<?, ?>) prefixItem) : Map.of();
                validateValue(values.get(prefixCount), prefixSchema,
                        path + "[" + prefixCount + "]", errors);
                prefixCount++;
            }
        }

        if (values.size() <= prefixCount) {
            return;
        }
        Object remainingItems = schema.get("items");
        if (Boolean.FALSE.equals(remainingItems)) {
            errors.add(path + " must not contain items after index " + (prefixCount - 1));
        } else if (remainingItems instanceof Map<?, ?>) {
            Map<String, Object> itemSchema =
                    stringKeyMap((Map<?, ?>) remainingItems);
            for (int index = prefixCount; index < values.size(); index++) {
                validateValue(values.get(index), itemSchema,
                        path + "[" + index + "]", errors);
            }
        }
    }

    private static void validateString(
            Object value,
            Map<String, Object> schema,
            String path,
            List<String> errors) {
        if (!(value instanceof String)) {
            errors.add(path + " must be a string");
            return;
        }
        String text = (String) value;
        int length = text.codePointCount(0, text.length());
        Integer minimum = integerKeyword(schema, "minLength");
        Integer maximum = integerKeyword(schema, "maxLength");
        if (minimum != null && length < minimum) {
            errors.add(path + " must contain at least " + minimum + " characters");
        }
        if (maximum != null && length > maximum) {
            errors.add(path + " must contain at most " + maximum + " characters");
        }
        if (schema.get("pattern") instanceof String) {
            try {
                if (!Pattern.compile((String) schema.get("pattern")).matcher(text).find()) {
                    errors.add(path + " does not match the required pattern");
                }
            } catch (PatternSyntaxException invalidSchema) {
                errors.add(path + " has an invalid schema pattern");
            }
        }
    }

    private static void validateNumber(
            Object value,
            Map<String, Object> schema,
            String path,
            List<String> errors,
            boolean integerOnly) {
        if (!(value instanceof Number)
                || integerOnly && !isIntegral((Number) value)) {
            errors.add(path + (integerOnly ? " must be an integer" : " must be a number"));
            return;
        }
        BigDecimal number = decimal((Number) value);
        compareBound(number, schema.get("minimum"), path, "at least", true, errors);
        compareBound(number, schema.get("maximum"), path, "at most", false, errors);
        compareBound(number, schema.get("exclusiveMinimum"), path,
                "greater than", true, errors);
        compareBound(number, schema.get("exclusiveMaximum"), path,
                "less than", false, errors);
    }

    private static void compareBound(
            BigDecimal value,
            Object boundObject,
            String path,
            String message,
            boolean lower,
            List<String> errors) {
        if (!(boundObject instanceof Number)) {
            return;
        }
        BigDecimal bound = decimal((Number) boundObject);
        int comparison = value.compareTo(bound);
        boolean exclusive = message.startsWith("greater") || message.startsWith("less");
        boolean invalid = lower
                ? exclusive ? comparison <= 0 : comparison < 0
                : exclusive ? comparison >= 0 : comparison > 0;
        if (invalid) {
            errors.add(path + " must be " + message + " " + bound.toPlainString());
        }
    }

    private static boolean matchesConstAndEnum(
            Object value, Map<String, Object> schema) {
        if (schema.containsKey("const")
                && !equivalent(value, schema.get("const"))) {
            return false;
        }
        Object values = schema.get("enum");
        if (!(values instanceof Collection<?>)) {
            return true;
        }
        for (Object allowed : (Collection<?>) values) {
            if (equivalent(value, allowed)) {
                return true;
            }
        }
        return false;
    }

    private static boolean equivalent(Object left, Object right) {
        if (left instanceof Number && right instanceof Number) {
            return decimal((Number) left).compareTo(decimal((Number) right)) == 0;
        }
        return Objects.deepEquals(left, right);
    }

    private static boolean isIntegral(Number value) {
        try {
            return decimal(value).stripTrailingZeros().scale() <= 0;
        } catch (NumberFormatException ignored) {
            return false;
        }
    }

    private static BigDecimal decimal(Number value) {
        return new BigDecimal(value.toString());
    }

    private static Integer integerKeyword(
            Map<String, Object> schema, String keyword) {
        Object value = schema.get(keyword);
        if (!(value instanceof Number)) {
            return null;
        }
        int integer = ((Number) value).intValue();
        return integer < 0 ? null : integer;
    }

    private static List<?> asList(Object value) {
        if (value instanceof List<?>) {
            return (List<?>) value;
        }
        if (value instanceof Collection<?>) {
            return new ArrayList<>((Collection<?>) value);
        }
        if (value == null || !value.getClass().isArray()) {
            return null;
        }
        int length = Array.getLength(value);
        List<Object> result = new ArrayList<>(length);
        for (int index = 0; index < length; index++) {
            result.add(Array.get(value, index));
        }
        return result;
    }

    private static Map<String, Object> stringKeyMap(Map<?, ?> source) {
        java.util.LinkedHashMap<String, Object> result = new java.util.LinkedHashMap<>();
        source.forEach((key, value) -> result.put(String.valueOf(key), value));
        return result;
    }
}
