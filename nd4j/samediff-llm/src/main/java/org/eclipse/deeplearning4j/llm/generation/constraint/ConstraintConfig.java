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

import lombok.Builder;
import lombok.Data;
import org.eclipse.deeplearning4j.llm.tokenizer.ChatTemplate;

import java.util.Arrays;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * Configuration carrier for structured-output / constrained decoding.
 *
 * <p>A {@code ConstraintConfig} is an immutable, serialization-friendly description of
 * the constraint to apply during token generation. Pass it to
 * {@link ConstraintMasker} (via {@link #buildConstraint()}) to obtain the live
 * automaton instance.</p>
 *
 * <h2>Supported types</h2>
 * <ul>
 *   <li>{@code "json_object"} — enforce a syntactically valid JSON object;
 *       use the factory {@link #jsonObject()}.</li>
 *   <li>{@code "tool_call"} — enforce the canonical JSON tool-call shape
 *       {@code {"tool": "<name>", "args": {...}}}; use the factory
 *       {@link #toolCall(String...)}.</li>
 *   <li>{@code "native_tool_call"} — enforce the native function-call envelope
 *       {@code <|tool_call_start|>[name(key=value)]}; use the factory
 *       {@link #nativeToolCall(String...)}.</li>
 * </ul>
 *
 * <h2>Top-K evaluation cap</h2>
 * <p>{@link #evalTopK} limits how many of the highest-logit tokens are checked against
 * the constraint before falling back to the full vocabulary. This is a performance
 * knob: setting it to 256 (default) means only the top-256 logits are evaluated per
 * step; if none pass the constraint the masker widens to the full vocab automatically
 * (see {@link ConstraintMasker#maskLogits}).</p>
 *
 * <h2>Example</h2>
 * <pre>{@code
 * // JSON object output
 * ConstraintConfig cfg = ConstraintConfig.jsonObject();
 *
 * // Tool call with specific tool names
 * ConstraintConfig tc = ConstraintConfig.toolCall("search_web", "run_code");
 *
 * // Build the live automaton
 * TextConstraint constraint = cfg.buildConstraint();
 * }</pre>
 *
 * @author Eclipse Deeplearning4j Contributors
 * @see ConstraintMasker
 * @see JsonObjectConstraint
 * @see ToolCallConstraint
 */
@Data
@Builder(toBuilder = true)
public class ConstraintConfig {

    /**
     * Type of constraint to enforce.
     *
     * <p>Recognised values: {@code "json_object"}, {@code "tool_call"}.</p>
     */
    private String type;

    /**
     * For {@code type="tool_call"}: the set of allowed tool names.
     *
     * <p>At least one name must be supplied when type is {@code "tool_call"}.
     * Ignored for other constraint types.</p>
     */
    @Builder.Default
    private List<String> toolNames = Collections.emptyList();

    /**
     * Ordered required argument names for native tools, keyed by tool name.
     *
     * <p>An empty map preserves the generic native-call constraint. When populated, the
     * automaton requires each tool's declared arguments exactly once and in schema order,
     * preventing small models from repeating one field and stuffing the remaining values
     * into it.</p>
     */
    @Builder.Default
    private Map<String, List<String>> toolArgumentNames = Collections.emptyMap();

    /**
     * Optional exact string values from each argument's JSON-schema enum/const.
     */
    @Builder.Default
    private Map<String, Map<String, List<String>>> toolArgumentValues =
            Collections.emptyMap();

    /**
     * Complete JSON parameter schema for each native tool.
     *
     * <p>The native constraint uses this schema while masking tokens so value
     * types and collection bounds are not lost after the chat template renders
     * the tool declaration.</p>
     */
    @Builder.Default
    private Map<String, Map<String, Object>> toolParameterSchemas =
            Collections.emptyMap();

    /**
     * Top-K cap for constraint evaluation.
     *
     * <p>Only the top {@code evalTopK} logit positions are checked against the
     * constraint before masking. If none of the top-K tokens pass, the masker
     * widens to the full vocabulary. Larger values are more thorough but slower
     * for large vocabularies. Default: 256.</p>
     */
    @Builder.Default
    private int evalTopK = 256;

    /**
     * Template-declared blocks which are already open when assistant generation begins.
     *
     * <p>The definitions are ordered outermost to innermost. Their content remains
     * unconstrained, but each exact closing delimiter is required before the configured
     * structured-output grammar begins. This composes model-owned output sections with
     * JSON, native, or XML payload constraints without naming any particular block type.</p>
     */
    @Builder.Default
    private List<ChatTemplate.OutputBlockDefinition> outputBlocks = Collections.emptyList();

    // -------------------------------------------------------------------------
    // Factory methods
    // -------------------------------------------------------------------------

    /**
     * Creates a {@code ConstraintConfig} for JSON-object constrained decoding.
     *
     * @return a new {@code ConstraintConfig} with {@code type="json_object"}
     */
    public static ConstraintConfig jsonObject() {
        return ConstraintConfig.builder()
                .type(JsonObjectConstraint.TYPE)
                .build();
    }

    /**
     * Creates a {@code ConstraintConfig} for tool-call constrained decoding.
     *
     * @param names one or more valid tool names
     * @return a new {@code ConstraintConfig} with {@code type="tool_call"} and the
     *         supplied names
     * @throws IllegalArgumentException if no names are provided
     */
    public static ConstraintConfig toolCall(String... names) {
        return namedToolConstraint(ToolCallConstraint.TYPE, "toolCall", names);
    }

    /**
     * Creates a constraint for the native sentinel/function-call protocol.
     *
     * @param names one or more valid tool names
     * @return a native tool-call constraint configuration
     */
    public static ConstraintConfig nativeToolCall(String... names) {
        return namedToolConstraint(NativeToolCallConstraint.TYPE, "nativeToolCall", names);
    }

    /**
     * Creates a schema-aware native tool-call constraint.
     *
     * @param argumentNamesByTool ordered required argument names keyed by tool name
     * @return a native constraint that enforces both tool and argument names
     */
    public static ConstraintConfig nativeToolCall(
            Map<String, List<String>> argumentNamesByTool) {
        return nativeToolCall(argumentNamesByTool, Collections.emptyMap());
    }

    public static ConstraintConfig nativeToolCall(
            Map<String, List<String>> argumentNamesByTool,
            Map<String, Map<String, List<String>>> argumentValuesByTool) {
        return nativeToolCall(
                argumentNamesByTool, argumentValuesByTool, Collections.emptyMap());
    }

    public static ConstraintConfig nativeToolCall(
            Map<String, List<String>> argumentNamesByTool,
            Map<String, Map<String, List<String>>> argumentValuesByTool,
            Map<String, Map<String, Object>> parameterSchemasByTool) {
        if (argumentNamesByTool == null || argumentNamesByTool.isEmpty()) {
            throw new IllegalArgumentException(
                    "nativeToolCall() requires at least one tool schema");
        }
        Map<String, List<String>> copied = new LinkedHashMap<>();
        argumentNamesByTool.forEach((name, arguments) -> {
            if (name == null || name.isBlank()) {
                throw new IllegalArgumentException("Native tool names must not be blank");
            }
            copied.put(name, arguments == null ? List.of() : List.copyOf(arguments));
        });

        Map<String, Map<String, List<String>>> copiedValues = new LinkedHashMap<>();
        if (argumentValuesByTool != null) {
            argumentValuesByTool.forEach((toolName, valuesByArgument) -> {
                if (!copied.containsKey(toolName) || valuesByArgument == null) {
                    return;
                }
                Map<String, List<String>> toolValues = new LinkedHashMap<>();
                valuesByArgument.forEach((argumentName, values) -> {
                    if (argumentName != null && values != null && !values.isEmpty()) {
                        toolValues.put(argumentName, List.copyOf(values));
                    }
                });
                if (!toolValues.isEmpty()) {
                    copiedValues.put(toolName, Collections.unmodifiableMap(toolValues));
                }
            });
        }
        Map<String, Map<String, Object>> copiedSchemas = new LinkedHashMap<>();
        if (parameterSchemasByTool != null) {
            parameterSchemasByTool.forEach((toolName, schema) -> {
                if (copied.containsKey(toolName) && schema != null) {
                    copiedSchemas.put(toolName,
                            Collections.unmodifiableMap(new LinkedHashMap<>(schema)));
                }
            });
        }
        return ConstraintConfig.builder()
                .type(NativeToolCallConstraint.TYPE)
                .toolNames(List.copyOf(copied.keySet()))
                .toolArgumentNames(Collections.unmodifiableMap(copied))
                .toolArgumentValues(Collections.unmodifiableMap(copiedValues))
                .toolParameterSchemas(Collections.unmodifiableMap(copiedSchemas))
                .build();
    }

    /** Creates a schema-aware constraint for a template-declared XML function protocol. */
    public static ConstraintConfig xmlToolCall(
            Map<String, List<String>> argumentNamesByTool,
            Map<String, Map<String, List<String>>> argumentValuesByTool,
            Map<String, Map<String, Object>> parameterSchemasByTool) {
        return nativeToolCall(
                argumentNamesByTool, argumentValuesByTool, parameterSchemasByTool)
                .toBuilder()
                .type(XmlToolCallConstraint.TYPE)
                .build();
    }

    private static ConstraintConfig namedToolConstraint(
            String type, String factoryName, String... names) {
        if (names == null || names.length == 0) {
            throw new IllegalArgumentException(
                    factoryName + "() requires at least one tool name");
        }
        return ConstraintConfig.builder()
                .type(type)
                .toolNames(Arrays.asList(names))
                .build();
    }

    // -------------------------------------------------------------------------
    // Constraint instantiation
    // -------------------------------------------------------------------------

    /**
     * Builds and returns a fresh {@link TextConstraint} instance described by this config.
     *
     * @return a new, reset constraint automaton
     * @throws IllegalArgumentException if the {@link #type} is unrecognised or
     *                                  required parameters (e.g., tool names) are missing
     */
    public TextConstraint buildConstraint() {
        TextConstraint constraint;
        if (JsonObjectConstraint.TYPE.equals(type)) {
            constraint = new JsonObjectConstraint();
        } else if (ToolCallConstraint.TYPE.equals(type)
                || NativeToolCallConstraint.TYPE.equals(type)
                || XmlToolCallConstraint.TYPE.equals(type)) {
            if (toolNames == null || toolNames.isEmpty()) {
                throw new IllegalArgumentException(
                        "ConstraintConfig with type=\"" + type
                                + "\" requires at least one toolName");
            }
            if (ToolCallConstraint.TYPE.equals(type)) {
                constraint = new ToolCallConstraint(toolNames);
            } else if (NativeToolCallConstraint.TYPE.equals(type)) {
                constraint = new NativeToolCallConstraint(
                        toolNames, toolArgumentNames, toolArgumentValues,
                        toolParameterSchemas);
            } else {
                constraint = new XmlToolCallConstraint(
                        toolNames, toolArgumentNames, toolParameterSchemas);
            }
        } else {
            throw new IllegalArgumentException(
                    "Unknown constraint type: \"" + type
                            + "\". Supported: \"json_object\", \"tool_call\", "
                            + "\"native_tool_call\", \"xml_tool_call\"");
        }
        return outputBlocks == null || outputBlocks.isEmpty()
                ? constraint
                : new OutputBlockSequenceConstraint(outputBlocks, constraint);
    }

    /**
     * Prefix grammar for template-opened output blocks followed by a strict payload.
     *
     * <p>This adapter is deliberately agnostic to block names and payload protocol. It
     * derives progress entirely from the ordered delimiter definitions and delegates the
     * suffix to the configured constraint after every block has closed.</p>
     */
    private static final class OutputBlockSequenceConstraint implements TextConstraint {
        private final List<ChatTemplate.OutputBlockDefinition> blocks;
        private final TextConstraint payload;

        private OutputBlockSequenceConstraint(
                List<ChatTemplate.OutputBlockDefinition> blocks, TextConstraint payload) {
            this.blocks = List.copyOf(blocks);
            this.payload = payload;
        }

        @Override
        public boolean canExtend(String currentText, String piece) {
            String current = currentText == null ? "" : currentText;
            String extension = piece == null ? "" : piece;
            BlockState before = state(current);
            BlockState after = state(current + extension);
            if (!after.blocksClosed) {
                return true;
            }
            if (!before.blocksClosed) {
                if (after.payloadText.isEmpty()) {
                    return true;
                }
                return payload.canExtend("", after.payloadText);
            }
            if (!after.payloadText.startsWith(before.payloadText)) {
                return false;
            }
            String payloadExtension = after.payloadText.substring(before.payloadText.length());
            if (payloadExtension.isEmpty()) {
                return before.rawPayloadText.isEmpty()
                        && !after.rawPayloadText.isEmpty()
                        && after.rawPayloadText.chars().allMatch(Character::isWhitespace);
            }
            return payload.canExtend(before.payloadText, payloadExtension);
        }

        @Override
        public boolean allowsSpecialToken(String currentText, String piece) {
            BlockState current = state(currentText == null ? "" : currentText);
            if (!current.blocksClosed) {
                String expected = blocks.get(current.openBlockIndex).getClosingDelimiter();
                return expected.equals(piece) && canExtend(currentText, piece);
            }
            return payload.allowsSpecialToken(current.payloadText, piece);
        }

        @Override
        public boolean isAccepting(String currentText) {
            BlockState current = state(currentText == null ? "" : currentText);
            return current.blocksClosed && payload.isAccepting(current.payloadText);
        }

        @Override
        public TextConstraint reset() {
            return new OutputBlockSequenceConstraint(blocks, payload.reset());
        }

        @Override
        public String type() {
            return "output_blocks_then_" + payload.type();
        }

        private BlockState state(String text) {
            int cursor = 0;
            for (int index = blocks.size() - 1; index >= 0; index--) {
                String closing = blocks.get(index).getClosingDelimiter();
                int closingAt = text.indexOf(closing, cursor);
                if (closingAt < 0) {
                    return new BlockState(false, index, "", "");
                }
                cursor = closingAt + closing.length();
            }
            String rawPayload = text.substring(cursor);
            return new BlockState(true, -1, rawPayload.stripLeading(), rawPayload);
        }

        private static final class BlockState {
            private final boolean blocksClosed;
            private final int openBlockIndex;
            private final String payloadText;
            private final String rawPayloadText;

            private BlockState(boolean blocksClosed, int openBlockIndex,
                               String payloadText, String rawPayloadText) {
                this.blocksClosed = blocksClosed;
                this.openBlockIndex = openBlockIndex;
                this.payloadText = payloadText;
                this.rawPayloadText = rawPayloadText;
            }
        }
    }
}
