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

import java.util.Arrays;
import java.util.Collections;
import java.util.List;

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
 *   <li>{@code "tool_call"} — enforce the canonical tool-call shape
 *       {@code {"tool": "<name>", "args": {...}}}; use the factory
 *       {@link #toolCall(String...)}.</li>
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
     * Top-K cap for constraint evaluation.
     *
     * <p>Only the top {@code evalTopK} logit positions are checked against the
     * constraint before masking. If none of the top-K tokens pass, the masker
     * widens to the full vocabulary. Larger values are more thorough but slower
     * for large vocabularies. Default: 256.</p>
     */
    @Builder.Default
    private int evalTopK = 256;

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
        if (names == null || names.length == 0) {
            throw new IllegalArgumentException("toolCall() requires at least one tool name");
        }
        return ConstraintConfig.builder()
                .type(ToolCallConstraint.TYPE)
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
        if (JsonObjectConstraint.TYPE.equals(type)) {
            return new JsonObjectConstraint();
        }
        if (ToolCallConstraint.TYPE.equals(type)) {
            if (toolNames == null || toolNames.isEmpty()) {
                throw new IllegalArgumentException(
                        "ConstraintConfig with type=\"tool_call\" requires at least one toolName");
            }
            return new ToolCallConstraint(toolNames);
        }
        throw new IllegalArgumentException(
                "Unknown constraint type: \"" + type + "\". Supported: \"json_object\", \"tool_call\"");
    }
}
