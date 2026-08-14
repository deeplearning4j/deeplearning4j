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

import java.util.Arrays;
import java.util.Collections;
import java.util.List;

/**
 * A {@link TextConstraint} that accepts exactly one tool-call JSON object in the
 * canonical form:
 *
 * <pre>{@code
 * {"tool": "<name>", "args": <json-object>}
 * }</pre>
 *
 * <h2>Format contract</h2>
 * <p>This constraint enforces the <em>canonical spaced form</em> shown above.
 * LLM prompts that use this constraint should explicitly request:
 * {@code Respond ONLY with JSON in the form: {"tool": "toolname", "args": {...}}}</p>
 * <p>Specifically, the literal substrings {@code {"tool": "} and {@code ", "args": }
 * are enforced character-for-character. This makes the automaton O(n) simple string
 * matching rather than a general parser. The LLM's in-context instruction is responsible
 * for requesting this format.</p>
 *
 * <h2>Phase model</h2>
 * <p>The constraint advances through five phases, determined purely from the emitted text
 * (no mutable state is kept between calls):</p>
 * <ol>
 *   <li>{@link Phase#PREFIX} — waiting for the literal {@code {"tool": "}</li>
 *   <li>{@link Phase#TOOL_NAME} — collecting the tool name; only prefixes of at least one
 *       known tool name are allowed</li>
 *   <li>{@link Phase#AFTER_NAME} — waiting for the literal {@code ", "args": }</li>
 *   <li>{@link Phase#ARGS} — the args value; delegated to
 *       {@link JsonObjectConstraint} prefix logic</li>
 *   <li>{@link Phase#DONE} — the whole expression is complete and accepted</li>
 * </ol>
 *
 * @author Eclipse Deeplearning4j Contributors
 * @see JsonObjectConstraint
 * @see ConstraintConfig#toolCall(String...)
 */
public class ToolCallConstraint implements TextConstraint {

    /** Type identifier returned by {@link #type()}. */
    public static final String TYPE = "tool_call";

    /**
     * The literal prefix that every tool call must begin with.
     * Canonical spaced form is enforced (see class Javadoc).
     */
    static final String TOOL_PREFIX = "{\"tool\": \"";

    /**
     * The literal separator between the tool name and the args value.
     * Note: this begins with {@code "} which closes the tool-name string.
     */
    static final String AFTER_TOOL_SUFFIX = "\", \"args\": ";

    /**
     * The phases of the tool-call automaton.
     */
    enum Phase {
        /** Waiting for the literal {@value ToolCallConstraint#TOOL_PREFIX}. */
        PREFIX,
        /** Collecting the tool name; text is a prefix of at least one known name. */
        TOOL_NAME,
        /** Waiting for the literal {@value ToolCallConstraint#AFTER_TOOL_SUFFIX}. */
        AFTER_NAME,
        /** Collecting the args value (must be a complete JSON object). */
        ARGS,
        /** The whole expression is complete and accepted. */
        DONE
    }

    private final List<String> toolNames;

    /**
     * Constructs a ToolCallConstraint that accepts calls to any of the given tool names.
     *
     * @param toolNames the set of valid tool names; must not be null or empty
     * @throws IllegalArgumentException if {@code toolNames} is empty
     */
    public ToolCallConstraint(List<String> toolNames) {
        if (toolNames == null || toolNames.isEmpty()) {
            throw new IllegalArgumentException("ToolCallConstraint requires at least one tool name");
        }
        this.toolNames = Collections.unmodifiableList(toolNames);
    }

    /**
     * Varargs convenience constructor.
     *
     * @param toolNames one or more valid tool names
     */
    public ToolCallConstraint(String... toolNames) {
        this(Arrays.asList(toolNames));
    }

    // -------------------------------------------------------------------------
    // Phase detection — derived purely from emitted text
    // -------------------------------------------------------------------------

    /**
     * Determines the current automaton phase from the accumulated text.
     *
     * @param text the text emitted so far
     * @return the current phase
     */
    Phase detectPhase(String text) {
        // text shorter than TOOL_PREFIX — still assembling the opening literal.
        if (text.length() < TOOL_PREFIX.length()) {
            return Phase.PREFIX;
        }

        if (!text.startsWith(TOOL_PREFIX)) {
            // Structurally broken — will be rejected in canExtend.
            return Phase.PREFIX;
        }

        // text starts with TOOL_PREFIX; extract the part after it.
        String afterPrefix = text.substring(TOOL_PREFIX.length());

        // Try to find AFTER_TOOL_SUFFIX in afterPrefix; the text before it is the tool name.
        int suffixIdx = afterPrefix.indexOf(AFTER_TOOL_SUFFIX);
        if (suffixIdx >= 0) {
            String toolName = afterPrefix.substring(0, suffixIdx);
            if (toolNames.contains(toolName)) {
                // Complete tool name found. The args start after AFTER_TOOL_SUFFIX.
                // The whole expression is a JSON object: {"tool": "name", "args": <value>}
                // argsText includes everything after the separator, INCLUDING the outer closing '}':
                // e.g. for text = {"tool": "name", "args": {}}, argsText = {}
                // But the LLM generates the whole as one JSON object, so argsText may include
                // the outer closing '}'. We check DONE by verifying the WHOLE string is an
                // accepting JSON object AND has the right structure.
                String argsText = afterPrefix.substring(suffixIdx + AFTER_TOOL_SUFFIX.length());
                if (argsText.isEmpty()) {
                    return Phase.ARGS;
                }
                // Check if the WHOLE text is a complete JSON object (DONE state).
                // A complete tool call IS a complete JSON object: {"tool":"name","args":{...}}
                JsonObjectConstraint jc = new JsonObjectConstraint();
                if (jc.isAccepting(text)) {
                    return Phase.DONE;
                }
                // Partial: the args portion hasn't been closed yet (or the outer } not emitted).
                // Validate the prefix is still structurally sound.
                // argsText may have a trailing '}' that belongs to the outer object.
                // Strip the trailing '}' (outer close) if brace depth would go to 0:
                // Actually, just check if the whole text is still a valid JSON prefix.
                if (JsonObjectConstraint.isValidJsonPrefix(text)) {
                    return Phase.ARGS;
                }
                return Phase.ARGS; // structurally broken but we stay in ARGS
            }
            // AFTER_TOOL_SUFFIX found but extracted name is not in the allowed list.
            // Fall through to check AFTER_NAME / TOOL_NAME below.
        }

        // Check whether afterPrefix is exactly a known tool name (name typed, no suffix yet).
        for (String name : toolNames) {
            if (afterPrefix.equals(name)) {
                // Complete name, waiting for AFTER_TOOL_SUFFIX to start.
                return Phase.AFTER_NAME;
            }
            // Check if afterPrefix = name + (partial AFTER_TOOL_SUFFIX).
            if (afterPrefix.startsWith(name)
                    && AFTER_TOOL_SUFFIX.startsWith(afterPrefix.substring(name.length()))) {
                return Phase.AFTER_NAME;
            }
        }

        // Still assembling the tool name.
        return Phase.TOOL_NAME;
    }

    // -------------------------------------------------------------------------
    // Helpers
    // -------------------------------------------------------------------------

    /**
     * Returns {@code true} if {@code candidate} is a prefix of (or equal to)
     * {@code target}.
     */
    private static boolean isPrefixOf(String candidate, String target) {
        if (candidate.length() > target.length()) return false;
        return target.startsWith(candidate);
    }

    /**
     * Returns {@code true} if {@code candidate} is a valid prefix of the literal
     * {@link #TOOL_PREFIX} string — i.e., the first {@code candidate.length()} chars
     * of TOOL_PREFIX equal {@code candidate}.
     */
    private static boolean isValidToolPrefixPrefix(String candidate) {
        return isPrefixOf(candidate, TOOL_PREFIX);
    }

    /**
     * Extracts the tool-name segment from {@code text} — the part after
     * {@link #TOOL_PREFIX} up to (but not including) the closing quote of the name.
     * Returns null if {@code text} does not start with TOOL_PREFIX.
     */
    private static String extractToolNameSegment(String text) {
        if (!text.startsWith(TOOL_PREFIX)) return null;
        return text.substring(TOOL_PREFIX.length());
    }

    /**
     * Returns {@code true} if {@code nameSegment} is a non-empty prefix of at least
     * one tool name in the list.  An exact match is also a valid prefix.
     */
    private boolean isValidToolNamePrefix(String nameSegment) {
        if (nameSegment.isEmpty()) return true; // empty prefix matches all
        for (String name : toolNames) {
            if (name.startsWith(nameSegment) || nameSegment.equals(name)) {
                return true;
            }
        }
        return false;
    }

    /**
     * Extracts the args portion from {@code text} assuming it is at or beyond
     * {@link Phase#AFTER_NAME}. Returns empty string if the args portion has not
     * started yet.
     */
    private String extractArgsText(String text) {
        String afterPrefix = text.substring(TOOL_PREFIX.length());
        int suffixIdx = afterPrefix.indexOf(AFTER_TOOL_SUFFIX);
        if (suffixIdx < 0) return "";
        return afterPrefix.substring(suffixIdx + AFTER_TOOL_SUFFIX.length());
    }

    /**
     * The canonical tool protocol requires an argument object, never a scalar, array, or
     * parenthesized expression. Empty/whitespace-only text remains a valid prefix while the
     * opening brace is still being generated.
     */
    private boolean hasValidArgsObjectPrefix(String text) {
        if (text == null || !text.startsWith(TOOL_PREFIX)) {
            return false;
        }
        String argsText = extractArgsText(text);
        for (int index = 0; index < argsText.length(); index++) {
            char value = argsText.charAt(index);
            if (!Character.isWhitespace(value)) {
                return value == '{';
            }
        }
        return true;
    }

    /**
     * Extracts the text that follows TOOL_PREFIX and the closed tool-name quote, i.e.,
     * the part that should match (a prefix of) AFTER_TOOL_SUFFIX.  Returns null if
     * a complete tool name cannot yet be found.
     */
    private String extractAfterNameText(String text) {
        String afterPrefix = text.substring(TOOL_PREFIX.length());
        // After the tool name (some prefix of a known name), the next char should start
        // AFTER_TOOL_SUFFIX.  We need to find where the tool name ends.
        // AFTER_TOOL_SUFFIX starts with '"' which closes the name.
        // So: afterPrefix = <partial-name> | <full-name> + <partial-or-full-suffix>
        // Try each known tool name as the one being typed.
        for (String name : toolNames) {
            String expected = name + AFTER_TOOL_SUFFIX;
            if (afterPrefix.equals(name)) {
                // Exact name typed, suffix not yet started.
                return "";
            }
            if (afterPrefix.startsWith(name)) {
                // The full name is present; extract whatever follows it.
                return afterPrefix.substring(name.length());
            }
        }
        return null; // No known name matches as a completed prefix — still in TOOL_NAME phase.
    }

    // -------------------------------------------------------------------------
    // TextConstraint implementation
    // -------------------------------------------------------------------------

    @Override
    public boolean canExtend(String currentText, String piece) {
        if (piece == null || piece.isEmpty()) {
            // Empty piece — valid unless already done or broken.
            Phase phase = detectPhase(currentText);
            return phase != Phase.DONE;
        }

        String combined = currentText + piece;
        Phase phase = detectPhase(currentText);

        switch (phase) {
            case PREFIX: {
                // The combined text must be a valid prefix of TOOL_PREFIX.
                return isValidToolPrefixPrefix(combined);
            }
            case TOOL_NAME: {
                String nameSegment = extractToolNameSegment(combined);
                if (nameSegment == null) {
                    // Combined no longer starts with TOOL_PREFIX — broken.
                    return false;
                }
                // The name segment might already contain the AFTER_TOOL_SUFFIX start ('"').
                // Check if a complete tool name + partial AFTER_TOOL_SUFFIX is being typed.
                for (String name : toolNames) {
                    String withSuffix = name + AFTER_TOOL_SUFFIX;
                    // nameSegment must be a prefix of (name + AFTER_TOOL_SUFFIX) OR exactly name
                    if (isPrefixOf(nameSegment, name) || isPrefixOf(nameSegment, withSuffix)) {
                        return true;
                    }
                }
                return false;
            }
            case AFTER_NAME: {
                // The part after TOOL_PREFIX must start with a complete tool name, then
                // the combined after-name text must be a prefix of AFTER_TOOL_SUFFIX OR
                // start with AFTER_TOOL_SUFFIX (separator complete, entering ARGS phase).
                String afterNameText = extractAfterNameText(combined);
                if (afterNameText == null) return false;
                // Case 1: afterNameText is still a prefix of the separator → still building separator.
                if (isPrefixOf(afterNameText, AFTER_TOOL_SUFFIX)) return true;
                // Case 2: separator is fully present and args have started → treat as ARGS.
                if (afterNameText.startsWith(AFTER_TOOL_SUFFIX)) {
                    // Tool arguments are a JSON object by contract. Validate that root before
                    // delegating nested structural-prefix checks to the JSON constraint.
                    return hasValidArgsObjectPrefix(combined)
                            && JsonObjectConstraint.isValidJsonPrefix(combined);
                }
                return false;
            }
            case ARGS: {
                // The whole expression is one JSON object; validate it as a JSON prefix.
                // We use the whole combined string (not just the args portion) because
                // the outer closing '}' is part of the same JSON object.
                return hasValidArgsObjectPrefix(combined)
                        && JsonObjectConstraint.isValidJsonPrefix(combined);
            }
            case DONE: {
                // Already complete — no extension allowed.
                return false;
            }
            default:
                return false;
        }
    }

    @Override
    public boolean isAccepting(String currentText) {
        return hasValidArgsObjectPrefix(currentText)
                && detectPhase(currentText) == Phase.DONE;
    }

    @Override
    public TextConstraint reset() {
        return new ToolCallConstraint(toolNames);
    }

    @Override
    public String type() {
        return TYPE;
    }

    /**
     * Returns the list of tool names this constraint accepts.
     *
     * @return immutable list of valid tool names
     */
    public List<String> getToolNames() {
        return toolNames;
    }
}
