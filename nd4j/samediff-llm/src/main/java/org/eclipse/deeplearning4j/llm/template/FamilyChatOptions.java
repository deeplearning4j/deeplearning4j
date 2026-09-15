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
 *  *  distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 *  *  WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 *  *  License for the specific language governing permissions and limitations
 *  *  under the License.
 *  *
 *  * SPDX-License-Identifier: Apache-2.0
 *  *****************************************************************************
 */

package org.eclipse.deeplearning4j.llm.template;

import java.util.Objects;

/**
 * Shared knobs for {@link FamilyChatTemplate} rendering. Each field mirrors a knob
 * from the official upstream template that defines it; families ignore fields they
 * do not define. Immutable; use {@link #toBuilder()} to derive variants.
 *
 * @author Eclipse Deeplearning4j Contributors
 */
public final class FamilyChatOptions {

    private final boolean addGenerationPrompt;
    private final boolean enableThinking;
    private final QwenReasoningEffort qwenReasoningEffort;
    private final boolean preserveThinking;
    private final GlmReasoningEffort glmReasoningEffort;
    private final boolean glmClearThinking;
    private final Integer deepseekReasoningEffort;
    private final boolean deepseekDropThinking;
    private final boolean addBosToken;

    private FamilyChatOptions(Builder builder) {
        this.addGenerationPrompt = builder.addGenerationPrompt;
        this.enableThinking = builder.enableThinking;
        this.qwenReasoningEffort = builder.qwenReasoningEffort;
        this.preserveThinking = builder.preserveThinking;
        this.glmReasoningEffort = builder.glmReasoningEffort;
        this.glmClearThinking = builder.glmClearThinking;
        this.deepseekReasoningEffort = builder.deepseekReasoningEffort;
        this.deepseekDropThinking = builder.deepseekDropThinking;
        this.addBosToken = builder.addBosToken;
    }

    public static Builder builder() {
        return new Builder();
    }

    /** Defaults matching each official template's implicit defaults. */
    public static FamilyChatOptions defaults() {
        return builder().build();
    }

    public boolean isAddGenerationPrompt() {
        return addGenerationPrompt;
    }

    /** Qwen/GLM-style thinking toggle. Default true. */
    public boolean isEnableThinking() {
        return enableThinking;
    }

    /** Qwen3.8: reasoning_effort knob. Default xhigh per the official template. */
    public QwenReasoningEffort getQwenReasoningEffort() {
        return qwenReasoningEffort;
    }

    /** Qwen3.8: preserve_thinking knob. Default true per the official template. */
    public boolean isPreserveThinking() {
        return preserveThinking;
    }

    /** GLM-5: reasoning_effort knob (low/high/max). Default max per the official template. */
    public GlmReasoningEffort getGlmReasoningEffort() {
        return glmReasoningEffort;
    }

    /** GLM-5: clear_thinking knob. Default false per the official template. */
    public boolean isGlmClearThinking() {
        return glmClearThinking;
    }

    /** DeepSeek-V4.1: numeric budget 1-100; null means the official default (75). */
    public Integer getDeepseekReasoningEffort() {
        return deepseekReasoningEffort;
    }

    /** DeepSeek-V4.1: drop thinking from pre-last-query assistant turns. Default true. */
    public boolean isDeepseekDropThinking() {
        return deepseekDropThinking;
    }

    /** Emit the family BOS token where the official encoding does. Default true. */
    public boolean isAddBosToken() {
        return addBosToken;
    }

    public Builder toBuilder() {
        return builder()
                .addGenerationPrompt(addGenerationPrompt)
                .enableThinking(enableThinking)
                .qwenReasoningEffort(qwenReasoningEffort)
                .preserveThinking(preserveThinking)
                .glmReasoningEffort(glmReasoningEffort)
                .glmClearThinking(glmClearThinking)
                .deepseekReasoningEffort(deepseekReasoningEffort)
                .deepseekDropThinking(deepseekDropThinking)
                .addBosToken(addBosToken);
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) {
            return true;
        }
        if (!(o instanceof FamilyChatOptions)) {
            return false;
        }
        FamilyChatOptions that = (FamilyChatOptions) o;
        return addGenerationPrompt == that.addGenerationPrompt
                && enableThinking == that.enableThinking
                && preserveThinking == that.preserveThinking
                && glmClearThinking == that.glmClearThinking
                && deepseekDropThinking == that.deepseekDropThinking
                && addBosToken == that.addBosToken
                && qwenReasoningEffort == that.qwenReasoningEffort
                && glmReasoningEffort == that.glmReasoningEffort
                && Objects.equals(deepseekReasoningEffort, that.deepseekReasoningEffort);
    }

    @Override
    public int hashCode() {
        return Objects.hash(addGenerationPrompt, enableThinking, qwenReasoningEffort,
                preserveThinking, glmReasoningEffort, glmClearThinking,
                deepseekReasoningEffort, deepseekDropThinking, addBosToken);
    }

    @Override
    public String toString() {
        return "FamilyChatOptions{addGenerationPrompt=" + addGenerationPrompt
                + ", enableThinking=" + enableThinking
                + ", qwenReasoningEffort=" + qwenReasoningEffort
                + ", preserveThinking=" + preserveThinking
                + ", glmReasoningEffort=" + glmReasoningEffort
                + ", glmClearThinking=" + glmClearThinking
                + ", deepseekReasoningEffort=" + deepseekReasoningEffort
                + ", deepseekDropThinking=" + deepseekDropThinking
                + ", addBosToken=" + addBosToken + '}';
    }

    /** Qwen3.8 reasoning_effort values accepted by the official template. */
    public enum QwenReasoningEffort {
        XHIGH("xhigh"),
        MEDIUM("medium"),
        LOW("low");

        private final String label;

        QwenReasoningEffort(String label) {
            this.label = label;
        }

        public String label() {
            return label;
        }
    }

    /** GLM-5 reasoning_effort values accepted by the official template. */
    public enum GlmReasoningEffort {
        LOW("low"),
        HIGH("high"),
        MAX("max");

        private final String label;

        GlmReasoningEffort(String label) {
            this.label = label;
        }

        /** Template-level label, already capitalized like the Jinja {@code capitalize} filter. */
        public String capitalized() {
            return Character.toUpperCase(label.charAt(0)) + label.substring(1);
        }
    }

    /** Builder for {@link FamilyChatOptions}. */
    public static final class Builder {
        private boolean addGenerationPrompt = true;
        private boolean enableThinking = true;
        private QwenReasoningEffort qwenReasoningEffort = QwenReasoningEffort.XHIGH;
        private boolean preserveThinking = true;
        private GlmReasoningEffort glmReasoningEffort = GlmReasoningEffort.MAX;
        private boolean glmClearThinking = false;
        private Integer deepseekReasoningEffort = null;
        private boolean deepseekDropThinking = true;
        private boolean addBosToken = true;

        public Builder addGenerationPrompt(boolean value) {
            this.addGenerationPrompt = value;
            return this;
        }

        public Builder enableThinking(boolean value) {
            this.enableThinking = value;
            return this;
        }

        public Builder qwenReasoningEffort(QwenReasoningEffort value) {
            this.qwenReasoningEffort = value;
            return this;
        }

        public Builder preserveThinking(boolean value) {
            this.preserveThinking = value;
            return this;
        }

        public Builder glmReasoningEffort(GlmReasoningEffort value) {
            this.glmReasoningEffort = value == null ? GlmReasoningEffort.MAX : value;
            return this;
        }

        public Builder glmClearThinking(boolean value) {
            this.glmClearThinking = value;
            return this;
        }

        /** DeepSeek numeric effort budget in [1,100]. */
        public Builder deepseekReasoningEffort(int value) {
            if (value < 1 || value > 100) {
                throw new IllegalArgumentException(
                        "DeepSeek reasoning effort must be within [1,100]: " + value);
            }
            this.deepseekReasoningEffort = value;
            return this;
        }

        /**
         * Nullable variant used by toBuilder(): null means "no explicit budget"
         * (the official default applies) and is preserved instead of unboxed.
         */
        public Builder deepseekReasoningEffort(Integer value) {
            if (value == null) {
                return this;
            }
            return deepseekReasoningEffort(value.intValue());
        }

        /** DeepSeek effort alias: low -> 50, high -> 75, max -> 100. */
        public Builder deepseekReasoningEffort(String alias) {
            return deepseekReasoningEffort(DeepSeekFamilyChatTemplate.effortAlias(alias));
        }

        public Builder deepseekDropThinking(boolean value) {
            this.deepseekDropThinking = value;
            return this;
        }

        public Builder addBosToken(boolean value) {
            this.addBosToken = value;
            return this;
        }

        public FamilyChatOptions build() {
            return new FamilyChatOptions(this);
        }
    }
}
