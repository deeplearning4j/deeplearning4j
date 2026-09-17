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

import java.util.List;

/**
 * Per-architecture-family chat template.
 *
 * <p>Implementations are pure functions from an ordered message list plus options to a
 * prompt string: no model I/O, no tokenizer, no global state. Golden tests in
 * platform-tests pin the rendered output byte-for-byte against the official upstream
 * templates.</p>
 *
 * @author Eclipse Deeplearning4j Contributors
 */
public interface FamilyChatTemplate {

    /** Render the full prompt for the conversation. */
    String apply(List<FamilyMessage> messages, FamilyChatOptions options);

    /** Render the full prompt with default options. */
    default String apply(List<FamilyMessage> messages) {
        return apply(messages, FamilyChatOptions.defaults());
    }

    /**
     * Render the prompt for the next assistant generation: history rendered with
     * {@code addGenerationPrompt=false} semantics where the family defines it
     * (Qwen: no trailing {@code <|im_start|>assistant<think>}; DeepSeek: no assistant
     * header appended), plus the family generation prefix. Equivalent to calling
     * {@link #apply(List, FamilyChatOptions)} with
     * {@code options.toBuilder().addGenerationPrompt(true).build()}.
     */
    default String applyForGeneration(List<FamilyMessage> messages, FamilyChatOptions options) {
        return apply(messages, options.toBuilder().addGenerationPrompt(true).build());
    }

    /** Architecture family implemented by this template. */
    Family family();
}
