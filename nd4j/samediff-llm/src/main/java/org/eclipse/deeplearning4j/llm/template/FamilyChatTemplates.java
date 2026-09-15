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

/**
 * Entry point for the architecture-family chat template library.
 *
 * <p>Each family mirrors the official upstream template for that model line; goldens in
 * platform-tests pin the rendering byte-for-byte. All templates are pure string
 * transforms, so they can be unit tested and swapped without touching
 * {@code GenerationPipeline}.</p>
 *
 * <pre>{@code
 * FamilyChatTemplate template = FamilyChatTemplates.forFamily(Family.QWEN);
 * String prompt = template.applyForGeneration(
 *     List.of(FamilyMessage.system("You are helpful."), FamilyMessage.user("Hi")),
 *     FamilyChatOptions.defaults());
 * }</pre>
 *
 * @author Eclipse Deeplearning4j Contributors
 */
public final class FamilyChatTemplates {

    private FamilyChatTemplates() {
    }

    /** Template for the given family. */
    public static FamilyChatTemplate forFamily(Family family) {
        switch (family) {
            case QWEN:
                return QwenFamilyChatTemplate.flashNext();
            case GLM:
                return GlmFamilyChatTemplate.instance();
            case DEEPSEEK:
                return DeepSeekFamilyChatTemplate.instance();
            default:
                throw new IllegalArgumentException("Unsupported chat template family: " + family);
        }
    }

    /** Qwen3.8-Flash-Next style template (reasoning_effort + preserve_thinking). */
    public static QwenFamilyChatTemplate qwen38FlashNext() {
        return QwenFamilyChatTemplate.flashNext();
    }

    /** Qwen3.5-27B style template (inline think extraction, drop history thinking). */
    public static QwenFamilyChatTemplate qwen35() {
        return QwenFamilyChatTemplate.qwen35();
    }

    /** GLM-5.x (zai-org/GLM-5.3-Flash) template. */
    public static GlmFamilyChatTemplate glm() {
        return GlmFamilyChatTemplate.instance();
    }

    /** DeepSeek-V4.1-Flash prompt encoder. */
    public static DeepSeekFamilyChatTemplate deepseek() {
        return DeepSeekFamilyChatTemplate.instance();
    }
}
