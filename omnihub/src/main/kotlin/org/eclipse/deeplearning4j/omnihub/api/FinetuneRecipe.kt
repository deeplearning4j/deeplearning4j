/*
 *  ******************************************************************************
 *  *
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
 *  * See the NOTICE file distributed with this work for additional
 *  * information regarding copyright ownership.
 *  * Unless required by applicable law or agreed to in writing, software
 *  * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 *  * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 *  * License for the specific language governing permissions and limitations
 *  * under the License.
 *  *
 *  * SPDX-License-Identifier: Apache-2.0
 *  *****************************************************************************
 */
package org.eclipse.deeplearning4j.omnihub.api

data class FinetuneRecipe(
    val loraTargetModules: List<String>,
    val recommendedRank: Int = 16,
    val recommendedAlpha: Int = 32,
    val recommendedLr: Double = 2e-4,
    val chatTemplate: String? = null,
    val maxSeqLength: Int = 2048,
    val quantizedBase: Boolean = false,
    val recommendedBatchSize: Int = 4,
    val gradientAccumulationSteps: Int = 4
) {
    companion object {
        @JvmStatic
        fun llama3(): FinetuneRecipe = FinetuneRecipe(
            loraTargetModules = listOf("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"),
            recommendedRank = 16,
            recommendedAlpha = 32,
            recommendedLr = 2e-4,
            maxSeqLength = 8192
        )

        @JvmStatic
        fun qwen2(): FinetuneRecipe = FinetuneRecipe(
            loraTargetModules = listOf("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"),
            recommendedRank = 16,
            recommendedAlpha = 32,
            recommendedLr = 2e-4,
            maxSeqLength = 4096
        )

        @JvmStatic
        fun mistral(): FinetuneRecipe = FinetuneRecipe(
            loraTargetModules = listOf("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"),
            recommendedRank = 16,
            recommendedAlpha = 32,
            recommendedLr = 2e-4,
            maxSeqLength = 4096
        )

        @JvmStatic
        fun gemma(): FinetuneRecipe = FinetuneRecipe(
            loraTargetModules = listOf("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"),
            recommendedRank = 16,
            recommendedAlpha = 16,
            recommendedLr = 2e-4,
            maxSeqLength = 2048
        )

        @JvmStatic
        fun phi3(): FinetuneRecipe = FinetuneRecipe(
            loraTargetModules = listOf("qkv_proj", "o_proj", "gate_up_proj", "down_proj"),
            recommendedRank = 16,
            recommendedAlpha = 32,
            recommendedLr = 2e-4,
            maxSeqLength = 4096
        )

        @JvmStatic
        fun deepseek(): FinetuneRecipe = FinetuneRecipe(
            loraTargetModules = listOf("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"),
            recommendedRank = 16,
            recommendedAlpha = 32,
            recommendedLr = 1e-4,
            maxSeqLength = 4096
        )
    }
}
