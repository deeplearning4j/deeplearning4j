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
package org.eclipse.deeplearning4j.omnihub.catalog

import org.eclipse.deeplearning4j.omnihub.api.FinetuneRecipe

data class LLMEntry(
    val name: String,
    val huggingFaceId: String,
    val recipe: FinetuneRecipe,
    val description: String
)

object LLMCatalog {

    @JvmStatic
    val models: Map<String, LLMEntry> = mapOf(
        "llama-3-8b" to LLMEntry(
            name = "llama-3-8b",
            huggingFaceId = "meta-llama/Meta-Llama-3-8B",
            recipe = FinetuneRecipe.llama3(),
            description = "Meta Llama 3 8B base model"
        ),
        "llama-3-70b" to LLMEntry(
            name = "llama-3-70b",
            huggingFaceId = "meta-llama/Meta-Llama-3-70B",
            recipe = FinetuneRecipe.llama3(),
            description = "Meta Llama 3 70B base model"
        ),
        "mistral-7b-v0.3" to LLMEntry(
            name = "mistral-7b-v0.3",
            huggingFaceId = "mistralai/Mistral-7B-v0.3",
            recipe = FinetuneRecipe.mistral(),
            description = "Mistral 7B v0.3 base model"
        ),
        "qwen2.5-7b" to LLMEntry(
            name = "qwen2.5-7b",
            huggingFaceId = "Qwen/Qwen2.5-7B",
            recipe = FinetuneRecipe.qwen2(),
            description = "Qwen 2.5 7B base model"
        ),
        "gemma-2-9b" to LLMEntry(
            name = "gemma-2-9b",
            huggingFaceId = "google/gemma-2-9b",
            recipe = FinetuneRecipe.gemma(),
            description = "Google Gemma 2 9B base model"
        ),
        "phi-3.5-mini" to LLMEntry(
            name = "phi-3.5-mini",
            huggingFaceId = "microsoft/Phi-3.5-mini-instruct",
            recipe = FinetuneRecipe.phi3(),
            description = "Microsoft Phi 3.5 Mini Instruct model"
        ),
        "deepseek-v2-lite" to LLMEntry(
            name = "deepseek-v2-lite",
            huggingFaceId = "deepseek-ai/DeepSeek-V2-Lite",
            recipe = FinetuneRecipe.deepseek(),
            description = "DeepSeek V2 Lite base model"
        )
    )

    @JvmStatic
    fun getRecipe(modelName: String): FinetuneRecipe? {
        return models[modelName]?.recipe
    }

    @JvmStatic
    fun getHuggingFaceId(modelName: String): String? {
        return models[modelName]?.huggingFaceId
    }

    @JvmStatic
    fun listModels(): List<String> {
        return models.keys.toList()
    }
}
