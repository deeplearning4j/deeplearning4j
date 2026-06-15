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

/**
 * Multi-model pipeline framework for samediff-llm.
 *
 * <p>This package provides a modular framework for assembling multiple LLM models
 * into pipelines for complex text processing tasks. Similar to the VLM OCR pipeline
 * in samediff-vlm, it supports:</p>
 *
 * <ul>
 *   <li>Model role-based configuration (summarization, classification, sorting,
 *       relation extraction, language detection, NER)</li>
 *   <li>Built-in prompt templates for each model type</li>
 *   <li>Structured output parsing</li>
 *   <li>DSP (Dynamic Shape Plan) optimizations</li>
 *   <li>Memory leak protection and GPU memory management</li>
 *   <li>Model reuse detection and optimization</li>
 * </ul>
 *
 * <p>Key classes:</p>
 * <ul>
 *   <li>{@link org.eclipse.deeplearning4j.llm.pipeline.ModelRole} - Enumerates model roles</li>
 *   <li>{@link org.eclipse.deeplearning4j.llm.pipeline.ModelType} - Defines model types with prompts</li>
 *   <li>{@link org.eclipse.deeplearning4j.llm.pipeline.MultiModelPipeline} - Main pipeline executor</li>
 *   <li>{@link org.eclipse.deeplearning4j.llm.pipeline.MultiModelPipelineConfig} - Configuration</li>
 *   <li>{@link org.eclipse.deeplearning4j.llm.pipeline.PipelineStage} - Individual stage definition</li>
 *   <li>{@link org.eclipse.deeplearning4j.llm.pipeline.PipelineResult} - Execution results</li>
 * </ul>
 *
 * <p>Example usage:</p>
 * <pre>{@code
 * MultiModelPipelineConfig config = MultiModelPipelineConfig.builder()
 *     .tokenizer(tokenizer)
 *     .maxTokensPerModel(256)
 *     .dspEnabled(true)
 *     .trimMemoryBetweenStages(true)
 *     .build();
 *
 * try (MultiModelPipeline pipeline = new MultiModelPipeline(config)) {
 *     // Register models for different roles
 *     pipeline.registerModel("summarizer", ModelType.SUMMARIZER, summarizerModel);
 *     pipeline.registerModel("classifier", ModelType.CLASSIFIER, classifierModel);
 *     pipeline.registerModel("ner", ModelType.NER_MODEL, nerModel);
 *
 *     // Define pipeline stages
 *     List<PipelineStage> stages = Arrays.asList(
 *         new PipelineStage("summarizer", "summary"),
 *         new PipelineStage("classifier", "category"),
 *         new PipelineStage("ner", "entities")
 *     );
 *
 *     // Execute pipeline
 *     PipelineResult result = pipeline.execute(stages, "Long document text...");
 *
 *     // Access outputs
 *     String summary = result.getStageText("summary");
 *     String category = result.getStageText("category");
 *     List<ModelType.Entity> entities = (List<ModelType.Entity>)
 *         result.getStageStructuredData("entities");
 * }
 * }</pre>
 *
 * @see org.eclipse.deeplearning4j.llm.generation.GenerationPipeline
 * @see org.eclipse.deeplearning4j.llm.generation.SameDiffMemoryUtils
 */
package org.eclipse.deeplearning4j.llm.pipeline;
