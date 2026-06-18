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
 * Text generation utilities for Large Language Models.
 *
 * <p>This package provides reusable components for autoregressive text generation:</p>
 *
 * <h2>Core Classes</h2>
 * <ul>
 *   <li>{@link org.eclipse.deeplearning4j.llm.generation.TextGenerator} - High-level text generation API</li>
 *   <li>{@link org.eclipse.deeplearning4j.llm.generation.sampling.SamplingConfig} - Configuration for sampling strategies</li>
 *   <li>{@link org.eclipse.deeplearning4j.llm.generation.GenerationResult} - Generation output with metadata</li>
 * </ul>
 *
 * <h2>Sampling Strategies</h2>
 * <ul>
 *   <li>{@link org.eclipse.deeplearning4j.llm.generation.sampling.Sampler} - Interface for sampling strategies</li>
 *   <li>{@link org.eclipse.deeplearning4j.llm.generation.sampling.GreedySampler} - Deterministic argmax selection</li>
 *   <li>{@link org.eclipse.deeplearning4j.llm.generation.sampling.CompositeSampler} - Combined temperature/top-k/top-p sampling</li>
 * </ul>
 *
 * <h2>Utilities</h2>
 * <ul>
 *   <li>{@link org.eclipse.deeplearning4j.llm.generation.sampling.SamplerUtils} - Low-level sampling operations using Nd4j.math() and Nd4j.nn()</li>
 * </ul>
 *
 * <h2>Usage Example</h2>
 * <pre>{@code
 * // Create a generator with default settings
 * TextGenerator generator = TextGenerator.builder()
 *     .model(decoderModel)
 *     .tokenizer(tokenizer)
 *     .config(SamplingConfig.defaultConfig())
 *     .build();
 *
 * // Generate text
 * String output = generator.generate("Once upon a time");
 *
 * // Or with streaming
 * generator.generateStreaming("Once upon a time", token -> {
 *     System.out.print(token);
 * });
 *
 * // Custom sampling configuration
 * SamplingConfig creative = SamplingConfig.builder()
 *     .temperature(0.9)
 *     .topK(50)
 *     .topP(0.95)
 *     .build();
 * }</pre>
 *
 * <h2>Implementation Notes</h2>
 * <p>All operations use the ND4J namespace APIs for optimal performance:</p>
 * <ul>
 *   <li>{@code Nd4j.nn().softmax()} for probability distributions</li>
 *   <li>{@code Nd4j.math().exp()}, {@code Nd4j.math().log()} for numerical operations</li>
 *   <li>{@code Nd4j.argMax()} for greedy selection</li>
 * </ul>
 *
 * Adam Gibson
 */
package org.eclipse.deeplearning4j.llm.generation;
