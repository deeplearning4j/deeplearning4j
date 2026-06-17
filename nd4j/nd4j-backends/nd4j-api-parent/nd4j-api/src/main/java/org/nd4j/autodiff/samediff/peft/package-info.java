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
 * Parameter-Efficient Fine-Tuning (PEFT) support for SameDiff.
 * <p>
 * This package provides PEFT methods for efficiently fine-tuning large models
 * with minimal parameter updates. Based on the Hugging Face PEFT library.
 * <p>
 * Supported methods:
 * <ul>
 *   <li><b>LoRA</b> - Low-Rank Adaptation via weight matrix decomposition</li>
 *   <li><b>Prompt Tuning</b> - Trainable soft prompt tokens added to input</li>
 *   <li><b>Prefix Tuning</b> - Prefix vectors added to all transformer layers</li>
 *   <li><b>Adapters</b> - Bottleneck layers inserted between transformer layers</li>
 *   <li><b>IA³</b> - Learned vectors for activation scaling</li>
 * </ul>
 *
 * <p>Quick start example:</p>
 * <pre>{@code
 * // Load pretrained model
 * SameDiff baseModel = SameDiff.load(new File("model.sd"));
 *
 * // Create LoRA configuration
 * LoraConfig config = LoraConfig.builder()
 *     .r(16)
 *     .loraAlpha(32)
 *     .targetModules(Arrays.asList("query", "key", "value"))
 *     .build();
 *
 * // Create PEFT model
 * PeftModel peftModel = PeftModel.fromPretrained(baseModel, config);
 *
 * // Check parameter efficiency
 * peftModel.printTrainableParameters();
 * // Output: trainable params: 294,912 || all params: 110,000,000 || trainable%: 0.2681
 *
 * // Train
 * peftModel.fit(trainingData, 3);
 *
 * // Save adapter
 * peftModel.saveAdapter(new File("my_lora_adapter"));
 *
 * // For deployment: merge adapter into base model
 * SameDiff mergedModel = peftModel.mergeAndUnload();
 * }</pre>
 *
 * @see org.nd4j.autodiff.samediff.peft.PeftModel
 * @see org.nd4j.autodiff.samediff.config.LoraConfig
 * @see org.nd4j.autodiff.samediff.config.PeftConfig
 * @see <a href="https://github.com/huggingface/peft">Hugging Face PEFT</a>
 */
package org.nd4j.autodiff.samediff.peft;
