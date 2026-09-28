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
package org.eclipse.deeplearning4j.llm.generation;

/**
 * Source of draft tokens for native speculative decoding. Verification is the same
 * for every source (lossless: the target model's greedy tokens are emitted), so the
 * choice only affects how many drafts are accepted per verification step.
 */
public enum SpeculativeDrafter {
    /** The model's bundled MTP predictor when present, otherwise the n-gram proposer. */
    AUTO,
    /** The bundled MTP predictor (a preparation error if the model has none). */
    MTP,
    /** The n-gram proposer over the verified output (no draft model forward passes). */
    NGRAM
}
