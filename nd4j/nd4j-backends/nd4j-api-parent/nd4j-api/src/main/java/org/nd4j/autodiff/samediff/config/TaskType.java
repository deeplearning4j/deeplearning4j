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

package org.nd4j.autodiff.samediff.config;

/**
 * Task types for PEFT fine-tuning.
 * Different task types may require different handling of inputs/outputs
 * and different default configurations.
 *
 * @author Adam Gibson
 * @see PeftConfig
 */
public enum TaskType {

    /**
     * Sequence-to-sequence tasks (translation, summarization).
     */
    SEQ_2_SEQ_LM,

    /**
     * Causal/autoregressive language modeling.
     */
    CAUSAL_LM,

    /**
     * Token classification (NER, POS tagging).
     */
    TOKEN_CLS,

    /**
     * Sequence classification (sentiment, topic classification).
     */
    SEQ_CLS,

    /**
     * Question answering tasks.
     */
    QUESTION_ANS,

    /**
     * Feature extraction (embeddings).
     */
    FEATURE_EXTRACTION,

    /**
     * Image classification.
     */
    IMAGE_CLS,

    /**
     * Image-to-text tasks (captioning, OCR).
     */
    IMAGE_TO_TEXT,

    /**
     * Object detection.
     */
    OBJECT_DETECTION,

    /**
     * Image segmentation.
     */
    SEGMENTATION,

    /**
     * Speech recognition (ASR).
     */
    SPEECH_RECOGNITION,

    /**
     * Multi-modal tasks.
     */
    MULTI_MODAL,

    /**
     * Regression tasks.
     */
    REGRESSION,

    /**
     * General/custom task type.
     */
    GENERAL
}
