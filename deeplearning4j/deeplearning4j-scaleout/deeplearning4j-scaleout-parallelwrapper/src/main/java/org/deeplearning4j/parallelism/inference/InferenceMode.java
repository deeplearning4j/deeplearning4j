/*******************************************************************************
 * Copyright (c) 2015-2018 Skymind, Inc.
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

package org.deeplearning4j.parallelism.inference;

/**
 * @author raver119@gmail.com
 */
public enum InferenceMode {
    /**
     * input will be passed into the model as is
     */
    SEQUENTIAL,

    /**
     * input will be included into the batch if computation device is busy, and executed immediately otherwise
     */
    BATCHED,

    /**
     * Inference will applied in the calling thread instead of workers.
     */
    INPLACE,
}
