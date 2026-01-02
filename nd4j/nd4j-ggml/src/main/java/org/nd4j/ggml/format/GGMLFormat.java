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

package org.nd4j.ggml.format;

/**
 * Enumeration of supported GGML file formats.
 */
public enum GGMLFormat {
    /**
     * Original GGML format (legacy).
     * The earliest format, with minimal header.
     */
    GGML,

    /**
     * GGMF format (legacy).
     * Adds version information to the header.
     */
    GGMF,

    /**
     * GGJT format (legacy).
     * Adds more parameters and tensor alignment.
     */
    GGJT,

    /**
     * Modern GGUF format (GGML Universal Format).
     * Introduced in llama.cpp in August 2023.
     * Features key-value metadata and improved extensibility.
     */
    GGUF
}
