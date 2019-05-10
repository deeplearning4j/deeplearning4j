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

package org.nd4j.linalg.compression;

/**
 * Compression algorithm enum
 *
 * @author Adam Gibson
 */
public enum CompressionAlgorithm {
    FLOAT8, FLOAT16, GZIP, INT8, INT16, NOOP, UNIT8, CUSTOM;

    /**
     * Return the appropriate compression algorithm
     * from the given string
     * @param algorithm the algorithm to return
     * @return the compression algorithm from the given string
     * or an IllegalArgumentException if the algorithm is invalid
     */
    public static CompressionAlgorithm fromString(String algorithm) {
        switch (algorithm.toUpperCase()) {
            case "FP16":
                return FLOAT8;
            case "FP32":
                return FLOAT16;
            case "FLOAT8":
                return FLOAT8;
            case "FLOAT16":
                return FLOAT16;
            case "GZIP":
                return GZIP;
            case "INT8":
                return INT8;
            case "INT16":
                return INT16;
            case "NOOP":
                return NOOP;
            case "UNIT8":
                return UNIT8;
            case "CUSTOM":
                return CUSTOM;
            default:
                throw new IllegalArgumentException("Wrong algorithm " + algorithm);
        }
    }

}
