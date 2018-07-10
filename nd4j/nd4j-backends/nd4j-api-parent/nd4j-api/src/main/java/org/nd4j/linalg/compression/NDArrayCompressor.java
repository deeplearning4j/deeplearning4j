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

import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.ndarray.INDArray;

/**
 *
 * NDArray compressor.
 * Given a compression algorithm,
 * it can compress/decompress
 * databuffers and ndarrays.
 *
 * @author raver119@gmail.com
 */
public interface NDArrayCompressor {

    /**
     * This method returns compression descriptor.
     * It should be
     * unique for any compressor implementation
     * @return
     */
    String getDescriptor();

    /**
     * This method allows to pass compressor-dependent configuration options to this compressor
     *
     * PLEASE NOTE: Each compressor has own options, please check corresponding implementations javadoc
     * @param vars
     */
    void configure(Object... vars);

    /**
     * This method returns compression opType provided
     * by specific NDArrayCompressor implementation
     * @return
     */
    CompressionType getCompressionType();

    /**
     * This method returns compressed copy of referenced array
     *
     * @param array
     * @return
     */
    INDArray compress(INDArray array);

    /**
     * Inplace compression of INDArray
     *
     * @param array
     */
    void compressi(INDArray array);

    /**
     *
     * @param buffer
     * @return
     */
    DataBuffer compress(DataBuffer buffer);

    /**
     * This method returns
     * decompressed copy of referenced array
     *
     * @param array
     * @return
     */
    INDArray decompress(INDArray array);

    /**
     * Inplace decompression of INDArray
     *
     * @param array
     */
    void decompressi(INDArray array);

    /**
     * Return a compressed databuffer
     * @param buffer the buffer to decompress
     * @return the decompressed data buffer
     */
    DataBuffer decompress(DataBuffer buffer);

    /**
     * This method creates compressed INDArray from Java float array, skipping usual INDArray instantiation routines
     * Please note: This method compresses input data as vector
     *
     * @param data
     * @return
     */
    INDArray compress(float[] data);

    /**
     * This method creates compressed INDArray from Java double array, skipping usual INDArray instantiation routines
     * Please note: This method compresses input data as vector
     *
     * @param data
     * @return
     */
    INDArray compress(double[] data);

    /**
     * This method creates compressed INDArray from Java float array, skipping usual INDArray instantiation routines
     *
     * @param data
     * @param shape
     * @return
     */
    INDArray compress(float[] data, int[] shape, char order);

    /**
     * This method creates compressed INDArray from Java double array, skipping usual INDArray instantiation routines
     *
     * @param data
     * @param shape
     * @return
     */
    INDArray compress(double[] data, int[] shape, char order);
}
