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

import lombok.NonNull;
import lombok.extern.slf4j.Slf4j;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.util.Map;
import java.util.ServiceLoader;
import java.util.Set;
import java.util.concurrent.ConcurrentHashMap;

/**
 * @author raver119@gmail.com
 */
@Slf4j
public class BasicNDArrayCompressor {
    private static final BasicNDArrayCompressor INSTANCE = new BasicNDArrayCompressor();

    protected Map<String, NDArrayCompressor> codecs;

    protected String defaultCompression = "FLOAT16";

    private BasicNDArrayCompressor() {
        loadCompressors();
    }

    protected void loadCompressors() {
        /*
            We scan classpath for NDArrayCompressor implementations and add them one by one to codecs map
         */
        codecs = new ConcurrentHashMap<>();

        ServiceLoader<NDArrayCompressor> loader = ServiceLoader.load(NDArrayCompressor.class);
        for (NDArrayCompressor compressor : loader) {
            codecs.put(compressor.getDescriptor().toUpperCase(), compressor);
        }

        if(codecs.isEmpty()){
            //No compressors found - bad uber-jar?
            String msg = "Error loading ND4J Compressors via service loader: No compressors were found. This usually occurs" +
                    " when running ND4J UI from an uber-jar, which was built incorrectly (without services resource" +
                    " files being included)";
            log.error(msg);
            throw new RuntimeException(msg);
        }
    }

    /**
     * Get the set of available codecs for
     * compression
     * @return
     */
    public Set<String> getAvailableCompressors() {
        return codecs.keySet();
    }

    /**
     * Prints available compressors to standard out
     */
    public void printAvailableCompressors() {
        StringBuilder builder = new StringBuilder();
        builder.append("Available compressors: ");
        for (String comp : codecs.keySet()) {
            builder.append("[").append(comp).append("] ");
        }

        System.out.println(builder.toString());
    }

    /**
     * Get the ndarray compressor
     * singleton
     * @return
     */
    public static BasicNDArrayCompressor getInstance() {
        return INSTANCE;
    }

    /**
     * Set the default compression algorithm
     * @param algorithm the algorithm to set
     * @return
     */
    public BasicNDArrayCompressor setDefaultCompression(@NonNull String algorithm) {
        algorithm = algorithm.toUpperCase();
        //       if (!codecs.containsKey(algorithm))
        //            throw new RuntimeException("Non-existent compression algorithm requested: [" + algorithm + "]");

        synchronized (this) {
            defaultCompression = algorithm;
        }

        return this;
    }

    /**
     * Get the default compression algorithm as a string.
     * This is an all caps algorithm with a representation in the
     * CompressionAlgorithm enum
     * @return
     */
    public String getDefaultCompression() {
        synchronized (this) {
            return defaultCompression;
        }
    }

    /**
     * Compress the given data buffer
     * given the default compression algorithm
     * @param buffer the data buffer to compress
     * @return the compressed version of the data buffer
     */
    public DataBuffer compress(DataBuffer buffer) {
        return compress(buffer, getDefaultCompression());
    }

    /**
     * Compress the data buffer
     * given a specified algorithm
     * @param buffer the buffer to compress
     * @param algorithm the algorithm to compress
     * use
     * @return the compressed data buffer
     */
    public DataBuffer compress(DataBuffer buffer, String algorithm) {
        algorithm = algorithm.toUpperCase();
        if (!codecs.containsKey(algorithm))
            throw new RuntimeException("Non-existent compression algorithm requested: [" + algorithm + "]");

        return codecs.get(algorithm).compress(buffer);
    }

    public INDArray compress(INDArray array) {
        Nd4j.getExecutioner().commit();

        return compress(array, getDefaultCompression());
    }

    /**
     * In place compression of the passed in ndarray
     * with the default compression algorithm
     * @param array
     */
    public void compressi(INDArray array) {

        compressi(array, getDefaultCompression());
    }


    /**
     * Returns a compressed version of the
     * given ndarray
     * @param array the array to compress
     * @param algorithm the algorithm to compress with
     * @return a compressed copy of this ndarray
     */
    public INDArray compress(INDArray array, String algorithm) {
        algorithm = algorithm.toUpperCase();
        if (!codecs.containsKey(algorithm))
            throw new RuntimeException("Non-existent compression algorithm requested: [" + algorithm + "]");

        return codecs.get(algorithm).compress(array);
    }

    /**
     * In place Compress the given ndarray
     * with the given algorithm
     * @param array the array to compress
     * @param algorithm
     */
    public void compressi(INDArray array, String algorithm) {
        algorithm = algorithm.toUpperCase();
        if (!codecs.containsKey(algorithm))
            throw new RuntimeException("Non-existent compression algorithm requested: [" + algorithm + "]");

        codecs.get(algorithm).compressi(array);
    }

    /**
     * Decompress the given databuffer
     * @param buffer the databuffer to compress
     * @return the decompressed databuffer
     */
    public DataBuffer decompress(DataBuffer buffer) {
        if (buffer.dataType() != DataType.COMPRESSED)
            throw new IllegalStateException("You can't decompress DataBuffer with dataType of: " + buffer.dataType());

        CompressedDataBuffer comp = (CompressedDataBuffer) buffer;
        CompressionDescriptor descriptor = comp.getCompressionDescriptor();

        if (!codecs.containsKey(descriptor.getCompressionAlgorithm()))
            throw new RuntimeException("Non-existent compression algorithm requested: ["
                            + descriptor.getCompressionAlgorithm() + "]");

        return codecs.get(descriptor.getCompressionAlgorithm()).decompress(buffer);
    }

    public NDArrayCompressor getCompressor(@NonNull String name) {
        return codecs.get(name);
    }

    /**
     *
     * @param array
     * @return
     */
    public INDArray decompress(INDArray array) {
        if (array.data().dataType() != DataType.COMPRESSED)
            return array;

        CompressedDataBuffer comp = (CompressedDataBuffer) array.data();
        CompressionDescriptor descriptor = comp.getCompressionDescriptor();

        if (!codecs.containsKey(descriptor.getCompressionAlgorithm()))
            throw new RuntimeException("Non-existent compression algorithm requested: ["
                            + descriptor.getCompressionAlgorithm() + "]");

        return codecs.get(descriptor.getCompressionAlgorithm()).decompress(array);
    }

    /**
     * in place decompression of the given
     * ndarray. If the ndarray isn't compressed
     * this will do nothing
     * @param array the array to decompressed
     *              if it is comprssed
     */
    public void decompressi(INDArray array) {
        if (array.data().dataType() != DataType.COMPRESSED)
            return;

        CompressedDataBuffer comp = (CompressedDataBuffer) array.data();
        CompressionDescriptor descriptor = comp.getCompressionDescriptor();

        if (!codecs.containsKey(descriptor.getCompressionAlgorithm()))
            throw new RuntimeException("Non-existent compression algorithm requested: ["
                            + descriptor.getCompressionAlgorithm() + "]");

        codecs.get(descriptor.getCompressionAlgorithm()).decompressi(array);
    }

    /**
     * Decompress several ndarrays
     * @param arrays
     */
    public void autoDecompress(INDArray... arrays) {
        for (INDArray array : arrays) {
            autoDecompress(array);
        }
    }

    /**
     *
     * @param array
     */
    public void autoDecompress(INDArray array) {
        if (array.isCompressed())
            decompressi(array);
    }

    /**
     * This method returns compressed INDArray instance which contains JVM array passed in
     *
     * @param array
     * @return
     */
    public INDArray compress(float[] array) {
        return codecs.get(defaultCompression).compress(array);
    }

    /**
     * This method returns compressed INDArray instance which contains JVM array passed in
     *
     * @param array
     * @return
     */
    public INDArray compress(double[] array) {
        return codecs.get(defaultCompression).compress(array);
    }
}
