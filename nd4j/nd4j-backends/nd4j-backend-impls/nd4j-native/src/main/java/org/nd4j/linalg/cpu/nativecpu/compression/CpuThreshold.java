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

package org.nd4j.linalg.cpu.nativecpu.compression;

import lombok.Getter;
import lombok.Setter;
import lombok.extern.slf4j.Slf4j;
import org.apache.commons.math3.util.FastMath;
import org.bytedeco.javacpp.IntPointer;
import org.bytedeco.javacpp.Pointer;
import org.nd4j.compression.impl.AbstractCompressor;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.buffer.DataTypeEx;
import org.nd4j.linalg.api.concurrency.AffinityManager;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.accum.longer.MatchCondition;
import org.nd4j.linalg.compression.CompressedDataBuffer;
import org.nd4j.linalg.compression.CompressionDescriptor;
import org.nd4j.linalg.compression.CompressionType;
import org.nd4j.linalg.exception.ND4JIllegalStateException;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.conditions.Conditions;

/**
 * This compression is very special case, and shouldn't be ever used outside of ParallelWrapper/ParameterServer implementation.
 * It encodes data as delta between zero and abs threshold.
 *
 * PLEASE NOTE: DO NOT USE THIS COMPRESSOR UNLESS YOU'RE 100% SURE WHAT YOU DO!
 *
 * @author raver119@gmail.com
 */
@Slf4j
public class CpuThreshold extends AbstractCompressor {
    @Getter @Setter protected float threshold = 1e-3f;

    /**
     * This method returns compression descriptor. It should be unique for any compressor implementation
     *
     * @return
     */
    @Override
    public String getDescriptor() {
        return "THRESHOLD";
    }

    /**
     * This method allows you to configure threshold for delta extraction. Pass it as float/double value
     *
     * Default value: 1e-3
     * @param vars
     */
    @Override
    public void configure(Object... vars) {
        if (vars[0] instanceof Number) {
            Number t = (Number) vars[0];
            threshold = FastMath.abs(t.floatValue());
            log.info("Setting threshold to [{}]", threshold);
        } else {
            throw new ND4JIllegalStateException("Threshold value should be Number");
        }
    }

    @Override
    public INDArray compress(INDArray array) {
        //logger.info("Threshold [{}] compression", threshold);

        Nd4j.getExecutioner().commit();
        Nd4j.getAffinityManager().ensureLocation(array, AffinityManager.Location.HOST);

        DataBuffer buffer = compress(array.data());
        if (buffer == null)
            return null;

        INDArray dup = Nd4j.createArrayFromShapeBuffer(buffer, array.shapeInfoDataBuffer());
        dup.markAsCompressed(true);

        return dup;
    }

    @Override
    public CompressionType getCompressionType() {
        return CompressionType.LOSSLESS;
    }

    @Override
    public DataBuffer decompress(DataBuffer buffer) {


        DataBuffer result = Nd4j.getNDArrayFactory().convertDataEx(DataTypeEx.THRESHOLD, buffer, getGlobalTypeEx());

        return result;
    }

    @Override
    public DataBuffer compress(DataBuffer buffer) {
        INDArray temp = Nd4j.createArrayFromShapeBuffer(buffer, Nd4j.getShapeInfoProvider().createShapeInformation(new long[]{1, buffer.length()}, DataType.INT).getFirst());
        MatchCondition condition = new MatchCondition(temp, Conditions.absGreaterThanOrEqual(threshold));
        int cntAbs = Nd4j.getExecutioner().exec(condition, Integer.MAX_VALUE).getInt(0);


        //log.info("density ratio: {}", String.format("%.2f", cntAbs * 100.0f / buffer.length()));

        if (cntAbs < 2)
            return null;

        long originalLength = buffer.length() * Nd4j.sizeOfDataType(buffer.dataType());
        int compressedLength = cntAbs + 4;
        // first 3 elements contain header
        IntPointer pointer = new IntPointer(compressedLength);
        pointer.put(0, cntAbs);
        pointer.put(1, (int) buffer.length());
        pointer.put(2, Float.floatToIntBits(threshold));
        pointer.put(3, 0);

        CompressionDescriptor descriptor = new CompressionDescriptor();
        descriptor.setCompressedLength(compressedLength * 4); // sizeOf(INT)
        descriptor.setOriginalLength(originalLength);
        descriptor.setOriginalElementSize(Nd4j.sizeOfDataType(buffer.dataType()));
        descriptor.setNumberOfElements(buffer.length());

        descriptor.setCompressionAlgorithm(getDescriptor());
        descriptor.setCompressionType(getCompressionType());



        CompressedDataBuffer cbuff = new CompressedDataBuffer(pointer, descriptor);

        Nd4j.getNDArrayFactory().convertDataEx(getBufferTypeEx(buffer), buffer.addressPointer(), DataTypeEx.THRESHOLD, pointer, buffer.length());

        Nd4j.getAffinityManager().tagLocation(buffer, AffinityManager.Location.HOST);

        return cbuff;
    }

    @Override
    protected CompressedDataBuffer compressPointer(DataTypeEx srcType, Pointer srcPointer, int length, int elementSize) {
        throw new UnsupportedOperationException();
    }
}
