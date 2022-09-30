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
package org.nd4j.linalg.jcublas.compression;


import org.apache.commons.math3.util.FastMath;
import org.bytedeco.javacpp.IntPointer;
import org.bytedeco.javacpp.Pointer;
import org.nd4j.compression.impl.AbstractCompressor;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.buffer.DataTypeEx;
import org.nd4j.linalg.api.concurrency.AffinityManager.Location;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.reduce.longer.MatchCondition;
import org.nd4j.linalg.compression.CompressedDataBuffer;
import org.nd4j.linalg.compression.CompressionDescriptor;
import org.nd4j.linalg.compression.CompressionType;
import org.nd4j.linalg.exception.ND4JIllegalStateException;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.conditions.Conditions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class CudaThreshold extends AbstractCompressor {
    private static final Logger log = LoggerFactory.getLogger(CudaThreshold.class);
    protected float threshold = 0.001F;

    public CudaThreshold() {
    }

    public String getDescriptor() {
        return "THRESHOLD";
    }

    public void configure(Object... vars) {
        if (vars[0] instanceof Number) {
            Number t = (Number)vars[0];
            this.threshold = FastMath.abs(t.floatValue());
            log.info("Setting threshold to [{}]", this.threshold);
        } else {
            throw new ND4JIllegalStateException("Threshold value should be Number");
        }
    }

    public INDArray compress(INDArray array) {
        Nd4j.getExecutioner().commit();
        Nd4j.getAffinityManager().ensureLocation(array, Location.HOST);
        DataBuffer buffer = this.compress(array.data());
        if (buffer == null) {
            return null;
        } else {
            INDArray dup = Nd4j.createArrayFromShapeBuffer(buffer, array.shapeInfoDataBuffer());
            dup.markAsCompressed(true);
            return dup;
        }
    }

    public CompressionType getCompressionType() {
        return CompressionType.LOSSLESS;
    }

    public DataBuffer decompress(DataBuffer buffer, DataType dataType) {
        DataBuffer result = Nd4j.getNDArrayFactory().convertDataEx(DataTypeEx.THRESHOLD, buffer, this.getGlobalTypeEx());
        return result;
    }

    public DataBuffer compress(DataBuffer buffer) {
        INDArray temp = Nd4j.createArrayFromShapeBuffer(buffer, (DataBuffer)Nd4j.getShapeInfoProvider().createShapeInformation(new long[]{1L, buffer.length()}, buffer.dataType()).getFirst());
        MatchCondition condition = new MatchCondition(temp, Conditions.absGreaterThanOrEqual(this.threshold), new int[0]);
        int cntAbs = Nd4j.getExecutioner().exec(condition).getInt(new int[]{0});
        if (cntAbs < 2) {
            return null;
        } else {
            long originalLength = buffer.length() * (long)Nd4j.sizeOfDataType(buffer.dataType());
            int compressedLength = cntAbs + 4;
            IntPointer pointer = new IntPointer(compressedLength);
            pointer.put(0L, cntAbs);
            pointer.put(1L, (int)buffer.length());
            pointer.put(2L, Float.floatToIntBits(this.threshold));
            pointer.put(3L, 0);
            CompressionDescriptor descriptor = new CompressionDescriptor();
            descriptor.setCompressedLength(compressedLength * 4);
            descriptor.setOriginalLength(originalLength);
            descriptor.setOriginalElementSize((long)Nd4j.sizeOfDataType(buffer.dataType()));
            descriptor.setNumberOfElements(buffer.length());
            descriptor.setCompressionAlgorithm(this.getDescriptor());
            descriptor.setCompressionType(this.getCompressionType());
            CompressedDataBuffer cbuff = new CompressedDataBuffer(pointer, descriptor);
            Nd4j.getNDArrayFactory().convertDataEx(getBufferTypeEx(buffer), buffer.addressPointer(), DataTypeEx.THRESHOLD, pointer, buffer.length());
            Nd4j.getAffinityManager().tagLocation(buffer, Location.HOST);
            return cbuff;
        }
    }

    protected CompressedDataBuffer compressPointer(DataTypeEx srcType, Pointer srcPointer, int length, int elementSize) {
        throw new UnsupportedOperationException();
    }

    public float getThreshold() {
        return this.threshold;
    }

    public void setThreshold(float threshold) {
        this.threshold = threshold;
    }
}