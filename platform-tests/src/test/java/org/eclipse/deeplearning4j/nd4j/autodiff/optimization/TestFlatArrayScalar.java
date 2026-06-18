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
package org.eclipse.deeplearning4j.nd4j.autodiff.optimization;

import com.google.flatbuffers.FlatBufferBuilder;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.MethodSource;
import org.nd4j.autodiff.samediff.serde.FlatBuffersMapper;
import org.nd4j.autodiff.samediff.serde.SameDiffSerializer;
import org.nd4j.common.tests.tags.TagNames;
import org.nd4j.graph.FlatArray;
import org.nd4j.linalg.BaseNd4jTestWithBackends;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.factory.Nd4jBackend;

import java.nio.ByteBuffer;

import static org.junit.Assert.*;

/**
 * Debug test to understand FlatArray scalar serialization.
 */
@Tag(TagNames.DL4J_OLD_API)
public class TestFlatArrayScalar extends BaseNd4jTestWithBackends {

    @Override
    public char ordering() {
        return 'c';
    }

    @Override
    public long getTimeoutMilliseconds() {
        return 1_000_000_000L;
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testScalarToFlatArray(Nd4jBackend nd4jBackend) {
        // Create a scalar with value 1.0
        INDArray scalar = Nd4j.scalar(1.0);
        System.out.println("Original scalar value: " + scalar.getDouble(0));
        System.out.println("Original scalar shape: " + java.util.Arrays.toString(scalar.shape()));
        System.out.println("Original scalar rank: " + scalar.rank());
        System.out.println("Original scalar dataType: " + scalar.dataType());

        // Serialize using toFlatArray
        FlatBufferBuilder builder = new FlatBufferBuilder(1024);
        int flatArrayOffset = scalar.toFlatArray(builder);
        builder.finish(flatArrayOffset);

        // Get the serialized bytes and create a new FlatArray
        ByteBuffer bb = builder.dataBuffer();
        FlatArray fa = FlatArray.getRootAsFlatArray(bb);

        System.out.println("\n=== After toFlatArray serialization ===");
        System.out.println("FlatArray shapeLength: " + fa.shapeLength());
        System.out.println("FlatArray dtype: " + fa.dtype());
        ByteBuffer bufferBB = fa.bufferAsByteBuffer();
        System.out.println("FlatArray bufferAsByteBuffer: " + bufferBB);
        if (bufferBB != null) {
            System.out.println("FlatArray buffer remaining: " + bufferBB.remaining());
        }

        // Deserialize
        INDArray restored = Nd4j.createFromFlatArray(fa);
        System.out.println("\nRestored scalar value: " + restored.getDouble(0));
        System.out.println("Restored scalar shape: " + java.util.Arrays.toString(restored.shape()));
        System.out.println("Restored scalar rank: " + restored.rank());

        assertEquals("Scalar value should be preserved", 1.0, restored.getDouble(0), 1e-5);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testScalarSerializeSmallNdArray(Nd4jBackend nd4jBackend) throws Exception {
        // Create a scalar with value 1.0
        INDArray scalar = Nd4j.scalar(1.0);
        System.out.println("Original scalar value: " + scalar.getDouble(0));
        System.out.println("Original scalar dataType: " + scalar.dataType());

        // Serialize using serializeSmallNdArrayToFlatBuffer
        FlatBufferBuilder builder = new FlatBufferBuilder(1024);
        int flatArrayOffset = SameDiffSerializer.serializeSmallNdArrayToFlatBuffer(scalar, builder);
        builder.finish(flatArrayOffset);

        // Get the serialized bytes and create a new FlatArray
        ByteBuffer bb = builder.dataBuffer();
        FlatArray fa = FlatArray.getRootAsFlatArray(bb);

        System.out.println("\n=== After serializeSmallNdArrayToFlatBuffer ===");
        System.out.println("FlatArray shapeLength: " + fa.shapeLength());
        System.out.println("FlatArray dtype: " + fa.dtype());
        ByteBuffer bufferBB = fa.bufferAsByteBuffer();
        System.out.println("FlatArray bufferAsByteBuffer: " + bufferBB);
        if (bufferBB != null) {
            System.out.println("FlatArray buffer remaining: " + bufferBB.remaining());
            if (bufferBB.remaining() >= 8) {
                // Try to read as double
                bufferBB.position(0);
                double value = bufferBB.getDouble();
                System.out.println("Buffer double value: " + value);
            }
        }

        // Try to restore using the same logic as fromFlatNode
        INDArray restored = null;
        if (fa.shapeLength() == 0) {
            System.out.println("\nDetected scalar (shapeLength == 0)");
            byte dtype = fa.dtype();
            System.out.println("dtype byte: " + dtype);
            DataType dataType = FlatBuffersMapper.getDataTypeFromByte(dtype);
            System.out.println("DataType: " + dataType);

            ByteBuffer scalarBb = fa.bufferAsByteBuffer();
            if (scalarBb != null && scalarBb.remaining() > 0) {
                // DO NOT reset position - bufferAsByteBuffer() already positions at data start
                System.out.println("Buffer capacity: " + scalarBb.capacity());
                System.out.println("Buffer position: " + scalarBb.position());
                System.out.println("Buffer limit: " + scalarBb.limit());
                System.out.println("Buffer remaining: " + scalarBb.remaining());
                switch (dataType) {
                    case DOUBLE:
                        double val = scalarBb.getDouble();
                        System.out.println("Read double value: " + val);
                        restored = Nd4j.scalar(val);
                        break;
                    case FLOAT:
                        restored = Nd4j.scalar(scalarBb.getFloat());
                        break;
                    default:
                        System.out.println("Unhandled type: " + dataType);
                }
            } else {
                System.out.println("Buffer is null or empty!");
            }
        } else {
            System.out.println("\nUsing standard createFromFlatArray (shapeLength: " + fa.shapeLength() + ")");
            restored = Nd4j.createFromFlatArray(fa);
        }

        if (restored != null) {
            System.out.println("\nRestored scalar value: " + restored.getDouble(0));
            assertEquals("Scalar value should be preserved", 1.0, restored.getDouble(0), 1e-5);
        } else {
            fail("Failed to restore scalar");
        }
    }
}
