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

package org.nd4j.serde.binary;

import lombok.extern.slf4j.Slf4j;
import lombok.val;
import org.bytedeco.javacpp.BytePointer;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.concurrency.AffinityManager;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.shape.Shape;
import org.nd4j.linalg.compression.CompressedDataBuffer;
import org.nd4j.linalg.compression.CompressionDescriptor;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.primitives.Pair;

import java.io.*;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.channels.Channels;
import java.nio.channels.FileChannel;
import java.nio.channels.WritableByteChannel;

/**
 * Created by agibsonccc on 7/1/17.
 */
@Slf4j
public class BinarySerde {


    /**
     * Create an ndarray
     * from the unsafe buffer
     * @param buffer the buffer to create the array from
     * @return the ndarray derived from this buffer
     */
    public static INDArray toArray(ByteBuffer buffer, int offset) {
        return toArrayAndByteBuffer(buffer, offset).getLeft();
    }

    /**
     * Create an ndarray
     * from the unsafe buffer
     * @param buffer the buffer to create the array from
     * @return the ndarray derived from this buffer
     */
    public static INDArray toArray(ByteBuffer buffer) {
        return toArray(buffer, 0);
    }



    /**
     * Create an ndarray and existing bytebuffer
     * @param buffer
     * @param offset
     * @return
     */
    public static Pair<INDArray, ByteBuffer> toArrayAndByteBuffer(ByteBuffer buffer, int offset) {
        ByteBuffer byteBuffer = buffer == null ? ByteBuffer.allocateDirect(buffer.array().length).put(buffer.array())
                .order(ByteOrder.nativeOrder()) : buffer.order(ByteOrder.nativeOrder());
        //bump the byte buffer to the proper position
        byteBuffer.position(offset);
        int rank = byteBuffer.getInt();
        if (rank < 0)
            throw new IllegalStateException("Found negative integer. Corrupt serialization?");
        //get the shape buffer length to create the shape information buffer
        int shapeBufferLength = Shape.shapeInfoLength(rank);
        //create the ndarray shape information
        DataBuffer shapeBuff = Nd4j.createBufferDetached(new int[shapeBufferLength]);

        //compute the databuffer opType from the index
        DataType type = DataType.values()[byteBuffer.getInt()];
        for (int i = 0; i < shapeBufferLength; i++) {
            shapeBuff.put(i, byteBuffer.getLong());
        }

        //after the rank,data opType, shape buffer (of length shape buffer length) * sizeof(int)
        if (type != DataType.COMPRESSED) {
            ByteBuffer slice = byteBuffer.slice();
            //wrap the data buffer for the last bit
            // FIXME: int cast
            DataBuffer buff = Nd4j.createBuffer(slice, type, (int) Shape.length(shapeBuff));
            //advance past the data
            int position = byteBuffer.position() + (buff.getElementSize() * (int) buff.length());
            byteBuffer.position(position);
            //create the final array
            //TODO: see how to avoid dup here
            INDArray arr = Nd4j.createArrayFromShapeBuffer(buff.dup(), shapeBuff.dup());
            return Pair.of(arr, byteBuffer);
        } else {
            CompressionDescriptor compressionDescriptor = CompressionDescriptor.fromByteBuffer(byteBuffer);
            ByteBuffer slice = byteBuffer.slice();
            //ensure that we only deal with the slice of the buffer that is actually the data
            BytePointer byteBufferPointer = new BytePointer(slice);
            //create a compressed array based on the rest of the data left in the buffer
            CompressedDataBuffer compressedDataBuffer =
                    new CompressedDataBuffer(byteBufferPointer, compressionDescriptor);
            //TODO: see how to avoid dup()
            INDArray arr = Nd4j.createArrayFromShapeBuffer(compressedDataBuffer.dup(), shapeBuff.dup());
            //advance past the data
            int compressLength = (int) compressionDescriptor.getCompressedLength();
            byteBuffer.position(byteBuffer.position() + compressLength);
            return Pair.of(arr, byteBuffer);
        }

    }


    /**
     * Convert an ndarray to an unsafe buffer
     * for use by aeron
     * @param arr the array to convert
     * @return the unsafebuffer representation of this array
     */
    public static ByteBuffer toByteBuffer(INDArray arr) {
        //subset and get rid of 1 off non 1 element wise stride cases
        if (arr.isView())
            arr = arr.dup();
        if (!arr.isCompressed()) {
            ByteBuffer b3 = ByteBuffer.allocateDirect(byteBufferSizeFor(arr)).order(ByteOrder.nativeOrder());
            doByteBufferPutUnCompressed(arr, b3, true);
            return b3;
        }
        //compressed array
        else {
            ByteBuffer b3 = ByteBuffer.allocateDirect(byteBufferSizeFor(arr)).order(ByteOrder.nativeOrder());
            doByteBufferPutCompressed(arr, b3, true);
            return b3;
        }

    }

    /**
     * Returns the byte buffer size for the given
     * ndarray. This is an auxillary method
     * for determining the size of the buffer
     * size to allocate for sending an ndarray via
     * the aeron media driver.
     *
     * The math break down for uncompressed is:
     * 2 ints for rank of the array and an ordinal representing the data opType of the data buffer
     * The rest is in order:
     * shape information
     * data buffer
     *
     * The math break down for compressed is:
     * 2 ints for rank and an ordinal representing the data opType for the data buffer
     *
     * The rest is in order:
     * shape information
     * codec information
     * data buffer
     *
     * @param arr the array to compute the size for
     * @return the size of the byte buffer that was allocated
     */
    public static int byteBufferSizeFor(INDArray arr) {
        if (!arr.isCompressed()) {
            ByteBuffer buffer = arr.data().pointer().asByteBuffer().order(ByteOrder.nativeOrder());
            ByteBuffer shapeBuffer = arr.shapeInfoDataBuffer().pointer().asByteBuffer().order(ByteOrder.nativeOrder());
            //2 four byte ints at the beginning
            int twoInts = 8;
            return twoInts + buffer.limit() + shapeBuffer.limit();
        } else {
            CompressedDataBuffer compressedDataBuffer = (CompressedDataBuffer) arr.data();
            CompressionDescriptor descriptor = compressedDataBuffer.getCompressionDescriptor();
            ByteBuffer codecByteBuffer = descriptor.toByteBuffer();
            ByteBuffer buffer = arr.data().pointer().asByteBuffer().order(ByteOrder.nativeOrder());
            ByteBuffer shapeBuffer = arr.shapeInfoDataBuffer().pointer().asByteBuffer().order(ByteOrder.nativeOrder());
            int twoInts = 2 * 4;
            return twoInts + buffer.limit() + shapeBuffer.limit() + codecByteBuffer.limit();
        }
    }



    /**
     * Setup the given byte buffer
     * for serialization (note that this is for uncompressed INDArrays)
     * 4 bytes int for rank
     * 4 bytes for data opType
     * shape buffer
     * data buffer
     *
     * @param arr the array to setup
     * @param allocated the byte buffer to setup
     * @param rewind whether to rewind the byte buffer or nt
     */
    public static void doByteBufferPutUnCompressed(INDArray arr, ByteBuffer allocated, boolean rewind) {
        // ensure we send data to host memory
        Nd4j.getExecutioner().commit();
        Nd4j.getAffinityManager().ensureLocation(arr, AffinityManager.Location.HOST);

        ByteBuffer buffer = arr.data().pointer().asByteBuffer().order(ByteOrder.nativeOrder());
        ByteBuffer shapeBuffer = arr.shapeInfoDataBuffer().pointer().asByteBuffer().order(ByteOrder.nativeOrder());
        //2 four byte ints at the beginning
        allocated.putInt(arr.rank());
        //put data opType next so its self describing
        allocated.putInt(arr.data().dataType().ordinal());
        allocated.put(shapeBuffer);
        allocated.put(buffer);
        if (rewind)
            allocated.rewind();
    }

    /**
     * Setup the given byte buffer
     * for serialization (note that this is for compressed INDArrays)
     * 4 bytes for rank
     * 4 bytes for data opType
     * shape information
     * codec information
     * data opType
     *
     * @param arr the array to setup
     * @param allocated the byte buffer to setup
     * @param rewind whether to rewind the byte buffer or not
     */
    public static void doByteBufferPutCompressed(INDArray arr, ByteBuffer allocated, boolean rewind) {
        CompressedDataBuffer compressedDataBuffer = (CompressedDataBuffer) arr.data();
        CompressionDescriptor descriptor = compressedDataBuffer.getCompressionDescriptor();
        ByteBuffer codecByteBuffer = descriptor.toByteBuffer();
        ByteBuffer buffer = arr.data().pointer().asByteBuffer().order(ByteOrder.nativeOrder());
        ByteBuffer shapeBuffer = arr.shapeInfoDataBuffer().pointer().asByteBuffer().order(ByteOrder.nativeOrder());
        allocated.putInt(arr.rank());
        //put data opType next so its self describing
        allocated.putInt(arr.data().dataType().ordinal());
        //put shape next
        allocated.put(shapeBuffer);
        //put codec information next
        allocated.put(codecByteBuffer);
        //finally put the data
        allocated.put(buffer);
        if (rewind)
            allocated.rewind();
    }


    /**
     * Write an array to an output stream.
     * @param arr the array to write
     * @param outputStream the output stream to write to
     */
    public static void writeArrayToOutputStream(INDArray arr, OutputStream outputStream) {
        ByteBuffer buffer = BinarySerde.toByteBuffer(arr);
        try (WritableByteChannel channel = Channels.newChannel(outputStream)) {
            channel.write(buffer);
        } catch (IOException e) {
            e.printStackTrace();
        }
    }


    /**
     * Write an ndarray to disk in
     * binary format
     * @param arr the array to write
     * @param toWrite the file tow rite to
     * @throws IOException
     */
    public static void writeArrayToDisk(INDArray arr, File toWrite) throws IOException {
        try (FileOutputStream os = new FileOutputStream(toWrite)) {
            FileChannel channel = os.getChannel();
            ByteBuffer buffer = BinarySerde.toByteBuffer(arr);
            channel.write(buffer);
        }
    }


    /**
     * Read an ndarray from disk
     * @param readFrom
     * @return
     * @throws IOException
     */
    public static INDArray readFromDisk(File readFrom) throws IOException {
        try (FileInputStream os = new FileInputStream(readFrom)) {
            FileChannel channel = os.getChannel();
            ByteBuffer buffer = ByteBuffer.allocateDirect((int) readFrom.length());
            channel.read(buffer);
            INDArray ret = toArray(buffer);
            return ret;
        }
    }


    /**
     * This method returns shape databuffer from saved earlier file
     *
     * @param readFrom
     * @return
     * @throws IOException
     */
    public static DataBuffer readShapeFromDisk(File readFrom) throws IOException {
        try (FileInputStream os = new FileInputStream(readFrom)) {
            FileChannel channel = os.getChannel();
            // we read shapeinfo up to max_rank value, which is 32
            int len = (int) Math.min((32 * 2 + 3) * 8, readFrom.length());
            ByteBuffer buffer = ByteBuffer.allocateDirect(len);
            channel.read(buffer);

            ByteBuffer byteBuffer = buffer == null ? ByteBuffer.allocateDirect(buffer.array().length)
                    .put(buffer.array()).order(ByteOrder.nativeOrder()) : buffer.order(ByteOrder.nativeOrder());

            buffer.position(0);
            int rank = byteBuffer.getInt();

            val result = new long[Shape.shapeInfoLength(rank)];

            // filling DataBuffer with shape info
            result[0] = rank;

            // skipping two next values (dtype and rank again)
            // please , that this time rank has dtype of LONG, so takes 8 bytes.
            byteBuffer.position(16);

            // filling shape information
            for (int e = 1; e < Shape.shapeInfoLength(rank); e++) {
                result[e] = byteBuffer.getLong();
            }

            // creating nd4j databuffer now
            DataBuffer dataBuffer = Nd4j.getDataBufferFactory().createLong(result);
            return dataBuffer;
        }
    }


}
