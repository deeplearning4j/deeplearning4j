package org.nd4j.aeron.ipc;

import org.agrona.DirectBuffer;
import org.agrona.concurrent.UnsafeBuffer;
import org.bytedeco.javacpp.BytePointer;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.shape.Shape;
import org.nd4j.linalg.compression.CompressedDataBuffer;
import org.nd4j.linalg.compression.CompressionDescriptor;
import org.nd4j.linalg.factory.Nd4j;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;

/**
 * Serialization and de serialization class for
 * aeron.
 * This is a low level class specifically meant for speed.
 *
 * @author Adam Gibson
 */
public class AeronNDArraySerde {
    /**
     * Convert an ndarray to an unsafe buffer
     * for use by aeron
     * @param arr the array to convert
     * @return the unsafebuffer representation of this array
     */
    public static UnsafeBuffer toBuffer(INDArray arr) {
        //subset and get rid of 1 off non 1 element wise stride cases
        if(arr.isView()) arr = arr.dup();
        if(!arr.isCompressed()) {
            ByteBuffer buffer = arr.data().pointer().asByteBuffer().order(ByteOrder.nativeOrder());
            ByteBuffer shapeBuffer = arr.shapeInfoDataBuffer().pointer().asByteBuffer().order(ByteOrder.nativeOrder());
            //2 four byte ints at the beginning
            int twoInts = 8;
            ByteBuffer b3 = ByteBuffer.allocateDirect(twoInts + buffer.limit() + shapeBuffer.limit()).order(ByteOrder.nativeOrder());
            b3.putInt(arr.rank());
            //put data type next so its self describing
            b3.putInt(arr.data().dataType().ordinal());
            b3.put(shapeBuffer);
            b3.put(buffer);
            b3.rewind();
            return new UnsafeBuffer(b3);
        }
        //compressed array
        else {
            CompressedDataBuffer compressedDataBuffer = (CompressedDataBuffer) arr.data();
            CompressionDescriptor descriptor = compressedDataBuffer.getCompressionDescriptor();
            ByteBuffer codecByteBuffer = descriptor.toByteBuffer();
            ByteBuffer buffer = arr.data().pointer().asByteBuffer().order(ByteOrder.nativeOrder());
            ByteBuffer shapeBuffer = arr.shapeInfoDataBuffer().pointer().asByteBuffer().order(ByteOrder.nativeOrder());
            //1 four byte int at the beginning for the rank
            int twoInts = 2 * 4;
            ByteBuffer b3 = ByteBuffer.allocateDirect(twoInts + buffer.limit() + shapeBuffer.limit() + codecByteBuffer.limit()).order(ByteOrder.nativeOrder());
            b3.putInt(arr.rank());
            //put data type next so its self describing
            b3.putInt(arr.data().dataType().ordinal());
            //put shape next
            b3.put(shapeBuffer);
            //put codec information next
            b3.put(codecByteBuffer);
            //finally put the data
            b3.put(buffer);
            b3.rewind();
            return new UnsafeBuffer(b3);
        }

    }



    /**
     * Create an ndarray
     * from the unsafe buffer
     * @param buffer the buffer to create the array from
     * @return the ndarray derived from this buffer
     */
    public static INDArray toArray(DirectBuffer buffer,int offset) {
        ByteBuffer byteBuffer = buffer.byteBuffer().order(ByteOrder.nativeOrder());
        byteBuffer.position(offset);
        int rank = byteBuffer.getInt();
        //get the shape buffer length to create the shape information buffer
        int shapeBufferLength = Shape.shapeInfoLength(rank);
        //create the ndarray shape information
        DataBuffer shapeBuff = Nd4j.createBuffer(new int[shapeBufferLength]);

        //compute the databuffer type from the index
        DataBuffer.Type type = DataBuffer.Type.values()[byteBuffer.getInt()];
        for(int i = 0; i < shapeBufferLength; i++) {
            shapeBuff.put(i,byteBuffer.getInt());
        }




        //after the rank,data type, shape buffer (of length shape buffer length) * sizeof(int)
        if(type != DataBuffer.Type.COMPRESSED) {
            byteBuffer = byteBuffer.slice();
            //wrap the data buffer for the last bit
            DataBuffer buff = Nd4j.createBuffer(byteBuffer,type,Shape.length(shapeBuff));
            //create the final array
            INDArray arr = Nd4j.createArrayFromShapeBuffer(buff,shapeBuff);
            return arr;
        }
        else {
            CompressionDescriptor compressionDescriptor = CompressionDescriptor.fromByteBuffer(byteBuffer);
            byteBuffer = byteBuffer.slice();
            //ensure that we only deal with the slice of the buffer that is actually the data
            BytePointer byteBufferPointer = new BytePointer(byteBuffer);
            //create a compressed array based on the rest of the data left in the buffer
            CompressedDataBuffer compressedDataBuffer = new CompressedDataBuffer(byteBufferPointer,compressionDescriptor);
            INDArray arr = Nd4j.createArrayFromShapeBuffer(compressedDataBuffer,shapeBuff);
            arr.markAsCompressed(true);
            return arr;
        }

    }

    /**
     * Create an ndarray
     * from the unsafe buffer
     * @param buffer the buffer to create the array from
     * @return the ndarray derived from this buffer
     */
    public static INDArray toArray(DirectBuffer buffer) {
        return toArray(buffer,0);
    }



}
