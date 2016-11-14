package org.nd4j.aeron.ipc;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;
import org.agrona.DirectBuffer;
import org.agrona.concurrent.UnsafeBuffer;
import org.apache.commons.lang3.tuple.Pair;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.io.Serializable;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.time.Instant;
import java.time.ZoneOffset;
import java.time.ZonedDateTime;

/**
 * A message sent over the wire for ndarrays
 * includ   ing the timestamp sent (in nanoseconds),
 * index (for tensor along dimension view based updates)
 * and dimensions.
 *
 * Fields:
 * arr: Using {@link AeronNDArraySerde#toArray(DirectBuffer, int)} we extract the array from a buffer
 * sent: the timestamp in milliseconds of when the message was sent (UTC timezone) - use {@link NDArrayMessage#getCurrentTimeUtc()}
 * when sending a message
 * index: the index of the tensor along dimension for update (use -1 if there is no index, eg: when you are going to use the whole array)
 * dimensions: the dimensions to do for a tensoralongdimension update, if you intend on updating the whole array send: new int[]{ -1} which
 * will indicate to use the whole array for an update.
 *
 *
 * @author Adam Gibson
 */
@Data
@Builder
@AllArgsConstructor
@NoArgsConstructor
public class NDArrayMessage implements Serializable {
    private INDArray arr;
    private long sent;
    private long index;
    private int[] dimensions;
    //default dimensions: a 1 length array of -1 means use the whole array for an update.
    private static int[] WHOLE_ARRAY_UPDATE = {-1};
    //represents the constant for indicating using the whole array for an update (-1)
    private static int WHOLE_ARRAY_INDEX = -1;

    public  enum MessageValidity {
        VALID,NULL_VALUE,INCONSISTENT_DIMENSIONS
    }


    /**
     * Prepare a whole array update
     * which includes the default dimensions
     * for indicating updating
     * the whole array (a 1 length int array with -1 as its only element)
     * -1 representing the dimension
     * @param arr
     * @return
     */
    public static NDArrayMessage wholeArrayUpdate(INDArray arr) {
        return NDArrayMessage.builder()
                .arr(arr).dimensions(WHOLE_ARRAY_UPDATE)
                .index(WHOLE_ARRAY_INDEX)
                .sent(getCurrentTimeUtc()).build();
    }

    /**
     * Factory method for creating an array
     * to send now (uses now in utc for the timestamp).
     * Note that this method will throw an
     * {@link IllegalArgumentException} if an invalid message is passed in.
     * An invalid message is as follows:
     * An index of -1 and dimensions that are of greater length than 1 with an element that isn't -1
     *
     * @param arr the array to send
     * @param dimensions the dimensions to use
     * @param index the index to use
     * @return the created
     */
    public static NDArrayMessage of(INDArray arr,int[] dimensions,long index) {
        //allow null dimensions as long as index is -1
        if(dimensions == null) {
            dimensions = WHOLE_ARRAY_UPDATE;
        }

        //validate index being built
        if(index > 0) {
            if(dimensions.length > 1 || dimensions.length == 1 && dimensions[0] != -1)
                throw new IllegalArgumentException("Inconsistent message. Your index is > 0 indicating you want to send a whole ndarray message but your dimensions indicate you are trying to send a partial update. Please ensure you use a 1 length int array with negative 1 as an element or use NDArrayMesage.wholeArrayUpdate(ndarray) for creation instead");
        }

        return NDArrayMessage.builder()
                .index(index)
                .dimensions(dimensions)
                .sent(getCurrentTimeUtc()).arr(arr).build();
    }


    /**
     * Returns if a message is valid or not based on a few simple conditions:
     * no null values
     * both index and the dimensions array must be -1 and of length 1 with an element of -1 in it
     * otherwise it is a valid message.
     * @param message the message to validate
     * @return 1 of: NULL_VALUE,INCONSISTENT_DIMENSIONS,VALID see {@link MessageValidity}
     */
    public static MessageValidity validMessage(NDArrayMessage message) {
        if(message.getDimensions() == null || message.getArr() == null)
            return MessageValidity.NULL_VALUE;

        if(message.getIndex() != -1 && message.getDimensions().length == 1 && message.getDimensions()[0] != -1)
            return MessageValidity.INCONSISTENT_DIMENSIONS;
        return MessageValidity.VALID;
    }


    /**
     * Get the current time in utc in milliseconds
     * @return the current time in utc in
     * milliseconds
     */
    public static long getCurrentTimeUtc() {
        Instant instant = Instant.now();
        ZonedDateTime dateTime = instant.atZone(ZoneOffset.UTC);
        return dateTime.toInstant().toEpochMilli();
    }

    /**
     * Returns the size needed in bytes
     * for a bytebuffer for a given ndarray message.
     * The formula is:
     * {@link AeronNDArraySerde#byteBufferSizeFor(INDArray)}
     * + size of dimension length (4)
     * + time stamp size (8)
     * + index size (8)
     * + 4 * message.getDimensions.length
     * @param message the message to get the length for
     * @return the size of the byte buffer for a message
     */
    public static int byteBufferSizeForMessage(NDArrayMessage message) {
        int nInts = 4 * message.getDimensions().length;
        int sizeofDimensionLength = 4;
        int timeStampSize = 8;
        int indexSize = 8;
        return nInts + sizeofDimensionLength + timeStampSize + indexSize + AeronNDArraySerde.byteBufferSizeFor(message.getArr());
    }

    /**
     * Convert a message to a direct buffer.
     * See {@link NDArrayMessage#fromBuffer(DirectBuffer, int)}
     * for a description of the format for the buffer
     * @param message the message to convert
     * @return a direct byte buffer representing this message.
     */
    public static DirectBuffer toBuffer(NDArrayMessage message) {
        ByteBuffer byteBuffer = ByteBuffer.allocateDirect(byteBufferSizeForMessage(message)).order(ByteOrder.nativeOrder());
        //perform the ndarray put on the
        if(message.getArr().isCompressed()) {
            AeronNDArraySerde.doByteBufferPutCompressed(message.getArr(),byteBuffer,false);
        }
        else {
            AeronNDArraySerde.doByteBufferPutUnCompressed(message.getArr(),byteBuffer,false);
        }

        long sent = message.getSent();
        long index = message.getIndex();
        byteBuffer.putLong(sent);
        byteBuffer.putLong(index);
        byteBuffer.putInt(message.getDimensions().length);
        for(int i = 0; i < message.getDimensions().length; i++) {
            byteBuffer.putInt(message.getDimensions()[i]);
        }

        //rewind the buffer before putting it in to the unsafe buffer
        //note that we set rewind to false in the do byte buffer put methods
        byteBuffer.rewind();

        return new UnsafeBuffer(byteBuffer);
    }

    /**
     * Convert a direct buffer to an ndarray
     * message.
     * The format of the byte buffer is:
     * ndarray
     * time
     * index
     * dimension length
     * dimensions
     *
     * We use {@link AeronNDArraySerde#toArrayAndByteBuffer(DirectBuffer, int)}
     * to read in the ndarray and just use normal {@link ByteBuffer#getInt()} and
     * {@link ByteBuffer#getLong()} to get the things like dimensions and index
     * and time stamp
     * @param buffer the buffer to convert
     * @param offset  the offset to start at with the buffer
     * @return the ndarray message based on this direct buffer.
     */
    public static NDArrayMessage fromBuffer(DirectBuffer buffer,int offset) {
        Pair<INDArray,ByteBuffer> pair = AeronNDArraySerde.toArrayAndByteBuffer(buffer, offset);
        INDArray arr = pair.getKey();
        if(arr.isCompressed())
            arr = Nd4j.getCompressor().decompress(arr);
        //use the rest of the buffer, of note here the offset is already set, we should only need to use
        ByteBuffer rest = pair.getRight();
        long time = rest.getLong();
        long index = rest.getLong();
        //get the array next for dimensions
        int dimensionLength = rest.getInt();
        if(dimensionLength <= 0)
            throw new IllegalArgumentException("Invalid dimension length " + dimensionLength);
        int[] dimensions = new int[dimensionLength];
        for(int i = 0; i < dimensionLength; i++)
            dimensions[i] = rest.getInt();
        return  NDArrayMessage.builder()
                .sent(time)
                .arr(arr)
                .index(index)
                .dimensions(dimensions)
                .build();
    }

}
