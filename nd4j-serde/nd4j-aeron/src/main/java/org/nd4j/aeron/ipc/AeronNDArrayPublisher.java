package org.nd4j.aeron.ipc;

import io.aeron.Aeron;
import io.aeron.Publication;
import io.aeron.exceptions.DriverTimeoutException;
import lombok.Builder;
import lombok.Data;
import org.agrona.CloseHelper;
import org.agrona.DirectBuffer;
import org.agrona.concurrent.BusySpinIdleStrategy;
import org.agrona.concurrent.UnsafeBuffer;
import org.nd4j.aeron.ipc.chunk.NDArrayMessageChunk;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.Closeable;
import java.nio.ByteBuffer;

/**
 * NDArray publisher
 * for aeron
 *
 * @author Adam Gibson
 */
@Data
@Builder
public class AeronNDArrayPublisher implements AutoCloseable {
    // A unique identifier for a stream within a channel. Stream ID 0 is reserved
    // for internal use and should not be used by applications.
    private int streamId;
    // The channel (an endpoint identifier) to send the message to
    private String channel;
    private boolean init = false;
    private Aeron.Context ctx;
    private Aeron aeron;
    private Publication publication;
    private static Logger log = LoggerFactory.getLogger(AeronNDArrayPublisher.class);
    public final static int NUM_RETRIES = 100;
    private boolean compress = true;
    private static final BusySpinIdleStrategy busySpinIdleStrategy = new BusySpinIdleStrategy();
    private int publishRetryTimeOut = 3000;

    private void init() {
        channel = channel == null ? "aeron:udp?endpoint=localhost:40123" : channel;
        streamId = streamId == 0 ? 10 : streamId;
        publishRetryTimeOut = publishRetryTimeOut == 0 ? 3000 : publishRetryTimeOut;
        ctx = ctx == null ? ctx = new Aeron.Context() : ctx;
        init = true;
        log.info("Channel publisher" + channel + " and stream " + streamId);
    }

    /**
     * Publish an ndarray
     * to an aeron channel
     * @param message
     * @throws Exception
     */
    public void publish(NDArrayMessage message) throws Exception {
        if (!init)
            init();
        // Create a context, needed for client connection to media driver
        // A separate media driver process needs to be running prior to starting this application

        // Create an Aeron instance with client-provided context configuration and connect to the
        // media driver, and create a Publication.  The Aeron and Publication classes implement
        // AutoCloseable, and will automatically clean up resources when this try block is finished.
        boolean connected = false;
        if (aeron == null) {
            try {
                while (!connected) {
                    aeron = Aeron.connect(ctx);
                    connected = true;
                }
            } catch (Exception e) {
                log.warn("Reconnecting on publisher...failed to connect");
            }
        }

        int connectionTries = 0;
        while (publication == null && connectionTries < NUM_RETRIES) {
            try {
                publication = aeron.addPublication(channel, streamId);
                log.info("Created publication on channel " + channel + " and stream " + streamId);
            } catch (DriverTimeoutException e) {
                Thread.sleep(1000 * (connectionTries + 1));
                log.warn("Failed to connect due to driver time out on channel " + channel + " and stream " + streamId
                                + "...retrying in " + connectionTries + " seconds");
                connectionTries++;
            }
        }

        if (!connected && connectionTries >= 3 || publication == null) {
            throw new IllegalStateException(
                            "Publisher unable to connect to channel " + channel + " and stream " + streamId);
        }


        // Allocate enough buffer size to hold maximum message length
        // The UnsafeBuffer class is part of the Agrona library and is used for efficient buffer management
        log.info("Publishing to " + channel + " on stream Id " + streamId);
        //ensure default values are set
        INDArray arr = message.getArr();
        if (isCompress())
            while (!message.getArr().isCompressed())
                Nd4j.getCompressor().compressi(arr, "GZIP");



        //array is large, need to segment
        if (NDArrayMessage.byteBufferSizeForMessage(message) >= publication.maxMessageLength()) {
            NDArrayMessageChunk[] chunks = NDArrayMessage.chunks(message, publication.maxMessageLength() / 128);
            for (int i = 0; i < chunks.length; i++) {
                ByteBuffer sendBuff = NDArrayMessageChunk.toBuffer(chunks[i]);
                sendBuff.rewind();
                DirectBuffer buffer = new UnsafeBuffer(sendBuff);
                sendBuffer(buffer);
            }
        } else {
            //send whole array
            DirectBuffer buffer = NDArrayMessage.toBuffer(message);
            sendBuffer(buffer);

        }

    }



    private void sendBuffer(DirectBuffer buffer) throws Exception {
        // Try to publish the buffer. 'offer' is a non-blocking call.
        // If it returns less than 0, the message was not sent, and the offer should be retried.
        long result;
        int tries = 0;
        while ((result = publication.offer(buffer, 0, buffer.capacity())) < 0L && tries < 5) {
            if (result == Publication.BACK_PRESSURED) {
                log.info("Offer failed due to back pressure");
            } else if (result == Publication.NOT_CONNECTED) {
                log.info("Offer failed because publisher is not connected to subscriber " + channel + " and stream "
                                + streamId);
            } else if (result == Publication.ADMIN_ACTION) {
                log.info("Offer failed because of an administration action in the system and channel" + channel
                                + " and stream " + streamId);
            } else if (result == Publication.CLOSED) {
                log.info("Offer failed publication is closed and channel" + channel + " and stream " + streamId);
            } else {
                log.info(" Offer failed due to unknown reason and channel" + channel + " and stream " + streamId);
            }



            Thread.sleep(publishRetryTimeOut);
            tries++;

        }

        if (tries >= 5 && result == 0)
            throw new IllegalStateException("Failed to send message");

    }

    /**
     * Publish an ndarray to an aeron channel
     * @param arr
     * @throws Exception
     */
    public void publish(INDArray arr) throws Exception {
        publish(NDArrayMessage.wholeArrayUpdate(arr));
    }


    /**
     * Closes this resource, relinquishing any underlying resources.
     * This method is invoked automatically on objects managed by the
     * {@code try}-with-resources statement.
     * <p>
     * <p>While this interface method is declared to throw {@code
     * Exception}, implementers are <em>strongly</em> encouraged to
     * declare concrete implementations of the {@code close} method to
     * throw more specific exceptions, or to throw no exception at all
     * if the close operation cannot fail.
     * <p>
     * <p> Cases where the close operation may fail require careful
     * attention by implementers. It is strongly advised to relinquish
     * the underlying resources and to internally <em>mark</em> the
     * resource as closed, prior to throwing the exception. The {@code
     * close} method is unlikely to be invoked more than once and so
     * this ensures that the resources are released in a timely manner.
     * Furthermore it reduces problems that could arise when the resource
     * wraps, or is wrapped, by another resource.
     * <p>
     * <p><em>Implementers of this interface are also strongly advised
     * to not have the {@code close} method throw {@link
     * InterruptedException}.</em>
     * <p>
     * This exception interacts with a thread's interrupted status,
     * and runtime misbehavior is likely to occur if an {@code
     * InterruptedException} is {@linkplain Throwable#addSuppressed
     * suppressed}.
     * <p>
     * More generally, if it would cause problems for an
     * exception to be suppressed, the {@code AutoCloseable.close}
     * method should not throw it.
     * <p>
     * <p>Note that unlike the {@link Closeable#close close}
     * method of {@link Closeable}, this {@code close} method
     * is <em>not</em> required to be idempotent.  In other words,
     * calling this {@code close} method more than once may have some
     * visible side effect, unlike {@code Closeable.close} which is
     * required to have no effect if called more than once.
     * <p>
     * However, implementers of this interface are strongly encouraged
     * to make their {@code close} methods idempotent.
     *
     * @throws Exception if this resource cannot be closed
     */
    @Override
    public void close() throws Exception {
        if (publication != null) {
            CloseHelper.quietClose(publication);
        }

    }
}
