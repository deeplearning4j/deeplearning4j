package org.nd4j.aeron.ipc;

import io.aeron.Aeron;
import io.aeron.Publication;
import io.aeron.exceptions.DriverTimeoutException;
import lombok.Builder;
import lombok.Data;

import org.agrona.concurrent.UnsafeBuffer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.Closeable;

/**
 * NDArray publisher for aeron
 *
 * @author Adam Gibson
 */
@Data
@Builder
public class AeronNDArrayPublisher implements  AutoCloseable {
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



    private void init() {
        channel = channel == null ? "aeron:udp?endpoint=localhost:40123" : channel;
        streamId = streamId == 0 ? 10 : streamId;
        ctx = ctx == null ? ctx = new Aeron.Context() : ctx;
        init = true;
        log.debug("Channel publisher" + channel + " and stream " + streamId);
    }

    /**
     * Publish an ndarray to an aeron channel
     * @param arr
     * @throws Exception
     */
    public void publish(INDArray arr) throws Exception {
        // Allocate enough buffer size to hold maximum message length
        // The UnsafeBuffer class is part of the Agrona library and is used for efficient buffer management
        log.debug("Publishing to " + channel + " on stream Id " + streamId);
        //ensure default values are set
        while(!arr.isCompressed())
            Nd4j.getCompressor().compressi(arr,"GZIP");


        UnsafeBuffer buffer = AeronNDArraySerde.toBuffer(arr);

        if(!init)
            init();
        // Create a context, needed for client connection to media driver
        // A separate media driver process needs to be running prior to starting this application

        // Create an Aeron instance with client-provided context configuration and connect to the
        // media driver, and create a Publication.  The Aeron and Publication classes implement
        // AutoCloseable, and will automatically clean up resources when this try block is finished.
        boolean connected = false;
        if(aeron == null) {
            try {
                while(!connected) {
                    aeron = Aeron.connect(ctx);
                    connected = true;
                }
            }catch (Exception e) {
                log.warn("Reconnecting on publisher...failed to connect");
            }
        }
        int connectionTries = 0;
        while(publication == null && connectionTries < 3) {
            try {
                publication = aeron.addPublication(channel, streamId);
            }catch (DriverTimeoutException e) {
                log.warn("Failed to connect due to driver time out on channel " + channel + " and stream" + streamId);
                connectionTries++;
            }
        }


        // Try to publish the buffer. 'offer' is a non-blocking call.
        // If it returns less than 0, the message was not sent, and the offer should be retried.
        long result;
        log.debug("Begin publish " + channel + " and stream " + streamId);
        while ((result = publication.offer(buffer, 0, buffer.capacity())) < 0L) {
            if (result == Publication.BACK_PRESSURED)
            {
                log.debug(" Offer failed due to back pressure");
            }
            else if (result == Publication.NOT_CONNECTED)
            {
                log.debug(" Offer failed because publisher is not connected to subscriber");
            }
            else if (result == Publication.ADMIN_ACTION)
            {
                log.debug("Offer failed because of an administration action in the system");
            }
            else if (result == Publication.CLOSED)
            {
                log.debug("Offer failed publication is closed");
            }
            else
            {
                log.debug(" Offer failed due to unknown reason");
            }
        }


        log.debug("Done sending.");

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
        if(aeron != null) {
            try {
                aeron.close();
            }catch (Exception e) {}
        }
        if(publication != null) {
            try {
                publication.close();
            }catch(Exception e) {}
        }
    }
}
