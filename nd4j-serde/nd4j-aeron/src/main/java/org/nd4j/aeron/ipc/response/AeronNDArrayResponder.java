package org.nd4j.aeron.ipc.response;

import io.aeron.Aeron;
import io.aeron.Subscription;
import lombok.Builder;
import lombok.Data;
import org.agrona.concurrent.SigInt;
import org.nd4j.aeron.ipc.*;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.concurrent.atomic.AtomicBoolean;

/**
 * An aeron subscriber which, given
 * aeron connection information and an
 * @link {NDArrayHolder} will retrieve
 * the ndarray from the holder and send
 * an ndarray over aeron to the given
 * aeron channel.
 *
 * @author Adam Gibson
 */
@Builder
@Data
public class AeronNDArrayResponder {
    // The channel (an endpoint identifier) to receive messages from
    private String channel;
    // A unique identifier for a stream within a channel. Stream ID 0 is reserved
    // for internal use and should not be used by applications.
    private int streamId = -1;
    // A unique identifier for a stream within a channel. Stream ID 0 is reserved
    // for internal use and should not be used by applications.
    private int responseStreamId = -1;

    // Maximum number of message fragments to receive during a single 'poll' operation
    private int fragmentLimitCount;
    // Create a context, needed for client connection to media driver
    // A separate media driver process need to run prior to running this application
    private  Aeron.Context ctx;
    private AtomicBoolean running = new AtomicBoolean(true);
    private final AtomicBoolean init = new AtomicBoolean(false);
    private static Logger log = LoggerFactory.getLogger(AeronNDArraySubscriber.class);
    private NDArrayHolder ndArrayHolder;
    private Aeron aeron;
    private AtomicBoolean launched;


    private void init() {
        ctx = ctx == null ? new Aeron.Context() : ctx;
        channel = channel == null ?  "aeron:udp?endpoint=localhost:40123" : channel;
        fragmentLimitCount = fragmentLimitCount == 0 ? 1000 : fragmentLimitCount;
        streamId = streamId == 0 ? 10 : streamId;
        responseStreamId = responseStreamId == 0 ? -1 : responseStreamId;
        running = running == null ? new AtomicBoolean(true) : running;
        if(ndArrayHolder == null)
            throw new IllegalStateException("NDArray callback must be specified in the builder.");
        init.set(true);
        launched = new AtomicBoolean(false);
        log.info("Channel subscriber " + channel + " and stream id " + streamId);
    }


    /**
     * Launch a background thread
     * that subscribes to  the aeron context
     * @throws Exception
     */
    public void launch() throws Exception {
        if(init.get())
            return;
        // Register a SIGINT handler for graceful shutdown.
        if(!init.get())
            init();

        log.info("Subscribing to " + channel + " on stream Id " + streamId);

        // Register a SIGINT handler for graceful shutdown.
        SigInt.register(() -> running.set(false));

        // Create an Aeron instance with client-provided context configuration, connect to the
        // media driver, and add a subscription for the given channel and stream using the supplied
        // dataHandler method, which will be called with new messages as they are received.
        // The Aeron and Subscription classes implement AutoCloseable, and will automatically
        // clean up resources when this try block is finished.
        //Note here that we are either creating 1 or 2 subscriptions.
        //The first one is a  normal 1 subscription listener.
        //The second one is when we want to send responses
        boolean started = false;
        while(!started) {
            try {
                try (final Aeron aeron = Aeron.connect(ctx);
                     final Subscription subscription = aeron.addSubscription(channel, streamId)) {
                    this.aeron = aeron;
                    log.info("Beginning subscribe on channel " + channel + " and stream " + streamId);
                    AeronUtil.subscriberLoop(NDArrayResponseFragmentHandler.builder()
                            .aeron(aeron)
                            .context(ctx).streamId(responseStreamId)
                            .holder(ndArrayHolder)
                            .build(), fragmentLimitCount, running
                            ,launched)
                            .accept(subscription);
                    started = true;
                }


            } catch (Exception e) {
                log.warn("Failed to connect..trying again");
            }
        }

    }


    /**
     * Returns the connection uri in the form of:
     * host:port:streamId
     * @return
     */
    public String connectionUrl() {
        String[] split = channel.replace("aeron:udp?endpoint=","").split(":");
        String host = split[0];
        int port = Integer.parseInt(split[1]);
        return AeronConnectionInformation.of(host,port,streamId).toString();
    }

    /**
     * Start a subscriber in another thread
     * based on the given parameters
     * @param context the context to use
     * @param host the host name to bind to
     * @param port the port to bind to
     * @param callback the call back to use for the subscriber
     * @param streamId the stream id to subscribe to
     * @return the subscriber reference
     */
    public static AeronNDArrayResponder startSubscriber(Aeron.Context context,
                                                        String host,
                                                        int port,
                                                        NDArrayHolder callback,
                                                        int streamId) {

        final AtomicBoolean running = new AtomicBoolean(true);


        AeronNDArrayResponder subscriber = AeronNDArrayResponder.builder()
                .streamId(streamId)
                .ctx(context).channel(String.format("aeron:udp?endpoint=%s:%d",host,port))
                .running(running)
                .ndArrayHolder(callback).build();


        Thread t = new Thread(() -> {
            try {
                subscriber.launch();
            } catch (Exception e) {
                e.printStackTrace();
            }

        });

        t.start();

        return subscriber;
    }

}
