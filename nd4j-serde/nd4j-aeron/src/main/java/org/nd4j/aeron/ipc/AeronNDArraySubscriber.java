package org.nd4j.aeron.ipc;

import io.aeron.Aeron;
import io.aeron.Subscription;
import lombok.Builder;
import lombok.Data;
import org.agrona.concurrent.SigInt;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import java.util.concurrent.atomic.AtomicBoolean;



@Data
@Builder
public class AeronNDArraySubscriber {
    // The channel (an endpoint identifier) to receive messages from
    private String channel;
    // A unique identifier for a stream within a channel. Stream ID 0 is reserved
    // for internal use and should not be used by applications.
    private int streamId;
    // Maximum number of message fragments to receive during a single 'poll' operation
    private int fragmentLimitCount;
    // Create a context, needed for client connection to media driver
    // A separate media driver process need to run prior to running this application
    private  Aeron.Context ctx;
    private  AtomicBoolean running = new AtomicBoolean(true);
    private final AtomicBoolean init = new AtomicBoolean(false);
    private static Logger log = LoggerFactory.getLogger(AeronNDArraySubscriber.class);
    private NDArrayCallback ndArrayCallback;


    private void init() {
        ctx = ctx == null ? new Aeron.Context() : ctx;
        channel = channel == null ?  "aeron:udp?endpoint=localhost:40123" : channel;
        fragmentLimitCount = fragmentLimitCount == 0 ? 1000 : fragmentLimitCount;
        streamId = streamId == 0 ? 10 : streamId;
        running = running == null ? new AtomicBoolean(true) : running;
        if(ndArrayCallback == null)
            throw new IllegalStateException("NDArray callback must be specified in the builder.");
        init.set(true);
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
        try (final Aeron aeron = Aeron.connect(ctx);
             final Subscription subscription = aeron.addSubscription(channel, streamId)) {
            log.info("Beginning subscribe on channel " + channel + " and stream " + streamId);
            AeronUtil.subscriberLoop(new NDArrayFragmentHandler(ndArrayCallback),fragmentLimitCount,running)
                    .accept(subscription);


        }
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
    public static AeronNDArraySubscriber startSubscriber(Aeron.Context context,
                                                         String host,
                                                         int port,
                                                         NDArrayCallback callback,
                                                         int streamId) {
        final AtomicBoolean running = new AtomicBoolean(true);


        AeronNDArraySubscriber subscriber = AeronNDArraySubscriber.builder().streamId(streamId)
                .ctx(context).channel(String.format("aeron:udp?endpoint=%s:%d",host,port))
                .running(running)
                .ndArrayCallback(callback).build();


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


    public static void main(String[] arg) {

    }

}




