package org.nd4j.aeron.ipc.multi;

import io.aeron.Aeron;
import io.aeron.Subscription;
import lombok.Builder;
import lombok.Data;
import org.agrona.concurrent.SigInt;
import org.nd4j.aeron.ipc.AeronConnectionInformation;
import org.nd4j.aeron.ipc.AeronUtil;
import org.nd4j.aeron.ipc.NDArrayCallback;
import org.nd4j.aeron.ipc.NDArrayFragmentHandler;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.concurrent.Executor;
import java.util.concurrent.Executors;
import java.util.concurrent.atomic.AtomicBoolean;


/**
 *
 * Subscriber for ndarrays.
 * This is a pass through class for aeron
 * that will pass ndarrays received from channels
 * to an {@link NDArrayCallback} for operation after
 * assembling the ndaray
 * from a raw {@link org.agrona.concurrent.UnsafeBuffer}
 *
 * @author Adam Gibson
 */
@Data
@Builder
public class MultiAeronNDArraySubscriber {
    // The channel (an endpoint identifier) to receive messages from
    private String channel;
    // A unique identifier for a stream within a channel. Stream ID 0 is reserved
    // for internal use and should not be used by applications.
    private int[] streamIds;
    // A unique identifier for a stream within a channel. Stream ID 0 is reserved
    // for internal use and should not be used by applications.
    // Maximum number of message fragments to receive during a single 'poll' operation
    private int fragmentLimitCount;
    // Create a context, needed for client connection to media driver
    // A separate media driver process need to run prior to running this application
    private  Aeron.Context ctx;
    private  AtomicBoolean running = new AtomicBoolean(true);
    private final AtomicBoolean init = new AtomicBoolean(false);
    private static Logger log = LoggerFactory.getLogger(MultiAeronNDArraySubscriber.class);
    private NDArrayCallback ndArrayCallback;
    private Aeron[] aeron;
    private Subscription[] subscriptions;
    private AtomicBoolean launched = new AtomicBoolean(false);
    private Executor executors;




    private void init() {
        ctx = ctx == null ? new Aeron.Context() : ctx;
        channel = channel == null ?  "aeron:udp?endpoint=localhost:40123" : channel;
        fragmentLimitCount = fragmentLimitCount == 0 ? 1000 : fragmentLimitCount;
        streamIds = streamIds ==  null ?  new int[] {10} : streamIds;
        running = running == null ? new AtomicBoolean(true) : running;
        if(ndArrayCallback == null)
            throw new IllegalStateException("NDArray callback must be specified in the builder.");
        init.set(true);
        for(int i = 0; i < streamIds.length; i++)
            log.info("Channel subscriber " + channel + " and stream id " + streamIds[i]);
        launched = new AtomicBoolean(false);
        executors = Executors.newFixedThreadPool(streamIds.length);

    }


    /**
     * Returns true if the subscriber
     * is launched or not
     * @return true if the subscriber is launched, false otherwise
     */
    public synchronized  boolean launched() {
        if(launched == null)
            launched = new AtomicBoolean(false);
        return launched.get();
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

        for(int i = 0; i < streamIds.length; i++)
            log.info("Subscribing to " + channel + " on stream Id " + streamIds[i]);
        log.info("Using aeron directory " + ctx.aeronDirectoryName());

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
            for(int i  = 0; i < streamIds.length; i++) {
                final int streamId = streamIds[i];
                final int j = i;
                executors.execute(() -> {
                    try (final Aeron aeron1 = Aeron.connect(ctx);
                         final Subscription subscription = aeron1.addSubscription(channel, streamId)) {
                        MultiAeronNDArraySubscriber.this.aeron[j] = aeron1;
                        MultiAeronNDArraySubscriber.this.subscriptions[j] = subscription;
                        log.info("Beginning subscribe on channel " + channel + " and stream " + streamId);
                        AeronUtil.subscriberLoop(
                                new NDArrayFragmentHandler(ndArrayCallback),
                                fragmentLimitCount,
                                running,launched)
                                .accept(subscription);

                    }
                    catch(Exception e) {
                        log.warn("Unable to connect...trying again on channel " + channel,e);
                    }
                });

            }

        }

    }

    /**
     * Returns the connection uri in the form of:
     * host:port:streamId
     * @return
     */
    public String connectionUrl(int streamIdIdx) {
        String[] split = channel.replace("aeron:udp?endpoint=","").split(":");
        String host = split[0];
        int port = Integer.parseInt(split[1]);
        return AeronConnectionInformation.of(host,port,streamIds[streamIdIdx]).toString();
    }

    /**
     * Start a subscriber in another thread
     * based on the given parameters
     * @param context the context to use
     * @param host the host name to bind to
     * @param port the port to bind to
     * @param callback the call back to use for the subscriber
     * @param streamIds the stream id to subscribe to
     * @return the subscriber reference
     */
    public static MultiAeronNDArraySubscriber startSubscriber(Aeron.Context context,
                                                              String host,
                                                              int port,
                                                              NDArrayCallback callback,
                                                              int[] streamIds,
                                                              AtomicBoolean running) {


        MultiAeronNDArraySubscriber subscriber = MultiAeronNDArraySubscriber
                .builder()
                .streamIds(streamIds)
                .ctx(context)
                .channel(AeronUtil.aeronChannel(host,port))
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


}




