package org.nd4j.parameterserver.distributed.transport;

import io.aeron.Aeron;
import io.aeron.FragmentAssembler;
import io.aeron.Publication;
import io.aeron.Subscription;
import io.aeron.driver.MediaDriver;
import io.aeron.logbuffer.Header;
import lombok.Getter;
import lombok.NonNull;
import lombok.extern.slf4j.Slf4j;
import org.agrona.CloseHelper;
import org.agrona.DirectBuffer;
import org.agrona.concurrent.IdleStrategy;
import org.agrona.concurrent.SleepingIdleStrategy;
import org.nd4j.linalg.exception.ND4JIllegalStateException;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.parameterserver.distributed.conf.VoidConfiguration;
import org.nd4j.parameterserver.distributed.enums.NodeRole;
import org.nd4j.parameterserver.distributed.logic.completion.Clipboard;
import org.nd4j.parameterserver.distributed.messages.Frame;
import org.nd4j.parameterserver.distributed.messages.MeaningfulMessage;
import org.nd4j.parameterserver.distributed.messages.VoidMessage;

import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.LinkedBlockingQueue;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicLong;

/**
 * @author raver119@gmail.com
 */
@Slf4j
public abstract class BaseTransport implements Transport {
    protected VoidConfiguration voidConfiguration;
    protected NodeRole nodeRole;

    protected Aeron aeron;
    protected Aeron.Context context;

    protected String unicastChannelUri;

    protected String ip;
    protected int port = 0;

    // TODO: move this to singleton holder
    protected MediaDriver driver;

    protected Publication publicationForShards;
    protected Publication publicationForClients;

    protected Subscription subscriptionForShards;
    protected Subscription subscriptionForClients;

    protected FragmentAssembler messageHandlerForShards;
    protected FragmentAssembler messageHandlerForClients;

    protected LinkedBlockingQueue<VoidMessage> messages = new LinkedBlockingQueue<>();

    protected Map<Long, MeaningfulMessage> completed = new ConcurrentHashMap<>();

    protected AtomicBoolean runner = new AtomicBoolean(true);

    // service threads where poll will happen
    protected Thread threadA;
    protected Thread threadB;

    protected Clipboard clipboard;

    protected AtomicLong frameCount = new AtomicLong(0);

    // TODO: make this configurable?
    protected IdleStrategy idler = new SleepingIdleStrategy(1000);
    protected IdleStrategy feedbackIdler = new SleepingIdleStrategy(100000);

    protected ThreadingModel threadingModel = ThreadingModel.DEDICATED_THREADS;

    protected long originatorId;

    // TODO: make this auto-configurable
    @Getter
    protected short targetIndex = 0;
    @Getter
    protected short shardIndex = 0;

    @Override
    public long getOwnOriginatorId() {
        return originatorId;
    }

    @Override
    public MeaningfulMessage sendMessageAndGetResponse(@NonNull VoidMessage message) {
        long startTime = System.currentTimeMillis();

        long taskId = message.getTaskId();
        sendCommandToShard(message);
        AtomicLong cnt = new AtomicLong(0);

        //        log.info("Sent message to shard: {}, taskId: {}, originalId: {}", message.getClass().getSimpleName(), message.getTaskId(), taskId);

        long currentTime = System.currentTimeMillis();

        MeaningfulMessage msg;
        while ((msg = completed.get(taskId)) == null) {
            try {
                //Thread.sleep(voidConfiguration.getResponseTimeframe());
                feedbackIdler.idle();

                if (System.currentTimeMillis() - currentTime > voidConfiguration.getResponseTimeout()) {
                    log.info("Resending request for taskId [{}]", taskId);
                    message.incrementRetransmitCount();

                    // TODO: make retransmit threshold configurable
                    if (message.getRetransmitCount() > 20)
                        throw new RuntimeException("Giving up on message delivery...");

                    return sendMessageAndGetResponse(message);
                }
            } catch (Exception e) {
                throw new RuntimeException(e);
            }
        }

        completed.remove(taskId);

        long endTime = System.currentTimeMillis();
        long timeSpent = endTime - startTime;

        if (message instanceof Frame && frameCount.incrementAndGet() % 1000 == 0)
            log.info("Frame of {} messages [{}] processed in {} ms", ((Frame) message).size(), message.getTaskId(),
                            timeSpent);


        return msg;
    }

    @Override
    public void setIpAndPort(@NonNull String ip, int port) {
        this.ip = ip;
        this.port = port;
    }

    @Override
    public void sendMessage(@NonNull VoidMessage message) {
        switch (message.getMessageType()) {
            // messages 0..9 inclusive are reserved for Client->Shard commands
            case 0:
            case 1:
            case 2:
            case 3:
            case 4:
            case 5:
            case 6:
            case 7:
            case 8:
            case 9:
                // TODO: check, if current role is Shard itself, in this case we want to modify command queue directly, to reduce network load
                // this command is possible to issue from any node role
                //log.info("Sending message to Shard: {}; Messages size: {}", message.getClass().getSimpleName(), messages.size());
                //while (messages.size() > 100) {
                //    log.info("sI_{} [{}] going to sleep..", shardIndex, nodeRole );
                //    idler.idle();
                //}

                if (message.isBlockingMessage()) {
                    // we issue blocking message, but we don't care about response
                    sendMessageAndGetResponse(message);
                } else
                    sendCommandToShard(message);
                break;
            // messages 10..19 inclusive are reserved for Shard->Clients commands
            case 10:
            case 11:
            case 12:
            case 13:
            case 19:
                // this command is possible to issue only from Shard
                //log.info("Sending feedback message to Client: {}", message.getClass().getSimpleName());
                sendFeedbackToClient(message);
                break;
            // messages 20..29 inclusive are reserved for Shard->Shard commands
            case 20:
            case 21:
            case 22:
            case 28:
                //log.info("Sending message to ALL Shards: {}", message.getClass().getSimpleName());
                sendCoordinationCommand(message);
                break;
            default:
                throw new RuntimeException("Unknown messageType passed for delivery");
        }
    }

    /**
     * This message handler is responsible for receiving messages on Shard side
     *
     * @param buffer
     * @param offset
     * @param length
     * @param header
     */
    protected void shardMessageHandler(DirectBuffer buffer, int offset, int length, Header header) {
        /**
         * All incoming messages here are supposed to be unicast messages.
         */
        // TODO: implement fragmentation handler here PROBABLY. Or forbid messages > MTU?
        //log.info("shardMessageHandler message request incoming...");
        byte[] data = new byte[length];
        buffer.getBytes(offset, data);

        VoidMessage message = VoidMessage.fromBytes(data);
        if (message.getMessageType() == 7) {
            // if that's vector request message - it's special case, we don't send it to other shards yet
            //log.info("Shortcut for vector request");
            messages.add(message);
        } else {
            // and send it away to other Shards
            publicationForShards.offer(buffer, offset, length);
        }
    }

    /**
     * This message handler is responsible for receiving coordination messages on Shard side
     *
     * @param buffer
     * @param offset
     * @param length
     * @param header
     */
    protected void internalMessageHandler(DirectBuffer buffer, int offset, int length, Header header) {
        /**
         * All incoming internal messages are either op commands, or aggregation messages that are tied to commands
         */
        byte[] data = new byte[length];
        buffer.getBytes(offset, data);

        VoidMessage message = VoidMessage.fromBytes(data);

        messages.add(message);

        //    log.info("internalMessageHandler message request incoming: {}", message.getClass().getSimpleName());
    }

    /**
     * This message handler is responsible for receiving messages on Client side
     * @param buffer
     * @param offset
     * @param length
     * @param header
     */
    protected void clientMessageHandler(DirectBuffer buffer, int offset, int length, Header header) {
        /**
         *  All incoming messages here are supposed to be "just messages", only unicast communication
         *  All of them should implement MeaningfulMessage interface
         */
        // TODO: to be implemented
        //  log.info("clientMessageHandler message request incoming");

        byte[] data = new byte[length];
        buffer.getBytes(offset, data);

        MeaningfulMessage message = (MeaningfulMessage) VoidMessage.fromBytes(data);
        completed.put(message.getTaskId(), message);
    }


    /**
     * @param message
     */
    @Override
    public void sendMessageToAllShards(VoidMessage message) {
        //        if (nodeRole != NodeRole.SHARD)
        //            throw new RuntimeException("This method shouldn't be called only from Shard context");

        //log.info("Sending message to All shards");

        message.setTargetId((short) -1);
        //publicationForShards.offer(message.asUnsafeBuffer());
        sendCoordinationCommand(message);
    }

    /**
     * This method does initialization of Transport instance
     *
     * @param voidConfiguration
     * @param clipboard
     * @param role
     * @param localIp
     * @param localPort
     * @param shardIndex
     */
    @Override
    public void init(VoidConfiguration voidConfiguration, Clipboard clipboard, NodeRole role, String localIp,
                    int localPort, short shardIndex) {
        //Runtime.getRuntime().addShutdownHook(new Thread(() -> shutdownSilent()));
    }

    /**
     * This method starts transport mechanisms.
     *
     * PLEASE NOTE: init() method should be called prior to launch() call
     */
    @Override
    public void launch(@NonNull ThreadingModel threading) {
        this.threadingModel = threading;

        switch (threading) {
            case SINGLE_THREAD: {

                log.warn("SINGLE_THREAD model is used, performance will be significantly reduced");

                // single thread for all queues. shouldn't be used in real world
                threadA = new Thread(() -> {
                    while (runner.get()) {
                        if (subscriptionForShards != null)
                            subscriptionForShards.poll(messageHandlerForShards, 512);

                        idler.idle(subscriptionForClients.poll(messageHandlerForClients, 512));
                    }
                });

                threadA.start();
            }
                break;
            case DEDICATED_THREADS: {
                // we start separate thread for each handler

                /**
                 * We definitely might use less conditional code here, BUT i'll keep it as is,
                 * only because we want code to be obvious for people
                 */
                final AtomicBoolean localRunner = new AtomicBoolean(false);
                if (nodeRole == NodeRole.NONE) {
                    throw new ND4JIllegalStateException("No role is set for current node!");
                } else if (nodeRole == NodeRole.SHARD || nodeRole == NodeRole.BACKUP || nodeRole == NodeRole.MASTER) {
                    // // Shard or Backup uses two subscriptions

                    // setting up thread for shard->client communication listener
                    if (messageHandlerForShards != null)
                        threadB = new Thread(() -> {
                            while (runner.get())
                                idler.idle(subscriptionForShards.poll(messageHandlerForShards, 512));

                        });

                    // setting up thread for inter-shard communication listener
                    threadA = new Thread(() -> {
                        localRunner.set(true);
                        while (runner.get())
                            idler.idle(subscriptionForClients.poll(messageHandlerForClients, 512));
                    });

                    if (threadB != null) {
                        Nd4j.getAffinityManager().attachThreadToDevice(threadB,
                                        Nd4j.getAffinityManager().getDeviceForCurrentThread());
                        threadB.setDaemon(true);
                        threadB.setName("VoidParamServer subscription threadB [" + nodeRole + "]");
                        threadB.start();
                    }
                } else {
                    // setting up thread for shard->client communication listener
                    threadA = new Thread(() -> {
                        localRunner.set(true);
                        while (runner.get())
                            idler.idle(subscriptionForClients.poll(messageHandlerForClients, 512));
                    });
                }

                // all roles have threadA anyway
                Nd4j.getAffinityManager().attachThreadToDevice(threadA,
                                Nd4j.getAffinityManager().getDeviceForCurrentThread());
                threadA.setDaemon(true);
                threadA.setName("VoidParamServer subscription threadA [" + nodeRole + "]");
                threadA.start();

                while (!localRunner.get())
                    try {
                        Thread.sleep(50);
                    } catch (Exception e) {
                    }
            }
                break;
            case SAME_THREAD: {
                // no additional threads at all, we do poll within takeMessage loop
                log.warn("SAME_THREAD model is used, performance will be dramatically reduced");
            }
                break;
            default:
                throw new IllegalStateException("Unknown thread model: [" + threading.toString() + "]");
        }
    }


    protected void shutdownSilent() {
        log.info("Shutting down Aeron infrastructure...");
        CloseHelper.quietClose(publicationForClients);
        CloseHelper.quietClose(publicationForShards);
        CloseHelper.quietClose(subscriptionForShards);
        CloseHelper.quietClose(subscriptionForClients);
        CloseHelper.quietClose(aeron);
        CloseHelper.quietClose(context);
        CloseHelper.quietClose(driver);
    }

    /**
     * This method stops transport system.
     */
    @Override
    public void shutdown() {
        // Since Aeron's poll isn't blocking, all we need is just special flag
        runner.set(false);
        try {
            threadA.join();

            if (threadB != null)
                threadB.join();
        } catch (Exception e) {
            //
        }
        CloseHelper.quietClose(driver);
        try {
            Thread.sleep(500);
        } catch (Exception e) {

        }
    }

    /**
     * This method saves incoming message to the Queue, for later dispatch from higher-level code, like actual TrainingFunction or VoidParameterServer itself
     *
     * @param message
     */
    @Override
    public void receiveMessage(VoidMessage message) {
        try {
            log.info("Message received, saving...");
            messages.put(message);
        } catch (Exception e) {
            // do nothing
        }
    }

    /**
     * This method takes 1 message from "incoming messages" queue, blocking if queue is empty
     *
     * @return
     */
    @Override
    public VoidMessage takeMessage() {
        if (threadingModel != ThreadingModel.SAME_THREAD) {
            try {
                return messages.take();
            } catch (InterruptedException e) {
                // probably we don't want to do anything here
                return null;
            } catch (Exception e) {
                throw new RuntimeException(e);
            }
        } else {
            /**
             * PLEASE NOTE: This branch is suitable for debugging only, should never be used in wild life
             */
            // we do inplace poll
            if (subscriptionForShards != null)
                subscriptionForShards.poll(messageHandlerForShards, 512);

            subscriptionForClients.poll(messageHandlerForClients, 512);

            return messages.poll();
        }
    }

    /**
     * This method puts message into processing queue
     *
     * @param message
     */
    @Override
    public void putMessage(@NonNull VoidMessage message) {
        messages.add(message);
    }

    /**
     * This method peeks 1 message from "incoming messages" queue, returning null if queue is empty
     *
     * PLEASE NOTE: This method is suitable for debug purposes only
     *
     * @return
     */
    @Override
    public VoidMessage peekMessage() {
        return messages.peek();
    }

    /**
     * This command is possible to issue only from Client
     *
     * @param message
     */
    protected synchronized void sendCommandToShard(VoidMessage message) {
        // if this node is shard - we just step over TCP/IP infrastructure
        // TODO: we want LocalTransport to be used in such cases
        if (nodeRole == NodeRole.SHARD) {
            message.setTargetId(shardIndex);
            messages.add(message);
            return;
        }


        //log.info("Sending CS: {}", message.getClass().getCanonicalName());

        message.setTargetId(targetIndex);
        DirectBuffer buffer = message.asUnsafeBuffer();

        long result = publicationForShards.offer(buffer);

        if (result < 0)
            for (int i = 0; i < 5 && result < 0; i++) {
                try {
                    // TODO: make this configurable
                    Thread.sleep(1000);
                } catch (Exception e) {
                }
                result = publicationForShards.offer(buffer);
            }

        // TODO: handle retransmit & backpressure separately

        if (result < 0)
            throw new RuntimeException("Unable to send message over the wire. Error code: " + result);
    }

    /**
     * This command is possible to issue only from Shard
     *
     * @param message
     */
    protected abstract void sendCoordinationCommand(VoidMessage message);

    /**
     * This command is possible to issue only from Shard
     * @param message
     */
    protected abstract void sendFeedbackToClient(VoidMessage message);

    public void addClient(String ip, int port) {
        //
    }

    @Override
    public String getIp() {
        return ip;
    }

    @Override
    public int getPort() {
        return port;
    }

    @Override
    public int numberOfKnownClients() {
        return 0;
    }

    @Override
    public int numberOfKnownShards() {
        return 0;
    }

    @Override
    public void addShard(String ip, int port) {
        // no-op
    }

    @Override
    public void sendMessageToAllClients(VoidMessage message, Long... exclusions) {
        // no-op
    }
}
