package org.nd4j.parameterserver.distributed.transport;

import io.aeron.Aeron;
import io.aeron.FragmentAssembler;
import io.aeron.Publication;
import io.aeron.driver.MediaDriver;
import io.aeron.logbuffer.Header;
import lombok.*;
import lombok.extern.slf4j.Slf4j;
import org.agrona.CloseHelper;
import org.agrona.DirectBuffer;
import org.nd4j.linalg.exception.ND4JIllegalStateException;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.io.StringUtils;
import org.nd4j.linalg.util.HashUtil;
import org.nd4j.parameterserver.distributed.conf.VoidConfiguration;
import org.nd4j.parameterserver.distributed.enums.NodeRole;
import org.nd4j.parameterserver.distributed.logic.ClientRouter;
import org.nd4j.parameterserver.distributed.logic.RetransmissionHandler;
import org.nd4j.parameterserver.distributed.logic.completion.Clipboard;
import org.nd4j.parameterserver.distributed.messages.*;
import org.nd4j.parameterserver.distributed.messages.requests.IntroductionRequestMessage;
import org.nd4j.parameterserver.distributed.logic.routing.InterleavedRouter;

import java.util.*;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.LinkedBlockingQueue;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.locks.LockSupport;

import static java.lang.System.setProperty;

/**
 * Transport implementation based on UDP unicast, for restricted environments, where multicast isn't available. I.e. AWS or Azure
 *
 * @author raver119@gmail.com
 */
@Slf4j
public class RoutedTransport extends BaseTransport {

    protected List<RemoteConnection> shards = new ArrayList<>();
    protected Map<Long, RemoteConnection> clients = new ConcurrentHashMap<>();
    @Getter
    @Setter
    protected ClientRouter router;

    public RoutedTransport() {
        //
    }

    @Override
    public void init(@NonNull VoidConfiguration voidConfiguration, @NonNull Clipboard clipboard, @NonNull NodeRole role,
                    @NonNull String localIp, int localPort, short shardIndex) {
        this.nodeRole = role;
        this.clipboard = clipboard;
        this.voidConfiguration = voidConfiguration;
        this.shardIndex = shardIndex;
        this.messages = new LinkedBlockingQueue<>();
        //shutdown hook
        super.init(voidConfiguration, clipboard, role, localIp, localPort, shardIndex);
        setProperty("aeron.client.liveness.timeout", "30000000000");

        context = new Aeron.Context().publicationConnectionTimeout(30000000000L).driverTimeoutMs(30000)
                        .keepAliveInterval(100000000);

        driver = MediaDriver.launchEmbedded();
        context.aeronDirectoryName(driver.aeronDirectoryName());
        aeron = Aeron.connect(context);



        if (router == null)
            router = new InterleavedRouter();


        /*
            Regardless of current role, we raise subscription for incoming messages channel
         */
        // we skip IPs assign process if they were defined externally
        if (port == 0) {
            ip = localIp;
            port = localPort;
        }
        unicastChannelUri = "aeron:udp?endpoint=" + ip + ":" + port;
        subscriptionForClients = aeron.addSubscription(unicastChannelUri, voidConfiguration.getStreamId());
        //clean shut down
        Runtime.getRuntime().addShutdownHook(new Thread(() -> {
            CloseHelper.quietClose(aeron);
            CloseHelper.quietClose(driver);
            CloseHelper.quietClose(context);
            CloseHelper.quietClose(subscriptionForClients);
        }));


        messageHandlerForClients = new FragmentAssembler(
                        (buffer, offset, length, header) -> jointMessageHandler(buffer, offset, length, header));

        /*
            Now, regardless of current role,
             we set up publication channel to each shard
         */
        String shardChannelUri = null;
        String remoteIp = null;
        int remotePort = 0;
        for (String ip : voidConfiguration.getShardAddresses()) {
            if (ip.contains(":")) {
                shardChannelUri = "aeron:udp?endpoint=" + ip;
                String[] split = ip.split(":");
                remoteIp = split[0];
                remotePort = Integer.valueOf(split[1]);
            } else {
                shardChannelUri = "aeron:udp?endpoint=" + ip + ":" + voidConfiguration.getUnicastPort();
                remoteIp = ip;
                remotePort = voidConfiguration.getUnicastPort();
            }

            Publication publication = aeron.addPublication(shardChannelUri, voidConfiguration.getStreamId());

            RemoteConnection connection = RemoteConnection.builder().ip(remoteIp).port(remotePort)
                            .publication(publication).locker(new Object()).build();

            shards.add(connection);
        }

        if (nodeRole == NodeRole.SHARD)
            log.info("Initialized as [{}]; ShardIndex: [{}]; Own endpoint: [{}]", nodeRole, shardIndex,
                            unicastChannelUri);
        else
            log.info("Initialized as [{}]; Own endpoint: [{}]", nodeRole, unicastChannelUri);

        switch (nodeRole) {
            case MASTER:
            case BACKUP: {

            }
            case SHARD: {
                /*
                    For unicast transport we want to have interconnects between all shards first of all, because we know their IPs in advance.
                    But due to design requirements, clients have the same first step, so it's kinda shared for all states :)
                 */

                /*
                    Next step is connections setup for backup nodes.
                    TODO: to be implemented
                 */
                addClient(ip, port);
            }
                break;
            case CLIENT: {
                /*
                    For Clients on unicast transport, we either set up connection to single Shard, or to multiple shards
                    But since this code is shared - we don't do anything here
                 */
            }
                break;
            default:
                throw new ND4JIllegalStateException("Unknown NodeRole being passed: " + nodeRole);
        }

        router.init(voidConfiguration, this);
        this.originatorId = HashUtil.getLongHash(this.getIp() + ":" + this.getPort());
    }


    @Override
    public void sendMessageToAllClients(VoidMessage message, Long... exclusions) {
        if (nodeRole != NodeRole.SHARD)
            throw new ND4JIllegalStateException("Only SHARD allowed to send messages to all Clients");

        final DirectBuffer buffer = message.asUnsafeBuffer();

        // no need to search for matches above number of then exclusions
        final AtomicInteger cnt = new AtomicInteger(0);

        //final StringBuilder builder = new StringBuilder("Got message from: [").append(message.getOriginatorId()).append("]; Resend: {");

        clients.values().parallelStream().filter(rc -> {
            // do not send message back to yourself :)
            if (rc.getLongHash() == this.originatorId || rc.getLongHash() == 0) {
                //                builder.append(", SKIP: ").append(rc.getLongHash());
                return false;
            }

            // we skip exclusions here
            if (exclusions != null && cnt.get() < exclusions.length) {
                for (Long exclude : exclusions)
                    if (exclude.longValue() == rc.getLongHash()) {
                        cnt.incrementAndGet();
                        //                        builder.append(", SKIP: ").append(rc.getLongHash());
                        return false;
                    }
            }

            //       builder.append(", PASS: ").append(rc.getLongHash());
            return true;
        }).forEach((rc) -> {
            //      log.info("Sending message to {}", rc.getLongHash());

            RetransmissionHandler.TransmissionStatus res;
            long retr = 0;
            boolean delivered = false;

            while (!delivered) {
                // still stupid. maybe use real reentrant lock here?
                synchronized (rc.locker) {
                    res = RetransmissionHandler.getTransmissionStatus(rc.getPublication().offer(buffer));
                }

                switch (res) {
                    case NOT_CONNECTED: {
                        if (!rc.getActivated().get()) {
                            retr++;

                            if (retr > 20)
                                throw new ND4JIllegalStateException(
                                                "Can't connect to Shard: [" + rc.getPublication().channel() + "]");

                            try {
                                //Thread.sleep(voidConfiguration.getRetransmitTimeout());
                                LockSupport.parkNanos(voidConfiguration.getRetransmitTimeout() * 1000000);
                            } catch (Exception e) {
                                throw new RuntimeException(e);
                            }
                        } else {
                            throw new ND4JIllegalStateException("Shards reassignment is to be implemented yet");
                        }
                    }
                        break;
                    case ADMIN_ACTION:
                    case BACKPRESSURE: {
                        try {
                            //Thread.sleep(voidConfiguration.getRetransmitTimeout());
                            LockSupport.parkNanos(voidConfiguration.getRetransmitTimeout() * 1000000);
                        } catch (Exception e) {
                            throw new RuntimeException(e);
                        }
                    }
                        break;
                    case MESSAGE_SENT:
                        delivered = true;
                        rc.getActivated().set(true);
                        break;
                }
            }
        });

        //s   log.info("RESULT: {}", builder.toString());
    }

    /**
     * This method implements Shard -> Shards comms
     *
     * @param message
     */
    @Override
    protected void sendCoordinationCommand(VoidMessage message) {

        //        log.info("Sending [{}] to all Shards...", message.getClass().getSimpleName());
        message.setOriginatorId(this.originatorId);

        // if we're the only shard - we just put message into the queue
        if (nodeRole == NodeRole.SHARD && voidConfiguration.getNumberOfShards() == 1) {
            try {
                messages.put(message);
                return;
            } catch (Exception e) {
                throw new RuntimeException(e);
            }
        }

        final DirectBuffer buffer = message.asUnsafeBuffer();

        // TODO: check which approach is faster, lambda, direct roll through list, or queue approach
        shards.parallelStream().forEach((rc) -> {
            RetransmissionHandler.TransmissionStatus res;
            long retr = 0;
            boolean delivered = false;

            long address = HashUtil.getLongHash(rc.getIp() + ":" + rc.getPort());
            if (originatorId == address) {
                // this is local delivery
                try {
                    messages.put(message);
                } catch (Exception e) {
                    throw new RuntimeException(e);
                }
                return;
            }

            //      log.info("Trying to send [{}] to {}", message.getClass().getSimpleName(), address);
            while (!delivered) {
                synchronized (rc.locker) {
                    res = RetransmissionHandler.getTransmissionStatus(rc.getPublication().offer(buffer));
                }

                switch (res) {
                    case NOT_CONNECTED: {
                        if (!rc.getActivated().get()) {
                            retr++;

                            if (retr > 20)
                                throw new ND4JIllegalStateException(
                                                "Can't connect to Shard: [" + rc.getPublication().channel() + "]");

                            try {
                                Thread.sleep(voidConfiguration.getRetransmitTimeout());
                            } catch (Exception e) {
                            }
                        } else {
                            throw new ND4JIllegalStateException("Shards reassignment is to be implemented yet");
                        }
                    }
                        break;
                    case ADMIN_ACTION:
                    case BACKPRESSURE: {
                        try {
                            Thread.sleep(voidConfiguration.getRetransmitTimeout());
                        } catch (Exception e) {
                        }
                    }
                        break;
                    case MESSAGE_SENT:
                        delivered = true;
                        rc.getActivated().set(true);
                        break;
                }

                if (!delivered)
                    log.info("Attempting to resend message");
            }
        });
    }

    /**
     * This method implements Shard -> Client comms
     *
     * @param message
     */
    @Override
    protected void sendFeedbackToClient(VoidMessage message) {
        /*
            PLEASE NOTE: In this case we don't change target. We just discard message if something goes wrong.
         */
        // TODO: discard message if it's not sent for enough time?
        long targetAddress = message.getOriginatorId();

        if (targetAddress == originatorId) {
            completed.put(message.getTaskId(), (MeaningfulMessage) message);
            return;
        }

        RetransmissionHandler.TransmissionStatus result;

        //log.info("sI_{} trying to send back {}/{}", shardIndex, targetAddress, message.getClass().getSimpleName());

        RemoteConnection connection = clients.get(targetAddress);
        boolean delivered = false;

        if (connection == null) {
            log.info("Can't get client with address [{}]", targetAddress);
            log.info("Known clients: {}", clients.keySet());
            throw new RuntimeException();
        }

        while (!delivered) {
            synchronized (connection.locker) {
                result = RetransmissionHandler
                                .getTransmissionStatus(connection.getPublication().offer(message.asUnsafeBuffer()));
            }

            switch (result) {
                case ADMIN_ACTION:
                case BACKPRESSURE: {
                    try {
                        Thread.sleep(voidConfiguration.getRetransmitTimeout());
                    } catch (Exception e) {
                    }
                }
                    break;
                case NOT_CONNECTED: {
                    // client dead? sleep and forget
                    // TODO: we might want to delay this message & move it to separate queue?
                    try {
                        Thread.sleep(voidConfiguration.getRetransmitTimeout());
                    } catch (Exception e) {
                    }
                }
                // do not break here, we can't do too much here, if client is dead
                case MESSAGE_SENT:
                    delivered = true;
                    break;
            }
        }
    }

    @Override
    public int numberOfKnownClients() {
        return clients.size();
    }

    @Override
    public int numberOfKnownShards() {
        return shards.size();
    }

    @Override
    protected void shutdownSilent() {
        // closing shards
        shards.forEach((rc) -> {
            rc.getPublication().close();
        });

        // closing clients connections
        clients.values().forEach((rc) -> {
            rc.getPublication().close();
        });

        subscriptionForClients.close();

        aeron.close();
        context.close();
        driver.close();
    }

    @Override
    public void shutdown() {
        runner.set(false);

        if (threadB != null)
            threadB.interrupt();

        if (threadA != null)
            threadA.interrupt();

        shutdownSilent();
    }


    @Override
    protected void sendCommandToShard(VoidMessage message) {
        // fastpath for local Shard
        if (nodeRole == NodeRole.SHARD && message instanceof TrainingMessage) {
            router.setOriginator(message);
            message.setTargetId(getShardIndex());

            try {
                messages.put(message);
            } catch (Exception e) {
                throw new RuntimeException(e);
            }
            return;
        }

        //log.info("sI_{} {}: message class: {}", shardIndex, nodeRole, message.getClass().getSimpleName());

        RetransmissionHandler.TransmissionStatus result;

        int targetShard = router.assignTarget(message);

        //log.info("Sending message {} to shard {}", message.getClass().getSimpleName(), targetShard);
        boolean delivered = false;
        RemoteConnection connection = shards.get(targetShard);

        while (!delivered) {
            synchronized (connection.locker) {
                result = RetransmissionHandler
                                .getTransmissionStatus(connection.getPublication().offer(message.asUnsafeBuffer()));
            }

            switch (result) {
                case BACKPRESSURE:
                case ADMIN_ACTION: {
                    // we just sleep, and retransmit again later
                    try {
                        Thread.sleep(voidConfiguration.getRetransmitTimeout());
                    } catch (Exception e) {
                    }
                }
                    break;
                case NOT_CONNECTED:
                    /*
                        two possible cases here:
                        1) We hadn't sent any messages to this Shard before
                        2) It was active before, and suddenly died
                     */
                    if (!connection.getActivated().get()) {
                        // wasn't initialized before, just sleep and re-transmit
                        try {
                            Thread.sleep(voidConfiguration.getRetransmitTimeout());
                        } catch (Exception e) {
                        }
                    } else {
                        throw new ND4JIllegalStateException("Shards reassignment is to be implemented yet");
                    }
                    break;
                case MESSAGE_SENT:
                    delivered = true;
                    connection.getActivated().set(true);
                    break;
            }
        }
    }

    /**
     * This message handler is responsible for receiving messages on any side of p2p network
     *
     * @param buffer
     * @param offset
     * @param length
     * @param header
     */
    protected void jointMessageHandler(DirectBuffer buffer, int offset, int length, Header header) {
        /**
         *  All incoming messages here are supposed to be "just messages", only unicast communication
         *  All of them should implement MeaningfulMessage interface
         */

        byte[] data = new byte[length];
        buffer.getBytes(offset, data);

        VoidMessage message = VoidMessage.fromBytes(data);

        //        log.info("sI_{} received message: {}", shardIndex, message.getClass().getSimpleName());

        //if (messages.size() > 500)
        //    log.info("sI_{} got {} messages", shardIndex, messages.size());

        if (message instanceof MeaningfulMessage) {
            MeaningfulMessage msg = (MeaningfulMessage) message;
            completed.put(message.getTaskId(), msg);
        } else if (message instanceof RequestMessage) {
            try {
                messages.put((RequestMessage) message);
            } catch (InterruptedException e) {
                // do nothing
            } catch (Exception e) {
                throw new RuntimeException(e);
            }
        } else if (message instanceof DistributedMessage) {
            try {
                messages.put((DistributedMessage) message);
            } catch (InterruptedException e) {
                // do nothing
            } catch (Exception e) {
                throw new RuntimeException(e);
            }
        } else if (message instanceof TrainingMessage) {
            try {
                messages.put((TrainingMessage) message);
            } catch (InterruptedException e) {
                // do nothing
            } catch (Exception e) {
                throw new RuntimeException(e);
            }
        } else if (message instanceof VoidAggregation) {
            try {
                messages.put((VoidAggregation) message);
            } catch (InterruptedException e) {
                // do nothing
            } catch (Exception e) {
                throw new RuntimeException(e);
            }
        } else if (message instanceof Frame) {
            try {
                messages.put((Frame) message);
            } catch (InterruptedException e) {
                // do nothing
            } catch (Exception e) {
                throw new RuntimeException(e);
            }
        } else {
            log.info("Unknown message: {}", message.getClass().getSimpleName());
        }
    }

    @Override
    public void launch(@NonNull ThreadingModel threading) {
        super.launch(threading);

        // send introductory message
        //        if (nodeRole == NodeRole.CLIENT) {
        //            shards.parallelStream().forEach((rc) -> {
        IntroductionRequestMessage irm = new IntroductionRequestMessage(getIp(), getPort());
        irm.setTargetId((short) -1);
        sendCoordinationCommand(irm);
        //            });
        //        }
    }


    @Override
    public synchronized void addShard(String ip, int port) {
        Long hash = HashUtil.getLongHash(ip + ":" + port);

        RemoteConnection connection = RemoteConnection.builder().ip(ip).port(port)
                        .publication(aeron.addPublication("aeron:udp?endpoint=" + ip + ":" + port,
                                        voidConfiguration.getStreamId()))
                        .longHash(hash).locker(new Object()).activated(new AtomicBoolean(false)).build();

        log.info("sI_{} {}: Adding SHARD: [{}] to {}:{}", shardIndex, nodeRole, hash, ip, port);
        shards.add(connection);
    }

    @Override
    public synchronized void addClient(String ip, int port) {
        Long hash = HashUtil.getLongHash(ip + ":" + port);
        if (clients.containsKey(hash))
            return;

        RemoteConnection connection = RemoteConnection.builder().ip(ip).port(port)
                        .publication(aeron.addPublication("aeron:udp?endpoint=" + ip + ":" + port,
                                        voidConfiguration.getStreamId()))
                        .longHash(hash).locker(new Object()).activated(new AtomicBoolean(false)).build();


        log.info("sI_{} {}: Adding connection: [{}] to {}:{}", shardIndex, nodeRole, hash, ip, port);
        this.clients.put(hash, connection);
        log.info("sI_{} {}: Known clients: {}", shardIndex, nodeRole, clients.keySet());
    }


    @Data
    @Builder
    public static class RemoteConnection {
        private String ip;
        private int port;
        private Publication publication;
        private Object locker;
        private AtomicBoolean activated;
        protected long longHash;



        public static class RemoteConnectionBuilder {
            private Object locker = new Object();
            private AtomicBoolean activated = new AtomicBoolean();
        }
    }

}
