package org.nd4j.parameterserver.distributed.transport;

import io.aeron.Aeron;
import io.aeron.FragmentAssembler;
import io.aeron.Publication;
import io.aeron.driver.MediaDriver;
import io.aeron.logbuffer.Header;
import lombok.*;
import lombok.extern.slf4j.Slf4j;
import org.agrona.DirectBuffer;
import org.nd4j.linalg.exception.ND4JIllegalStateException;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.io.StringUtils;
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
    @Getter @Setter protected ClientRouter router;

    public RoutedTransport(){
        log.info("Initializing RoutedTransport");
    }

    @Override
    public void init(@NonNull VoidConfiguration voidConfiguration, @NonNull Clipboard clipboard, @NonNull NodeRole role, @NonNull String localIp, short shardIndex) {
        this.nodeRole = role;
        this.clipboard = clipboard;
        this.voidConfiguration = voidConfiguration;
        this.shardIndex = shardIndex;
        this.messages = new LinkedBlockingQueue<>(128);

        setProperty("aeron.client.liveness.timeout", "30000000000");

        context = new Aeron.Context()
                .publicationConnectionTimeout(30000000000L)
                .driverTimeoutMs(30000)
                .keepAliveInterval(100000000);

        driver = MediaDriver.launchEmbedded();
        context.aeronDirectoryName(driver.aeronDirectoryName());
        aeron = Aeron.connect(context);

        if (router == null)
            router = new InterleavedRouter();


        /*
            Regardless of current role, we raise subscription for incoming messages channel
         */
        ip = localIp;
        unicastChannelUri = "aeron:udp?endpoint=" + ip + ":" + port;
        subscriptionForClients = aeron.addSubscription(unicastChannelUri, voidConfiguration.getStreamId());

        messageHandlerForClients = new FragmentAssembler(
                (buffer, offset, length, header) -> jointMessageHandler(buffer, offset, length, header)
        );

        /*
            Now, regardless of curent role, we set up publication channel to each shard
         */
        String shardChannelUri = null;
        String remoteIp = null;
        int remotePort = 0;
        for (String ip: voidConfiguration.getShardAddresses()){
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

            RemoteConnection connection = RemoteConnection.builder()
                    .ip(remoteIp)
                    .port(remotePort)
                    .publication(publication)
                    .locker(new Object())
                    .build();

            shards.add(connection);
        }



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
    }

    /**
     * This method implements Shard -> Shards comms
     *
     * @param message
     */
    @Override
    protected void sendCoordinationCommand(VoidMessage message) {

        // TODO: check which approach is faster, lambda, direct roll through list, or queue approach
        shards.parallelStream().forEach((rc) ->{
            RetransmissionHandler.TransmissionStatus res;
            long retr = 0;
            boolean delivered = false;

            while (!delivered) {
                synchronized (rc.locker) {
                    res = RetransmissionHandler.getTransmissionStatus(rc.getPublication().offer(message.asUnsafeBuffer()));
                }

                switch (res) {
                    case NOT_CONNECTED: {
                            if (!rc.getActivated().get()) {
                                retr++;

                                if (retr > 20)
                                    throw new ND4JIllegalStateException("Can't connect to Shard: [" + rc.getPublication().channel() + "]");

                                try {
                                    Thread.sleep(voidConfiguration.getRetransmitTimeout());
                                } catch (Exception e) {}
                            } else {
                                throw new ND4JIllegalStateException("Shards reassignment is to be implemented yet");
                            }
                        }
                        break;
                    case ADMIN_ACTION:
                    case BACKPRESSURE: {
                            try {
                                Thread.sleep(voidConfiguration.getRetransmitTimeout());
                            } catch (Exception e) {}
                        }
                        break;
                    case MESSAGE_SENT:
                        delivered = true;
                        rc.getActivated().set(true);
                        break;
                }
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
        RetransmissionHandler.TransmissionStatus result;

        //log.info("sI_{} trying to send back {}", shardIndex, message.getClass().getSimpleName());

        RemoteConnection connection = clients.get(targetAddress);
        boolean delivered = false;

        if (connection == null) {
            log.info("Can't get client with address [{}]", targetAddress);
            throw new RuntimeException();
        }

        while (!delivered) {
            synchronized (connection.locker) {
                result = RetransmissionHandler.getTransmissionStatus(connection.getPublication().offer(message.asUnsafeBuffer()));
            }

            switch (result) {
                case ADMIN_ACTION:
                case BACKPRESSURE: {
                        try {
                            Thread.sleep(voidConfiguration.getRetransmitTimeout());
                        } catch (Exception e) { }
                    }
                    break;
                case NOT_CONNECTED: {
                    // client dead? sleep and forget
                    // TODO: we might want to delay this message & move it to separate queue?
                        try {
                            Thread.sleep(voidConfiguration.getRetransmitTimeout());
                        } catch (Exception e) { }
                    }
                    // do not break here, we can't do too much here, if client is dead
                case MESSAGE_SENT:
                    delivered = true;
                    break;
            }
        }
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
        RetransmissionHandler.TransmissionStatus result;

        int targetShard = router.assignTarget(message);

        log.info("Sending message {} to shard {}", message.getClass().getSimpleName(), targetShard);
        boolean delivered = false;
        RemoteConnection connection = shards.get(targetShard);

        while (!delivered) {
            synchronized (connection.locker) {
                result = RetransmissionHandler.getTransmissionStatus(connection.getPublication().offer(message.asUnsafeBuffer()));
            }

            switch (result) {
                case BACKPRESSURE:
                case ADMIN_ACTION: {
                    // we just sleep, and retransmit again later
                        try {
                            Thread.sleep(voidConfiguration.getRetransmitTimeout());
                        } catch (Exception e) { }
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
                        } catch (Exception e) { }
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

        //log.info("sI_{} received message: {}", shardIndex, message.getClass().getSimpleName());

        if (messages.size() > 500)
            log.info("sI_{} got {} messages", shardIndex, messages.size());

        if (message instanceof MeaningfulMessage) {
            MeaningfulMessage msg = (MeaningfulMessage) message;
            completed.put(message.getTaskId(), msg);
        } else if (message instanceof RequestMessage) {
            try {
                messages.put((RequestMessage) message);
            } catch (Exception e) {
                throw new RuntimeException(e);
            }
        } else if (message instanceof DistributedMessage) {
            try {
                messages.put((DistributedMessage) message);
            } catch (Exception e) {
                throw new RuntimeException(e);
            }
        } else if (message instanceof TrainingMessage) {
            try {
                messages.put((TrainingMessage) message);
            } catch (Exception e) {
                throw new RuntimeException(e);
            }
        } else if (message instanceof VoidAggregation) {
            try {
                messages.put((VoidAggregation) message);
            } catch (Exception e) {
                throw new RuntimeException(e);
            }
        } else if (message instanceof Frame) {
            try {
                messages.put((Frame) message);
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
        if (nodeRole == NodeRole.CLIENT) {
//            shards.parallelStream().forEach((rc) -> {
                IntroductionRequestMessage irm = new IntroductionRequestMessage(getIp(), getPort());
                irm.setTargetId((short) -1);
                sendCoordinationCommand(irm);

//            });

        }
    }

    @Override
    public synchronized void addClient(String ip, int port) {
        Long hash = StringUtils.getLongHash(ip + ":" + port);
        if (clients.containsKey(hash))
            return;

        RemoteConnection connection = RemoteConnection.builder()
                .ip(ip)
                .port(port)
                .publication(aeron.addPublication("aeron:udp?endpoint=" + ip + ":" + port, voidConfiguration.getStreamId()))
                .locker(new Object())
                .activated(new AtomicBoolean(false))
                .build();


        log.info("sI_{}: Adding connection: [{}] to {}:{}", shardIndex, hash, ip, port);
        this.clients.put(hash, connection);
    }


    @Data
    @Builder
    public static class RemoteConnection {
        private String ip;
        private int port;
        private Publication publication;
        private Object locker;
        private AtomicBoolean activated;



        public static class RemoteConnectionBuilder {
            private Object locker = new Object();
            private AtomicBoolean activated = new AtomicBoolean();
        }
    }

}
