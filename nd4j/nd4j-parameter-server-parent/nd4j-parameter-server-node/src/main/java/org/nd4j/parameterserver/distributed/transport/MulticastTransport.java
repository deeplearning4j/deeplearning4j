/*******************************************************************************
 * Copyright (c) 2015-2018 Skymind, Inc.
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

package org.nd4j.parameterserver.distributed.transport;

import io.aeron.Aeron;
import io.aeron.FragmentAssembler;
import io.aeron.driver.MediaDriver;
import lombok.NonNull;
import lombok.extern.slf4j.Slf4j;
import org.agrona.CloseHelper;
import org.nd4j.linalg.exception.ND4JIllegalStateException;
import org.nd4j.parameterserver.distributed.conf.VoidConfiguration;
import org.nd4j.parameterserver.distributed.enums.NodeRole;
import org.nd4j.parameterserver.distributed.logic.completion.Clipboard;
import org.nd4j.parameterserver.distributed.messages.MeaningfulMessage;
import org.nd4j.parameterserver.distributed.messages.VoidMessage;

/**
 * Transport implementation based on Aeron UDP multicast
 *
 * PLEASE NOTE: This transport will NOT work on AWS or Azure out of box, due to Amazon/Microsoft restrictions within their networks.
 *
 * @author raver119@gmail.com
 */
@Slf4j
@Deprecated
public class MulticastTransport extends BaseTransport {
    protected String multicastChannelUri;

    public MulticastTransport() {
        // no-op
        log.info("Initializing MulticastTransport");
    }

    @Override
    public void init(@NonNull VoidConfiguration voidConfiguration, @NonNull Clipboard clipboard, @NonNull NodeRole role,
                    @NonNull String localIp, int localPort, short shardIndex) {
        if (voidConfiguration.getTtl() < 1)
            throw new ND4JIllegalStateException(
                            "For MulticastTransport you should have TTL >= 1, it won't work otherwise");

        if (voidConfiguration.getMulticastNetwork() == null || voidConfiguration.getMulticastNetwork().isEmpty())
            throw new ND4JIllegalStateException(
                            "For MulticastTransport you should provide IP from multicast network available/allowed in your environment, i.e.: 224.0.1.1");

        //shutdown hook
        super.init(voidConfiguration, clipboard, role, localIp, localPort, shardIndex);

        this.voidConfiguration = voidConfiguration;
        this.nodeRole = role;
        this.clipboard = clipboard;

        context = new Aeron.Context();

        driver = MediaDriver.launchEmbedded();

        context.aeronDirectoryName(driver.aeronDirectoryName());

        aeron = Aeron.connect(context);



        this.shardIndex = shardIndex;



        multicastChannelUri = "aeron:udp?endpoint=" + voidConfiguration.getMulticastNetwork() + ":"
                        + voidConfiguration.getMulticastPort();
        if (voidConfiguration.getMulticastInterface() != null && !voidConfiguration.getMulticastInterface().isEmpty())
            multicastChannelUri = multicastChannelUri + "|interface=" + voidConfiguration.getMulticastInterface();

        multicastChannelUri = multicastChannelUri + "|ttl=" + voidConfiguration.getTtl();

        if (voidConfiguration.getNumberOfShards() < 0)
            voidConfiguration.setNumberOfShards(voidConfiguration.getShardAddresses().size());

        switch (nodeRole) {
            case BACKUP:
            case SHARD:
                /*
                    In case of Shard, unicast address for communication is known in advance
                 */
                if (ip == null) {
                    ip = localIp;
                    port = voidConfiguration.getUnicastPort();
                }


                unicastChannelUri = "aeron:udp?endpoint=" + ip + ":" + port;
                log.info("Shard unicast URI: {}/{}", unicastChannelUri, voidConfiguration.getStreamId());

                // this channel will be used to receive batches from Clients
                subscriptionForShards = aeron.addSubscription(unicastChannelUri, voidConfiguration.getStreamId());

                // this channel will be used to send completion reports back to Clients
                publicationForClients = aeron.addPublication(multicastChannelUri, voidConfiguration.getStreamId() + 1);

                // this channel will be used for communication with other Shards
                publicationForShards = aeron.addPublication(multicastChannelUri, voidConfiguration.getStreamId() + 2);

                // this channel will be used to receive messages from other Shards
                subscriptionForClients =
                                aeron.addSubscription(multicastChannelUri, voidConfiguration.getStreamId() + 2);

                messageHandlerForShards = new FragmentAssembler((buffer, offset, length,
                                header) -> shardMessageHandler(buffer, offset, length, header));

                messageHandlerForClients = new FragmentAssembler(((buffer, offset, length,
                                header) -> internalMessageHandler(buffer, offset, length, header)));



                break;
            case CLIENT:
                ip = localIp;

                /*
                    In case of Client, unicast will be one of shards, picked up with random
                 */
                // FIXME: we don't want that

                String rts = voidConfiguration.getShardAddresses().get(0);//ArrayUtil.getRandomElement(configuration.getShardAddresses());
                String[] split = rts.split(":");
                if (split.length == 1) {
                    ip = rts;
                    port = voidConfiguration.getUnicastPort();
                } else {
                    ip = split[0];
                    port = Integer.valueOf(split[1]);
                }


                unicastChannelUri = "aeron:udp?endpoint=" + ip + ":" + port;
                //unicastChannelUri = "aeron:udp?endpoint=" + ip  + ":" + (configuration.getUnicastPort()) ;

                log.info("Client unicast URI: {}/{}", unicastChannelUri, voidConfiguration.getStreamId());

                /*
                 this channel will be used to send batches to Shards, it's 1:1 channel to one of the Shards
                */
                publicationForShards = aeron.addPublication(unicastChannelUri, voidConfiguration.getStreamId());

                // this channel will be used to receive completion reports from Shards
                subscriptionForClients =
                                aeron.addSubscription(multicastChannelUri, voidConfiguration.getStreamId() + 1);

                messageHandlerForClients = new FragmentAssembler((buffer, offset, length,
                                header) -> clientMessageHandler(buffer, offset, length, header));
                break;
            default:
                log.warn("Unknown role passed: {}", nodeRole);
                throw new RuntimeException();
        }



        // if that's local spark run - we don't need this
        if (voidConfiguration.getNumberOfShards() == 1 && nodeRole == NodeRole.SHARD)
            shutdownSilent();
    }

    /**
     * This command is possible to issue only from Shard
     *
     * @param message
     */
    @Override
    protected synchronized void sendCoordinationCommand(VoidMessage message) {
        if (nodeRole == NodeRole.SHARD && voidConfiguration.getNumberOfShards() == 1) {
            message.setTargetId((short) -1);
            messages.add(message);
            return;
        }

        //log.info("Sending CC: {}", message.getClass().getCanonicalName());

        message.setTargetId((short) -1);
        publicationForShards.offer(message.asUnsafeBuffer());
    }

    /**
     * This command is possible to issue only from Shard
     *
     * @param message
     */
    @Override
    protected synchronized void sendFeedbackToClient(VoidMessage message) {
        if (nodeRole == NodeRole.SHARD && voidConfiguration.getNumberOfShards() == 1
                        && message instanceof MeaningfulMessage) {
            message.setTargetId((short) -1);
            completed.put(message.getTaskId(), (MeaningfulMessage) message);
            return;
        }

        //log.info("Sending FC: {}", message.getClass().getCanonicalName());

        message.setTargetId((short) -1);
        publicationForClients.offer(message.asUnsafeBuffer());
    }
}
