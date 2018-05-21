/*-
 * Copyright 2014 - 2016 Real Logic Ltd.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package org.nd4j.aeron.util;

import io.aeron.CncFileDescriptor;
import io.aeron.CommonContext;
import io.aeron.driver.status.ChannelEndpointStatus;
import org.agrona.DirectBuffer;
import org.agrona.IoUtil;
import org.agrona.concurrent.SigInt;
import org.agrona.concurrent.status.CountersReader;

import java.io.File;
import java.io.PrintStream;
import java.nio.MappedByteBuffer;
import java.util.Date;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.function.Supplier;
import java.util.regex.Pattern;

import static io.aeron.CncFileDescriptor.*;
import static io.aeron.driver.status.PublisherLimit.PUBLISHER_LIMIT_TYPE_ID;
import static io.aeron.driver.status.ReceiveChannelStatus.RECEIVE_CHANNEL_STATUS_TYPE_ID;
import static io.aeron.driver.status.ReceiverPos.RECEIVER_POS_TYPE_ID;
import static io.aeron.driver.status.SendChannelStatus.SEND_CHANNEL_STATUS_TYPE_ID;
import static io.aeron.driver.status.StreamPositionCounter.*;
import static io.aeron.driver.status.SystemCounterDescriptor.SYSTEM_COUNTER_TYPE_ID;

/**
 * Tool for printing out Aeron counters. A command-and-control (cnc) file is maintained by media driver
 * in shared memory. This application reads the the cnc file and prints the counters. Layout of the cnc file is
 * described in {@link CncFileDescriptor}.
 *
 * This tool accepts filters on the command line, e.g. for connections only see example below:
 *
 * <code>
 *     java -cp aeron-samples/build/libs/samples.jar io.aeron.samples.AeronStat opType=[1-4] identity=12345
 * </code>
 */
public class AeronStat {
    /**
     * Types of the counters.
     * <ul>
     *     <li>0: System Counters</li>
     *     <li>1 - 4: Stream Positions</li>
     * </ul>
     */
    private static final String COUNTER_TYPE_ID = "opType";

    /**
     * The identity of each counter that can either be the system counter id or registration id for positions.
     */
    private static final String COUNTER_IDENTITY = "identity";

    /**
     * Session id filter to be used for position counters.
     */
    private static final String COUNTER_SESSION_ID = "session";

    /**
     * Stream id filter to be used for position counters.
     */
    private static final String COUNTER_STREAM_ID = "stream";

    /**
     * Channel filter to be used for position counters.
     */
    private static final String COUNTER_CHANNEL = "channel";

    private static final int ONE_SECOND = 1_000;

    private final CountersReader counters;
    private final Pattern typeFilter;
    private final Pattern identityFilter;
    private final Pattern sessionFilter;
    private final Pattern streamFilter;
    private final Pattern channelFilter;

    public AeronStat(final CountersReader counters, final Pattern typeFilter, final Pattern identityFilter,
                    final Pattern sessionFilter, final Pattern streamFilter, final Pattern channelFilter) {
        this.counters = counters;
        this.typeFilter = typeFilter;
        this.identityFilter = identityFilter;
        this.sessionFilter = sessionFilter;
        this.streamFilter = streamFilter;
        this.channelFilter = channelFilter;
    }

    public AeronStat(final CountersReader counters) {
        this.counters = counters;
        this.typeFilter = null;
        this.identityFilter = null;
        this.sessionFilter = null;
        this.streamFilter = null;
        this.channelFilter = null;
    }

    public static CountersReader mapCounters() {
        final File cncFile = CommonContext.newDefaultCncFile();
        System.out.println("Command `n Control file " + cncFile);

        final MappedByteBuffer cncByteBuffer = IoUtil.mapExistingFile(cncFile, "cnc");
        final DirectBuffer cncMetaData = createMetaDataBuffer(cncByteBuffer);
        final int cncVersion = cncMetaData.getInt(cncVersionOffset(0));

        if (CncFileDescriptor.CNC_VERSION != cncVersion) {
            throw new IllegalStateException("CnC version not supported: file version=" + cncVersion);
        }

        return new CountersReader(createCountersMetaDataBuffer(cncByteBuffer, cncMetaData),
                        createCountersValuesBuffer(cncByteBuffer, cncMetaData));
    }

    public static void main(final String[] args) throws Exception {
        Pattern typeFilter = null;
        Pattern identityFilter = null;
        Pattern sessionFilter = null;
        Pattern streamFilter = null;
        Pattern channelFilter = null;

        if (0 != args.length) {
            checkForHelp(args);

            for (final String arg : args) {
                final int equalsIndex = arg.indexOf('=');
                if (-1 == equalsIndex) {
                    System.out.println("Arguments must be in opName=pattern format: Invalid '" + arg + "'");
                    return;
                }

                final String argName = arg.substring(0, equalsIndex);
                final String argValue = arg.substring(equalsIndex + 1);

                switch (argName) {
                    case COUNTER_TYPE_ID:
                        typeFilter = Pattern.compile(argValue);
                        break;

                    case COUNTER_IDENTITY:
                        identityFilter = Pattern.compile(argValue);
                        break;

                    case COUNTER_SESSION_ID:
                        sessionFilter = Pattern.compile(argValue);
                        break;

                    case COUNTER_STREAM_ID:
                        streamFilter = Pattern.compile(argValue);
                        break;

                    case COUNTER_CHANNEL:
                        channelFilter = Pattern.compile(argValue);
                        break;

                    default:
                        System.out.println("Unrecognised argument: '" + arg + "'");
                        return;
                }
            }
        }

        final AeronStat aeronStat = new AeronStat(mapCounters(), typeFilter, identityFilter, sessionFilter,
                        streamFilter, channelFilter);
        final AtomicBoolean running = new AtomicBoolean(true);
        SigInt.register(() -> running.set(false));

        while (running.get()) {
            System.out.print("\033[H\033[2J");

            System.out.format("%1$tH:%1$tM:%1$tS - Aeron Stat%n", new Date());
            System.out.println("=========================");

            aeronStat.print(System.out);
            System.out.println("--");

            Thread.sleep(ONE_SECOND);
        }
    }

    public void print(final PrintStream out) {
        counters.forEach((counterId, typeId, keyBuffer, label) -> {
            if (filter(typeId, keyBuffer)) {
                final long value = counters.getCounterValue(counterId);
                out.format("%3d: %,20d - %s%n", counterId, value, label);
            }
        });
    }

    private static void checkForHelp(final String[] args) {
        for (final String arg : args) {
            if ("-?".equals(arg) || "-h".equals(arg) || "-help".equals(arg)) {
                System.out.format("Usage: [-Daeron.dir=<directory containing CnC file>] AeronStat%n"
                                + "\tfilter by optional regex patterns:%n" + "\t[opType=<pattern>]%n"
                                + "\t[identity=<pattern>]%n" + "\t[sessionId=<pattern>]%n" + "\t[streamId=<pattern>]%n"
                                + "\t[channel=<pattern>]%n");

                System.exit(0);
            }
        }
    }

    private boolean filter(final int typeId, final DirectBuffer keyBuffer) {
        if (!match(typeFilter, () -> Integer.toString(typeId))) {
            return false;
        }

        if (SYSTEM_COUNTER_TYPE_ID == typeId && !match(identityFilter, () -> Integer.toString(keyBuffer.getInt(0)))) {
            return false;
        } else if (typeId >= PUBLISHER_LIMIT_TYPE_ID && typeId <= RECEIVER_POS_TYPE_ID) {
            if (!match(identityFilter, () -> Long.toString(keyBuffer.getLong(REGISTRATION_ID_OFFSET)))
                            || !match(sessionFilter, () -> Integer.toString(keyBuffer.getInt(SESSION_ID_OFFSET)))
                            || !match(streamFilter, () -> Integer.toString(keyBuffer.getInt(STREAM_ID_OFFSET)))
                            || !match(channelFilter, () -> keyBuffer.getStringUtf8(CHANNEL_OFFSET))) {
                return false;
            }
        } else if (typeId >= SEND_CHANNEL_STATUS_TYPE_ID && typeId <= RECEIVE_CHANNEL_STATUS_TYPE_ID) {
            if (!match(channelFilter, () -> keyBuffer.getStringUtf8(ChannelEndpointStatus.CHANNEL_OFFSET))) {
                return false;
            }
        }

        return true;
    }

    private static boolean match(final Pattern pattern, final Supplier<String> supplier) {
        return null == pattern || pattern.matcher(supplier.get()).find();
    }
}
