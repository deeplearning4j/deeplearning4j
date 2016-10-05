package org.nd4j.aeron.ipc;
/*
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

import io.aeron.Image;
import io.aeron.Subscription;
import io.aeron.logbuffer.FragmentHandler;
import io.aeron.protocol.HeaderFlyweight;
import org.agrona.LangUtil;
import org.agrona.concurrent.BusySpinIdleStrategy;
import org.agrona.concurrent.IdleStrategy;

import java.util.concurrent.atomic.AtomicBoolean;
import java.util.function.Consumer;

/**
 * Utility functions for samples
 */
public class AeronUtil {
    /**
     * Aeron channel generation
     * @param host the host
     * @param port the port
     * @return the aeron channel via udp
     */
    public static String aeronChannel(String host,int port) {
        return String.format("aeron:udp?endpoint=%s:%d",host,port);
    }
    /**
     * Return a reusable, parameterised event loop that calls a default idler when no messages are received
     *
     * @param fragmentHandler to be called back for each message.
     * @param limit           passed to {@link Subscription#poll(FragmentHandler, int)}
     * @param running         indication for loop
     * @return loop function
     */
    public static Consumer<Subscription> subscriberLoop(
            final FragmentHandler fragmentHandler, final int limit, final AtomicBoolean running,final AtomicBoolean launched) {
        final IdleStrategy idleStrategy = new BusySpinIdleStrategy();
        return subscriberLoop(fragmentHandler, limit, running, idleStrategy,launched);
    }

    /**
     * Return a reusable, parameterized event
     * loop that calls and idler
     * when no messages are received
     *
     * @param fragmentHandler to be called back for each message.
     * @param limit           passed to {@link Subscription#poll(FragmentHandler, int)}
     * @param running         indication for loop
     * @param idleStrategy    to use for loop
     * @return loop function
     */
    public static Consumer<Subscription> subscriberLoop(
            final FragmentHandler fragmentHandler,
            final int limit,
            final AtomicBoolean running,
            final IdleStrategy idleStrategy,final AtomicBoolean launched)
    {
        return (subscription) -> {
            try {
                while (running.get()) {
                    idleStrategy.idle(subscription.poll(fragmentHandler, limit));
                    launched.set(true);
                }
            }
            catch (final Exception ex) {
                LangUtil.rethrowUnchecked(ex);
            }
        };
    }

    /**
     * Return a reusable, parameterized {@link FragmentHandler} that prints to stdout
     *
     * @param streamId to show when printing
     * @return subscription data handler function that prints the message contents
     */
    public static FragmentHandler printStringMessage(final int streamId)
    {
        return (buffer, offset, length, header) ->
        {
            final byte[] data = new byte[length];
            buffer.getBytes(offset, data);

            System.out.println(String.format(
                    "Message to stream %d from session %d (%d@%d) <<%s>>",
                    streamId, header.sessionId(), length, offset, new String(data)));
        };
    }


    /**
     * Generic error handler that just prints message to stdout.
     *
     * @param channel   for the error
     * @param streamId  for the error
     * @param sessionId for the error, if source
     * @param message   indicating what the error was
     * @param cause     of the error
     */
    public static void printError(
            final String channel,
            final int streamId,
            final int sessionId,
            final String message,
            final HeaderFlyweight cause)
    {
        System.out.println(message);
    }

    /**
     * Print the rates to stdout
     *
     * @param messagesPerSec being reported
     * @param bytesPerSec    being reported
     * @param totalMessages  being reported
     * @param totalBytes     being reported
     */
    public static void printRate(
            final double messagesPerSec,
            final double bytesPerSec,
            final long totalMessages,
            final long totalBytes)
    {
        System.out.println(String.format(
                "%.02g msgs/sec, %.02g bytes/sec, totals %d messages %d MB",
                messagesPerSec, bytesPerSec, totalMessages, totalBytes / (1024 * 1024)));
    }

    /**
     * Print the information for an available image to stdout.
     *
     * @param image that has been created
     */
    public static void printAvailableImage(final Image image)
    {
        final Subscription subscription = image.subscription();
        System.out.println(String.format(
                "Available image on %s streamId=%d sessionId=%d from %s",
                subscription.channel(), subscription.streamId(), image.sessionId(), image.sourceIdentity()));
    }

    /**
     * Print the information for an unavailable image to stdout.
     *
     * @param image that has gone inactive
     */
    public static void printUnavailableImage(final Image image)
    {
        final Subscription subscription = image.subscription();
        System.out.println(String.format(
                "Unavailable image on %s streamId=%d sessionId=%d",
                subscription.channel(), subscription.streamId(), image.sessionId()));
    }
}