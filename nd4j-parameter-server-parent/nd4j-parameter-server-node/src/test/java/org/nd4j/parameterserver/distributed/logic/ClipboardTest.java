package org.nd4j.parameterserver.distributed.logic;

import lombok.extern.slf4j.Slf4j;
import org.apache.commons.lang3.RandomUtils;
import org.junit.After;
import org.junit.Before;
import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.util.ArrayUtil;
import org.nd4j.parameterserver.distributed.messages.aggregations.VectorAggregation;
import org.nd4j.parameterserver.distributed.messages.aggregations.VoidAggregation;

import java.util.*;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicInteger;

import static org.junit.Assert.*;

/**
 * @author raver119@gmail.com
 */
@Slf4j
public class ClipboardTest {
    @Before
    public void setUp() throws Exception {

    }

    @After
    public void tearDown() throws Exception {

    }

    @Test
    public void testPin1() throws Exception {
        Clipboard clipboard = new Clipboard();

        Random rng = new Random(12345L);

        for (int i = 0; i < 100; i++) {
            VectorAggregation aggregation = new VectorAggregation(rng.nextLong(), (short) 100, (short) i, Nd4j.create(5));

            clipboard.pin(aggregation);
        }

        assertEquals(false, clipboard.hasCandidates());
        assertEquals(0, clipboard.getNumberOfCompleteStacks());
        assertEquals(100, clipboard.getNumberOfPinnedStacks());
    }

    @Test
    public void testPin2() throws Exception {
        Clipboard clipboard = new Clipboard();

        Random rng = new Random(12345L);

        Long validId = 123L;

        short shardIdx = 0;
        for (int i = 0; i < 300; i++) {
            VectorAggregation aggregation = new VectorAggregation(rng.nextLong(), (short) 100, (short) 1, Nd4j.create(5));

            // imitating valid
            if (i % 2 == 0 && shardIdx < 100) {
                aggregation.setTaskId(validId);
                aggregation.setShardIndex(shardIdx++);
            }

            clipboard.pin(aggregation);
        }

        VoidAggregation aggregation = clipboard.getStackFromClipboard(validId);
        assertNotEquals(null, aggregation);

        assertEquals(0, aggregation.getMissingChunks());

        assertEquals(true, clipboard.hasCandidates());
        assertEquals(1, clipboard.getNumberOfCompleteStacks());
    }

    /**
     * This test is VERY important, here we fill clipboard from one thread, and reading from other.
     *
     * Basically, we're trying to imitate real network environment
     *
     * @throws Exception
     */
    @Test
    public void testMultithreadedClipboard1() throws Exception {
        final Clipboard clipboard = new Clipboard();
        final int NUM_SHARDS = 10;
        final int MESSAGE_SIZE = 150;
        final int NUM_MESSAGES = 100;
        final INDArray exp = Nd4j.linspace(1, MESSAGE_SIZE, MESSAGE_SIZE);

        final AtomicBoolean producerPassed = new AtomicBoolean(false);
        final AtomicBoolean consumerPassed = new AtomicBoolean(false);
        final AtomicInteger messagesCount = new AtomicInteger(0);


        // this thread gets aggregations from clipboard, and we launch it first
        Thread consumer = new Thread(() -> {

            while (messagesCount.get() < NUM_MESSAGES) {
                if (clipboard.hasCandidates()) {
                    VoidAggregation aggregation = clipboard.nextCandidate();
                    INDArray result = aggregation.getAccumulatedResult();

                    assertEquals(exp, result);
                    log.info("Message {} received...", messagesCount.incrementAndGet());
                }
            }

            consumerPassed.set(true);
        });

        consumer.setDaemon(true);
        consumer.start();

        // obvously, this thread fills clipboard with shards
        Thread producer = new Thread(() -> {
            // at first we generate all aggregations, so later we'll send data completely out of order (mixing chunks & aggregations)
            Map<Integer, List<VoidAggregation>> aggregations = new HashMap<>();
            for (int m = 0; m < NUM_MESSAGES; m++) {
                List<VoidAggregation> list = new ArrayList<>();

                int stepSize =  MESSAGE_SIZE / NUM_SHARDS;
                for (int s = 0; s < NUM_SHARDS; s++) {
                    INDArray payload = Nd4j.linspace((stepSize * s) + 1, (stepSize * (s+1)), stepSize);

                    VoidAggregation aggregation = new VectorAggregation(m, (short) NUM_SHARDS, (short) s, payload);
                    list.add(aggregation);
                }

                aggregations.put(m, list);
                assertEquals(NUM_SHARDS, list.size());
            }

            // at this point we're sure, that we have whole test set formed up before "sending" messages
            assertEquals(NUM_MESSAGES, aggregations.size());

            AtomicInteger senderCount = new AtomicInteger(0);
            while (aggregations.size() > 0) {
                List<Integer> activeKeys = new ArrayList<>(aggregations.keySet());

                int keyIndex = ArrayUtil.getRandomElement(activeKeys);
                List<VoidAggregation> activeList = aggregations.get(keyIndex);
                int randomIndex = RandomUtils.nextInt(0, activeList.size());

                VoidAggregation aggregation = activeList.get(randomIndex);

                // removing it from list
                activeList.remove(randomIndex);

                // pin it to the clipboard
                boolean isLast = clipboard.pin(aggregation);
                senderCount.incrementAndGet();

                if (isLast)
                    aggregations.remove(keyIndex);
            }

            assertEquals(NUM_MESSAGES * NUM_SHARDS, senderCount.get());

            producerPassed.set(true);
        });

        producer.setDaemon(true);
        producer.start();


        producer.join();
        consumer.join();


        assertEquals(true, producerPassed.get());
        assertEquals(true, consumerPassed.get());
    }
}