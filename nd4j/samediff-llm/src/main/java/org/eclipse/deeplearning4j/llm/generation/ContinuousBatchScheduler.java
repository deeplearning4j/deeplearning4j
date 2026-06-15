/*
 *  ******************************************************************************
 *  *
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
 *  *  See the NOTICE file distributed with this work for additional
 *  *  information regarding copyright ownership.
 *  * Unless required by applicable law or agreed to in writing, software
 *  * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 *  * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 *  * License for the specific language governing permissions and limitations
 *  * under the License.
 *  *
 *  * SPDX-License-Identifier: Apache-2.0
 *  *****************************************************************************
 */

package org.eclipse.deeplearning4j.llm.generation;

import lombok.Getter;
import lombok.extern.slf4j.Slf4j;

import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.Deque;
import java.util.List;
import java.util.Queue;
import java.util.concurrent.ConcurrentLinkedQueue;

/**
 * Continuous batch scheduler for slot-based sequence management.
 *
 * <p>Unlike static batching (fixed batch for the entire generation), continuous
 * batching dynamically assigns and reclaims batch slots as sequences start and finish.
 * When a sequence finishes (EOS or max tokens), its slot is immediately freed and can
 * be assigned to a new request from the waiting queue — without waiting for the
 * entire batch to complete.</p>
 *
 * <h3>Design</h3>
 * <ul>
 *   <li><b>Fixed max batch size</b>: Pre-allocated for maxBatchSize slots. The decoder
 *       plan is compiled once for this size; active count varies.</li>
 *   <li><b>Slot-based management</b>: Each slot has a state (FREE, PREFILL, DECODE, DONE).
 *       Slots are assigned from a free list and returned on completion.</li>
 *   <li><b>Paged KV cache integration</b>: When a slot is freed, its KV blocks are
 *       returned to the pool via {@link PagedKVCache#freeSequence(int)}. New sequences
 *       get fresh blocks on demand — no dense KV tensor copying.</li>
 *   <li><b>Prefill/decode separation</b>: New sequences enter PREFILL state for
 *       prompt processing (variable sequence length), then transition to DECODE
 *       for autoregressive generation (seqLen=1 per step).</li>
 * </ul>
 *
 * <h3>Usage</h3>
 * <pre>{@code
 * ContinuousBatchScheduler scheduler = new ContinuousBatchScheduler(32);
 *
 * // Submit requests
 * scheduler.submit(new Request(promptTokens1, maxTokens));
 * scheduler.submit(new Request(promptTokens2, maxTokens));
 *
 * // Each decode step:
 * while (scheduler.hasWork()) {
 *     List<SlotAssignment> active = scheduler.getActiveSlots();
 *     // ... run decoder for active slots ...
 *     // ... check EOS, update tokens ...
 *     scheduler.finishSlot(slotIdx);     // frees slot + KV blocks
 *     scheduler.scheduleNewRequests();   // assigns waiting requests to free slots
 * }
 * }</pre>
 */
@Slf4j
public class ContinuousBatchScheduler {

    /**
     * State of a batch slot.
     */
    public enum SlotState {
        FREE,       // Available for new sequence
        PREFILL,    // Processing prompt (variable seqLen)
        DECODE,     // Autoregressive decoding (seqLen=1)
        DONE        // Finished, awaiting result collection
    }

    /**
     * A generation request waiting to be scheduled.
     */
    public static class Request {
        @Getter private final int[] promptTokenIds;
        @Getter private final int maxNewTokens;
        @Getter private final int requestId;
        private final long submitTimeNanos;

        public Request(int[] promptTokenIds, int maxNewTokens, int requestId) {
            this.promptTokenIds = promptTokenIds;
            this.maxNewTokens = maxNewTokens;
            this.requestId = requestId;
            this.submitTimeNanos = System.nanoTime();
        }

        public long getWaitTimeMs() {
            return (System.nanoTime() - submitTimeNanos) / 1_000_000;
        }
    }

    /**
     * Assignment of a request to a batch slot.
     */
    @Getter
    public static class SlotAssignment {
        private final int slotIndex;
        private final Request request;
        private SlotState state;
        private final List<Integer> generatedTokens;
        private int tokensGenerated;
        private GenerationResult.FinishReason finishReason;

        SlotAssignment(int slotIndex, Request request) {
            this.slotIndex = slotIndex;
            this.request = request;
            this.state = SlotState.PREFILL;
            this.generatedTokens = new ArrayList<>();
            this.tokensGenerated = 0;
        }

        public void addToken(int tokenId) {
            generatedTokens.add(tokenId);
            tokensGenerated++;
        }

        public boolean hasReachedMaxTokens() {
            return tokensGenerated >= request.maxNewTokens;
        }
    }

    @Getter private final int maxBatchSize;
    private final SlotAssignment[] slots;
    private final SlotState[] slotStates;
    private final Deque<Integer> freeSlots;
    private final Queue<Request> waitingQueue;
    private PagedKVCache kvCache;
    private int nextRequestId;

    /**
     * Create a continuous batch scheduler.
     *
     * @param maxBatchSize maximum number of concurrent sequences
     */
    public ContinuousBatchScheduler(int maxBatchSize) {
        this.maxBatchSize = maxBatchSize;
        this.slots = new SlotAssignment[maxBatchSize];
        this.slotStates = new SlotState[maxBatchSize];
        this.freeSlots = new ArrayDeque<>(maxBatchSize);
        this.waitingQueue = new ConcurrentLinkedQueue<>();
        this.nextRequestId = 0;

        for (int i = 0; i < maxBatchSize; i++) {
            slotStates[i] = SlotState.FREE;
            freeSlots.push(i);
        }
    }

    /**
     * Set the paged KV cache for block management.
     * When set, finishing a slot automatically frees its KV blocks.
     */
    public void setKvCache(PagedKVCache kvCache) {
        this.kvCache = kvCache;
    }

    /**
     * Submit a new generation request to the waiting queue.
     *
     * @param promptTokenIds the prompt token IDs
     * @param maxNewTokens maximum tokens to generate
     * @return request ID for tracking
     */
    public int submit(int[] promptTokenIds, int maxNewTokens) {
        int id = nextRequestId++;
        waitingQueue.add(new Request(promptTokenIds, maxNewTokens, id));
        return id;
    }

    /**
     * Submit a pre-built request.
     */
    public void submit(Request request) {
        waitingQueue.add(request);
    }

    /**
     * Assign waiting requests to free slots.
     *
     * @return list of newly assigned slots (for prefill processing)
     */
    public List<SlotAssignment> scheduleNewRequests() {
        List<SlotAssignment> newAssignments = new ArrayList<>();

        while (!freeSlots.isEmpty() && !waitingQueue.isEmpty()) {
            int slotIdx = freeSlots.pop();
            Request request = waitingQueue.poll();

            SlotAssignment assignment = new SlotAssignment(slotIdx, request);
            slots[slotIdx] = assignment;
            slotStates[slotIdx] = SlotState.PREFILL;

            newAssignments.add(assignment);
            log.debug("Assigned request {} to slot {} (wait: {}ms, prompt: {} tokens)",
                    request.getRequestId(), slotIdx, request.getWaitTimeMs(),
                    request.getPromptTokenIds().length);
        }

        return newAssignments;
    }

    /**
     * Transition a slot from PREFILL to DECODE after prompt processing.
     */
    public void transitionToDecode(int slotIdx) {
        if (slotStates[slotIdx] == SlotState.PREFILL) {
            slotStates[slotIdx] = SlotState.DECODE;
            slots[slotIdx].state = SlotState.DECODE;
        }
    }

    /**
     * Mark a slot as finished and return it to the free pool.
     * If a PagedKVCache is set, also frees the slot's KV blocks.
     *
     * @param slotIdx the slot to finish
     * @param reason why the sequence finished
     */
    public void finishSlot(int slotIdx, GenerationResult.FinishReason reason) {
        if (slotIdx < 0 || slotIdx >= maxBatchSize) return;
        if (slotStates[slotIdx] == SlotState.FREE) return;

        slots[slotIdx].finishReason = reason;
        slots[slotIdx].state = SlotState.DONE;
        slotStates[slotIdx] = SlotState.DONE;

        // Free KV blocks
        if (kvCache != null) {
            kvCache.freeSequence(slotIdx);
        }

        // Return slot to free pool
        slotStates[slotIdx] = SlotState.FREE;
        freeSlots.push(slotIdx);

        log.debug("Finished slot {} (reason: {}, tokens: {}), {} free slots",
                slotIdx, reason, slots[slotIdx].tokensGenerated, freeSlots.size());
    }

    /**
     * Get all slots currently in DECODE state.
     */
    public List<SlotAssignment> getDecodeSlots() {
        List<SlotAssignment> active = new ArrayList<>();
        for (int i = 0; i < maxBatchSize; i++) {
            if (slotStates[i] == SlotState.DECODE) {
                active.add(slots[i]);
            }
        }
        return active;
    }

    /**
     * Get all slots currently in PREFILL state.
     */
    public List<SlotAssignment> getPrefillSlots() {
        List<SlotAssignment> prefill = new ArrayList<>();
        for (int i = 0; i < maxBatchSize; i++) {
            if (slotStates[i] == SlotState.PREFILL) {
                prefill.add(slots[i]);
            }
        }
        return prefill;
    }

    /**
     * Get the number of active slots (PREFILL + DECODE).
     */
    public int getActiveCount() {
        int count = 0;
        for (int i = 0; i < maxBatchSize; i++) {
            if (slotStates[i] == SlotState.PREFILL || slotStates[i] == SlotState.DECODE) {
                count++;
            }
        }
        return count;
    }

    /**
     * Check if there is any work to do (active slots or waiting requests).
     */
    public boolean hasWork() {
        return getActiveCount() > 0 || !waitingQueue.isEmpty();
    }

    /**
     * Get the number of requests waiting in the queue.
     */
    public int getWaitingCount() {
        return waitingQueue.size();
    }

    /**
     * Get the number of free slots.
     */
    public int getFreeSlotCount() {
        return freeSlots.size();
    }

    /**
     * Get the slot assignment for a specific slot index.
     * Returns null if the slot is FREE.
     */
    public SlotAssignment getSlot(int slotIdx) {
        if (slotStates[slotIdx] == SlotState.FREE) return null;
        return slots[slotIdx];
    }

    /**
     * Get the state of a slot.
     */
    public SlotState getSlotState(int slotIdx) {
        return slotStates[slotIdx];
    }

    /**
     * Collect completed results and clear DONE slots.
     *
     * @return list of completed slot assignments with generated tokens
     */
    public List<SlotAssignment> collectCompleted() {
        List<SlotAssignment> completed = new ArrayList<>();
        for (int i = 0; i < maxBatchSize; i++) {
            if (slots[i] != null && slots[i].state == SlotState.DONE) {
                completed.add(slots[i]);
                slots[i] = null;
            }
        }
        return completed;
    }
}
