/*
 *  ******************************************************************************
 *  * Copyright (c) 2021 Deeplearning4j Contributors
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
 *  * Unless required by applicable law or agreed to in writing, software
 *  * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 *  * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 *  * License for the specific language governing permissions and limitations
 *  * under the License.
 *  *
 *  * SPDX-License-Identifier: Apache-2.0
 *  *****************************************************************************
 */
package org.nd4j.autodiff.listeners.profiler;

import lombok.*;
import lombok.extern.slf4j.Slf4j;
import org.apache.commons.lang3.ArrayUtils;
import org.nd4j.autodiff.listeners.At;
import org.nd4j.autodiff.listeners.BaseListener;
import org.nd4j.autodiff.listeners.Loss;
import org.nd4j.autodiff.listeners.Operation;
import org.nd4j.autodiff.listeners.profiler.data.Phase;
import org.nd4j.autodiff.listeners.profiler.data.TraceEvent;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.autodiff.samediff.internal.SameDiffOp;
import org.nd4j.common.base.Preconditions;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.OpContext;
import org.nd4j.linalg.dataset.api.MultiDataSet;
import org.nd4j.common.primitives.AtomicBoolean;
import org.nd4j.shade.jackson.databind.DeserializationFeature;
import org.nd4j.shade.jackson.databind.MapperFeature;
import org.nd4j.shade.jackson.databind.ObjectMapper;
import org.nd4j.shade.jackson.databind.SerializationFeature;

import java.io.*;
import java.lang.management.ManagementFactory;
import java.util.*;
import java.util.concurrent.BlockingQueue;
import java.util.concurrent.LinkedBlockingDeque;

/**
 * SameDiff profiling listener: for profiling operation execution<br>
 * Writes profiles to a file in JSON format<br>
 * Format is Chrome profiler format. The output can be read by Google Chrome browser; open Chrome and go to:
 * chrome://tracing and load the output JSON format data
 * <br>
 * At present, only operation execution is profiled, not other aspects such as memory allocation and training-related
 * functionality.<br>
 * <br>
 * Tracing can be configured in a few different ways via the builder, {@link #builder(File)}:<br>
 * (a) warmup - don't record traces for the first N iterations<br>
 * (b) "all" mode (default) - record all-iterations, with no limit (after warmup, if applicable)<br>
 * (c) "n iterations" mode: record at most the first N iterations (after warmup, if applicable)<br>
 * (d) "n ms" mod: record for at most N milliseconds since the start of the first op execution (after warmup, if applicable)<br>
 *
 * Note: The Chrome Trace Event format can be found here:<br>
 * <a href="https://docs.google.com/document/d/1CvAClvFfyA5R-PhYUmn5OOQtYMH4h6I0nSsKchNAySU/edit">https://docs.google.com/document/d/1CvAClvFfyA5R-PhYUmn5OOQtYMH4h6I0nSsKchNAySU/edit</a>
 * SameDiff uses the JSON Array Format, as this can be written in an online/streaming manner.<br>
 * Conversely, TensorFlow uses the JSON Object Format.<br>
 * <br>
 * For summarizing, analyzing and comparing the results (SameDiff or TensorFlow format), see {@link org.nd4j.autodiff.listeners.profiler.comparison.ProfileAnalyzer}<br>
 *
 * @author Alex Black
 */
@Getter
@Slf4j
public class ProfilingListener extends BaseListener {

    private final File outputFile;
    private final boolean all;
    private final int warmup;
    private final int nIter;
    private final long nMs;
    private final Operation[] operations;

    private final long pid;
    private final long tid;
    private Long firstOpStart = null;       //Used for time termination
    private int countTotalIter = 0;
    private boolean logActive = false;
    private long opStartNano;

    private Writer writer;
    private ObjectMapper json;

    private final Thread fileWritingThread;
    private final BlockingQueue<TraceEvent> writeQueue;
    private final AtomicBoolean writing = new AtomicBoolean(false);

    protected ProfilingListener(@NonNull File outputFile, boolean all, int warmup, int nIter, long nMs, Operation[] operations) {
        Preconditions.checkArgument(!outputFile.exists(), "Output file already exists: %s", outputFile);
        this.outputFile = outputFile;
        this.all = all;
        this.warmup = warmup;
        this.nIter = nIter;
        this.nMs = nMs;
        this.operations = operations;

        this.pid = getProcessId();
        this.tid = Thread.currentThread().getId();

        try {
            this.writer = new BufferedWriter(new FileWriter(outputFile, false));
            this.writer.write("[");     //JSON array open (array close is optional for Chrome profiler format)
        } catch (IOException e) {
            throw new RuntimeException(e);
        }

        this.json = jsonMapper();

        //Set up a queue so file access doesn't add latency to the execution thread
        writeQueue = new LinkedBlockingDeque<>();
        fileWritingThread = new Thread(new Runnable() {
            @Override
            public void run() {
                try {
                    runHelper();
                } catch (Throwable t) {
                    log.error("Error when attempting to write results to file", t);
                }
            }

            public void runHelper() throws Exception {
                while (true) {
                    TraceEvent te = writeQueue.take();    //Blocking
                    writing.set(true);
                    try {
                        String j = json.writeValueAsString(te);
                        writer.append(j);
                        writer.append(",\n");
                    } catch (IOException e) {
                        throw new RuntimeException(e);
                    } finally {
                        writing.set(false);
                    }
                }
            }
        });
        fileWritingThread.setDaemon(true);
        fileWritingThread.start();
    }

    @Override
    public boolean isActive(Operation operation) {
        return operations == null || ArrayUtils.contains(operations, operation);
    }

    @Override
    public void operationStart(SameDiff sd, Operation op) {
        this.logActive = operations == null || ArrayUtils.contains(operations, op);
    }

    @Override
    public void operationEnd(SameDiff sd, Operation op) {
        if (this.logActive) {
            while ((!writeQueue.isEmpty() || writing.get()) && fileWritingThread.isAlive()) {
                //Wait for file writing thread to catch up
                try {
                    Thread.sleep(100);
                } catch (InterruptedException e) {
                    throw new RuntimeException(e);
                }
            }
            try {
                writer.flush();
            } catch (IOException e) {
                throw new RuntimeException(e);
            }
        }
        this.logActive = false;
        if (op == Operation.INFERENCE) {
            //Increment for inference; iteration done is called only for TRAINING
            countTotalIter++;
        }
    }

    @Override
    public void iterationDone(SameDiff sd, At at, MultiDataSet dataSet, Loss loss) {
        //Increment for training
        if (logActive) {
            countTotalIter++;
        }
    }

    @Override
    public void preOpExecution(SameDiff sd, At at, SameDiffOp op, OpContext opContext) {
        if (logActive) {
            opStartNano = System.nanoTime();

            if(!all && nMs > 0 && firstOpStart == null)
                firstOpStart = opStartNano;
        }
    }

    @Override
    public void opExecution(SameDiff sd, At at, MultiDataSet batch, SameDiffOp op, OpContext opContext, INDArray[] outputs) {
        if (logActive) {
            long now = System.nanoTime();

            if (warmup > 0 && countTotalIter < warmup) {
                return;     //Skip due to warmup phase
            }

            //Iteration termination
            int terminationPt = this.nIter > 0 ? this.nIter : Integer.MAX_VALUE;
            if (warmup > 0 && this.nIter > 0)
                terminationPt += this.warmup;

            if (countTotalIter > terminationPt) {
                logActive = false;
                return;         //Skip due to max number of itertions
            }

            //Time termination
            if(!all && nMs > 0 && (now - firstOpStart)/1000 > nMs) {
                logActive = false;
                return;
            }

            TraceEvent event = TraceEvent.builder()
                    .name(op.getOp().opName())
                    .categories(Collections.singletonList("Op"))
                    .ts(opStartNano / 1000)
                    .dur((now - opStartNano) / 1000)
                    .pid((int)pid)
                    .tid(tid)
                    .ph(Phase.X)
                    .args(Collections.<String, Object>singletonMap("name", op.getName()))
                    .build();

            writeQueue.add(event);
        }
    }


    private long getProcessId() {
        // Note: may fail in some JVM implementations
        // therefore fallback has to be provided

        // something like '<pid>@<hostname>', at least in SUN / Oracle JVMs
        final String jvmName = ManagementFactory.getRuntimeMXBean().getName();
        final int index = jvmName.indexOf('@');

        if (index < 1) {
            // part before '@' empty (index = 0) / '@' not found (index = -1)
            return 0;
        }

        try {
            return Long.parseLong(jvmName.substring(0, index));
        } catch (NumberFormatException e) {
            // ignore
        }
        return 0;
    }

    /**
     * Get a new JSON mapper for use in serializing/deserializing JSON format
     */
    public static ObjectMapper jsonMapper() {
        ObjectMapper json = new ObjectMapper();
        json.configure(DeserializationFeature.FAIL_ON_UNKNOWN_PROPERTIES, false);
        json.configure(SerializationFeature.FAIL_ON_EMPTY_BEANS, false);
        json.configure(MapperFeature.SORT_PROPERTIES_ALPHABETICALLY, false);
        json.disable(SerializationFeature.INDENT_OUTPUT);   //One line

        return json;
    }

    /**
     * Create a new builder
     * @param outputFile Output file. Will be overwritten if file already exists
     */
    public static Builder builder(File outputFile) {
        return new Builder(outputFile);
    }

    public static class Builder {
        private final File outputFile;
        private boolean all = true;
        private int warmup = 0;
        private int nIter = -1;
        private long nMs = -1;
        private Operation[] operations;

        public Builder(@NonNull File outputFile) {
            this.outputFile = outputFile;
        }

        /**
         * If called, all data will be profiled with no limits (other than a warmup, if set)
         */
        public Builder recordAll() {
            this.all = true;
            this.nIter = -1;
            this.nMs = -1;
            return this;
        }

        /**
         * Specify the number of warmup iterations - i.e., these will be excluded from profiling results
         */
        public Builder warmup(int iterations) {
            this.warmup = iterations;
            return this;
        }

        /**
         * Set a limit on the maximum number of iterations to profile (after warmup, if any).
         * Any ops executed after the specified number of iterations will not be profiled/recorded
         */
        public Builder maxProfileIterations(int iterations) {
            this.nIter = iterations;
            this.all = false;
            return this;
        }

        /**
         * Set a limit on the maximum duration for profiling, in milliseconds.
         * Any ops executed after the specified amount of time since the first (non-warmup) operation start will not be
         * profiled/recorded
         */
        public Builder maxProfilerMilliseconds(long ms) {
            this.nMs = ms;
            this.all = false;
            return this;
        }

        /**
         * Specify the operations (training, inference, etc) to profile.
         * If not set, all operations are profiled
         */
        public Builder operations(Operation... operations) {
            this.operations = operations;
            return this;
        }

        /**
         * Create the profiling listener
         */
        public ProfilingListener build() {
            return new ProfilingListener(outputFile, all, warmup, nIter, nMs, operations);
        }
    }
}
