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

package org.deeplearning4j.remote;

import lombok.NonNull;
import lombok.val;
import org.deeplearning4j.nn.api.Model;
import org.deeplearning4j.nn.api.ModelAdapter;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.parallelism.ParallelInference;
import org.deeplearning4j.parallelism.inference.InferenceMode;
import org.deeplearning4j.parallelism.inference.LoadBalanceMode;
import org.nd4j.adapters.InferenceAdapter;
import org.nd4j.adapters.InputAdapter;
import org.nd4j.adapters.OutputAdapter;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.common.base.Preconditions;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.MultiDataSet;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.remote.SameDiffJsonModelServer;
import org.nd4j.remote.clients.serde.BinaryDeserializer;
import org.nd4j.remote.clients.serde.BinarySerializer;
import org.nd4j.remote.clients.serde.JsonDeserializer;
import org.nd4j.remote.clients.serde.JsonSerializer;


import java.util.List;

/**
 * This class provides JSON-based model serving ability for Deeplearning4j/SameDiff models
 *
 * Server url will be http://0.0.0.0:{port}>/v1/serving
 * Server only accepts POST requests
 *
 * @param <I> type of the input class, i.e. String
 * @param <O> type of the output class, i.e. Sentiment
 *
 * @author raver119@gmail.com
 * @author astoyakin
 */
public class JsonModelServer<I, O> extends SameDiffJsonModelServer<I, O> {

    // all serving goes through ParallelInference
    protected ParallelInference parallelInference;


    protected ModelAdapter<O> modelAdapter;

    // actual models
    protected ComputationGraph cgModel;
    protected MultiLayerNetwork mlnModel;

    // service stuff
    protected InferenceMode inferenceMode;
    protected int numWorkers;

    protected boolean enabledParallel = true;

    protected JsonModelServer(@NonNull SameDiff sdModel, InferenceAdapter<I, O> inferenceAdapter,
                              JsonSerializer<O> serializer, JsonDeserializer<I> deserializer,
                              BinarySerializer<O> binarySerializer, BinaryDeserializer<I> binaryDeserializer,
                              int port, String[] orderedInputNodes, String[] orderedOutputNodes) {
        super(sdModel, inferenceAdapter, serializer, deserializer, binarySerializer, binaryDeserializer, port, orderedInputNodes, orderedOutputNodes);
    }

    protected JsonModelServer(@NonNull ComputationGraph cgModel, InferenceAdapter<I, O> inferenceAdapter,
                              JsonSerializer<O> serializer, JsonDeserializer<I> deserializer,
                              BinarySerializer<O> binarySerializer, BinaryDeserializer<I> binaryDeserializer,
                              int port, @NonNull InferenceMode inferenceMode, int numWorkers) {
        super(inferenceAdapter, serializer, deserializer, binarySerializer, binaryDeserializer, port);

        this.cgModel = cgModel;
        this.inferenceMode = inferenceMode;
        this.numWorkers = numWorkers;
    }

    protected JsonModelServer(@NonNull MultiLayerNetwork mlnModel, InferenceAdapter<I, O> inferenceAdapter,
                              JsonSerializer<O> serializer, JsonDeserializer<I> deserializer,
                              BinarySerializer<O> binarySerializer, BinaryDeserializer<I> binaryDeserializer,
                              int port, @NonNull InferenceMode inferenceMode, int numWorkers) {
        super(inferenceAdapter, serializer, deserializer, binarySerializer, binaryDeserializer, port);

        this.mlnModel = mlnModel;
        this.inferenceMode = inferenceMode;
        this.numWorkers = numWorkers;
    }

    protected JsonModelServer(@NonNull ParallelInference pi, InferenceAdapter<I, O> inferenceAdapter,
                              JsonSerializer<O> serializer, JsonDeserializer<I> deserializer,
                              BinarySerializer<O> binarySerializer, BinaryDeserializer<I> binaryDeserializer,
                              int port) {
        super(inferenceAdapter, serializer, deserializer, binarySerializer, binaryDeserializer, port);

        this.parallelInference = pi;
    }

    /**
     * This method stops server
     *
     * @throws Exception
     */
    @Override
    public void stop() throws Exception {
        if (parallelInference != null)
            parallelInference.shutdown();
        super.stop();
    }

    /**
     * This method starts server
     * @throws Exception
     */
    @Override
    public void start() throws Exception {
        // if we're just serving sdModel - we'll just call super. no dl4j functionality required in this case
        if (sdModel != null) {
            super.start();
            return;
        }
        Preconditions.checkArgument(cgModel != null || mlnModel != null, "Model serving requires either MultilayerNetwork or ComputationGraph defined");

        val model = cgModel != null ? (Model) cgModel : (Model) mlnModel;
        // PI construction is optional, since we can have it defined
        if (enabledParallel) {
            if (parallelInference == null) {
                Preconditions.checkArgument(numWorkers >= 1, "Number of workers should be >= 1, got " + numWorkers + " instead");

                parallelInference = new ParallelInference.Builder(model)
                        .inferenceMode(inferenceMode)
                        .workers(numWorkers)
                        .loadBalanceMode(LoadBalanceMode.FIFO)
                        .batchLimit(16)
                        .queueLimit(128)
                        .build();
            }
            servingServlet = new DL4jServlet.Builder<I, O>(parallelInference)
                        .parallelEnabled(true)
                        .serializer(serializer)
                        .deserializer(deserializer)
                        .binarySerializer(binarySerializer)
                        .binaryDeserializer(binaryDeserializer)
                        .inferenceAdapter(inferenceAdapter)
                        .build();
        }
        else {
            servingServlet = new DL4jServlet.Builder<I, O>(model)
                        .parallelEnabled(false)
                        .serializer(serializer)
                        .deserializer(deserializer)
                        .binarySerializer(binarySerializer)
                        .binaryDeserializer(binaryDeserializer)
                        .inferenceAdapter(inferenceAdapter)
                        .build();
        }
        start(port, servingServlet);
    }

    /**
     * Creates servlet to serve different types of models
     *
     * @param <I> type of Input class
     * @param <O> type of Output class
     *
     * @author raver119@gmail.com
     * @author astoyakin
     */
    public static class Builder<I,O> {

        private SameDiff sdModel;
        private ComputationGraph cgModel;
        private MultiLayerNetwork mlnModel;
        private ParallelInference pi;

        private String[] orderedInputNodes;
        private String[] orderedOutputNodes;

        private InferenceAdapter<I, O> inferenceAdapter;
        private JsonSerializer<O> serializer;
        private JsonDeserializer<I> deserializer;
        private BinarySerializer<O> binarySerializer;
        private BinaryDeserializer<I> binaryDeserializer;

        private InputAdapter<I> inputAdapter;
        private OutputAdapter<O> outputAdapter;

        private int port;

        private boolean parallelMode = true;

        // these fields actually require defaults
        private InferenceMode inferenceMode = InferenceMode.BATCHED;
        private int numWorkers = Nd4j.getAffinityManager().getNumberOfDevices();

        public Builder(@NonNull SameDiff sdModel) {
            this.sdModel = sdModel;
        }

        public Builder(@NonNull MultiLayerNetwork mlnModel) {
            this.mlnModel = mlnModel;
        }

        public Builder(@NonNull ComputationGraph cgModel) {
            this.cgModel = cgModel;
        }

        public Builder(@NonNull ParallelInference pi) {
            this.pi = pi;
        }

        /**
         * This method defines InferenceAdapter implementation, which will be used to convert object of Input type to the set of INDArray(s), and for conversion of resulting INDArray(s) into object of Output type
         * @param inferenceAdapter
         * @return
         */
        public Builder<I,O> inferenceAdapter(@NonNull InferenceAdapter<I,O> inferenceAdapter) {
            this.inferenceAdapter = inferenceAdapter;
            return this;
        }

        /**
         * This method allows you to specify InputAdapter to be used for inference
         *
         * PLEASE NOTE: This method is optional, and will require OutputAdapter<O> defined
         * @param inputAdapter
         * @return
         */
        public Builder<I,O> inputAdapter(@NonNull InputAdapter<I> inputAdapter) {
            this.inputAdapter = inputAdapter;
            return this;
        }

        /**
         * This method allows you to specify OutputtAdapter to be used for inference
         *
         * PLEASE NOTE: This method is optional, and will require InputAdapter<I> defined
         * @param outputAdapter
         * @return
         */
        public Builder<I,O> outputAdapter(@NonNull OutputAdapter<O> outputAdapter) {
            this.outputAdapter = outputAdapter;
            return this;
        }

        /**
         * This method allows you to specify JSON serializer.
         * Incompatible with {@link #outputBinarySerializer(BinarySerializer)}
         * Only one serializer - deserializer pair can be used by client and server.
         *
         * @param serializer
         * @return
         */
        public Builder<I,O> outputSerializer(@NonNull JsonSerializer<O> serializer) {
            this.serializer = serializer;
            return this;
        }

        /**
         * This method allows you to specify JSON deserializer.
         * Incompatible with {@link #inputBinaryDeserializer(BinaryDeserializer)}
         * Only one serializer - deserializer pair can be used by client and server.
         *
         * @param deserializer
         * @return
         */
        public Builder<I,O> inputDeserializer(@NonNull JsonDeserializer<I> deserializer) {
            this.deserializer = deserializer;
            return this;
        }

        /**
         * This method allows you to specify binary serializer.
         * Incompatible with {@link #outputSerializer(JsonSerializer)}
         * Only one serializer - deserializer pair can be used by client and server.
         *
         * @param serializer
         * @return
         */
        public Builder<I,O> outputBinarySerializer(@NonNull BinarySerializer<O> serializer) {
            this.binarySerializer = serializer;
            return this;
        }

        /**
         * This method allows you to specify binary deserializer
         * Incompatible with {@link #inputDeserializer(JsonDeserializer)}
         * Only one serializer - deserializer pair can be used by client and server.
         *
         * @param deserializer
         * @return
         */
        public Builder<I,O> inputBinaryDeserializer(@NonNull BinaryDeserializer<I> deserializer) {
            this.binaryDeserializer = deserializer;
            return this;
        }

        /**
         * This method allows you to specify inference mode for parallel mode. See {@link InferenceMode} for more details
         *
         * @param inferenceMode
         * @return
         */
        public Builder<I,O> inferenceMode(@NonNull InferenceMode inferenceMode) {
            this.inferenceMode = inferenceMode;
            return this;
        }

        /**
         * This method allows you to specify number of worker threads for ParallelInference
         *
         * @param numWorkers
         * @return
         */
        public Builder<I,O> numWorkers(int numWorkers) {
            this.numWorkers = numWorkers;
            return this;
        }

        /**
         * This method allows you to specify the order in which the inputs should be mapped to the model placeholder arrays. This is only required for {@link SameDiff} models, not {@link MultiLayerNetwork} or {@link ComputationGraph} models
         *
         * PLEASE NOTE: this argument only used for SameDiff models
         * @param args
         * @return
         */
        public Builder<I,O> orderedInputNodes(String... args) {
            orderedInputNodes = args;
            return this;
        }

        /**
         * This method allows you to specify the order in which the inputs should be mapped to the model placeholder arrays. This is only required for {@link SameDiff} models, not {@link MultiLayerNetwork} or {@link ComputationGraph} models
         *
         * PLEASE NOTE: this argument only used for SameDiff models
         * @param args
         * @return
         */
        public Builder<I,O> orderedInputNodes(@NonNull List<String> args) {
            orderedInputNodes = args.toArray(new String[args.size()]);
            return this;
        }

        /**
         * This method allows you to specify output nodes
         *
         * PLEASE NOTE: this argument only used for SameDiff models
         * @param args
         * @return
         */
        public Builder<I,O> orderedOutputNodes(String... args) {
            Preconditions.checkArgument(args != null && args.length > 0, "OutputNodes should contain at least 1 element");
            orderedOutputNodes = args;
            return this;
        }

        /**
         * This method allows you to specify output nodes
         *
         * PLEASE NOTE: this argument only used for SameDiff models
         * @param args
         * @return
         */
        public Builder<I,O> orderedOutputNodes(@NonNull List<String> args) {
            Preconditions.checkArgument(args.size() > 0, "OutputNodes should contain at least 1 element");
            orderedOutputNodes = args.toArray(new String[args.size()]);
            return this;
        }

        /**
         * This method allows you to specify http port
         *
         * PLEASE NOTE: port must be free and be in range regular TCP/IP ports range
         * @param port
         * @return
         */
        public Builder<I,O> port(int port) {
            this.port = port;
            return this;
        }

        /**
         * This method switches on ParallelInference usage
         * @param - true - to use ParallelInference, false - to use ComputationGraph or
         * MultiLayerNetwork directly
         *
         * PLEASE NOTE: this doesn't apply to SameDiff models
         *
         * @throws Exception
         */
        public Builder<I,O> parallelMode(boolean enable) {
            this.parallelMode = enable;
            return this;
        }

        public JsonModelServer<I,O> build() {
            if (inferenceAdapter == null) {
                if (inputAdapter != null && outputAdapter != null) {
                    inferenceAdapter = new InferenceAdapter<I, O>() {
                        @Override
                        public MultiDataSet apply(I input) {
                            return inputAdapter.apply(input);
                        }

                        @Override
                        public O apply(INDArray... outputs) {
                            return outputAdapter.apply(outputs);
                        }
                    };
                } else
                    throw new IllegalArgumentException("Either InferenceAdapter<I,O> or InputAdapter<I> + OutputAdapter<O> should be configured");
            }

            JsonModelServer server = null;
            if (sdModel != null) {
                server = new JsonModelServer<I, O>(sdModel, inferenceAdapter, serializer, deserializer, binarySerializer, binaryDeserializer, port, orderedInputNodes, orderedOutputNodes);
            }
            else if (cgModel != null) {
                server = new JsonModelServer<I, O>(cgModel, inferenceAdapter, serializer, deserializer, binarySerializer, binaryDeserializer, port, inferenceMode, numWorkers);
            }
            else if (mlnModel != null) {
                server = new JsonModelServer<I, O>(mlnModel, inferenceAdapter, serializer, deserializer, binarySerializer, binaryDeserializer, port, inferenceMode, numWorkers);
            }
            else if (pi != null) {
                 server = new JsonModelServer<I, O>(pi, inferenceAdapter, serializer, deserializer, binarySerializer, binaryDeserializer, port);
            }
              else
                 throw new IllegalStateException("No models were defined for JsonModelServer");

            server.enabledParallel = parallelMode;
            return server;
        }
    }

}
