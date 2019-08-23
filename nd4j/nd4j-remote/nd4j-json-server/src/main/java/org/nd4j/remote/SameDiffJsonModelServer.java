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

package org.nd4j.remote;

import lombok.*;
import lombok.extern.slf4j.Slf4j;
import org.eclipse.jetty.server.Server;
import org.eclipse.jetty.servlet.ServletContextHandler;
import org.nd4j.adapters.InputAdapter;
import org.nd4j.adapters.OutputAdapter;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.base.Preconditions;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.MultiDataSet;
import org.nd4j.remote.clients.serde.BinaryDeserializer;
import org.nd4j.remote.clients.serde.BinarySerializer;
import org.nd4j.remote.clients.serde.JsonDeserializer;
import org.nd4j.remote.clients.serde.JsonSerializer;
import org.nd4j.adapters.InferenceAdapter;
import org.nd4j.remote.serving.ModelServingServlet;
import org.nd4j.remote.serving.SameDiffServlet;

import java.util.List;

/**
 * This class provides JSON-powered model serving functionality for SameDiff graphs.
 * Server url will be http://0.0.0.0:{port}>/v1/serving
 * Server only accepts POST requests
 *
 * @param <I> type of the input class, i.e. String
 * @param <O> type of the output class, i.e. Sentiment
 *
 * @author raver119@gmail.com
 */
@Slf4j
public class SameDiffJsonModelServer<I, O> {


    protected SameDiff sdModel;
    protected final JsonSerializer<O> serializer;
    protected final JsonDeserializer<I> deserializer;
    protected final BinarySerializer<O> binarySerializer;
    protected final BinaryDeserializer<I> binaryDeserializer;
    protected final InferenceAdapter<I, O> inferenceAdapter;
    protected final int port;

    // this servlet will be used to serve models
    protected ModelServingServlet<I, O> servingServlet;

    // HTTP server instance
    protected Server server;

    // for SameDiff only
    protected String[] orderedInputNodes;
    protected String[] orderedOutputNodes;

    protected SameDiffJsonModelServer(@NonNull InferenceAdapter<I, O> inferenceAdapter,
                                      JsonSerializer<O> serializer, JsonDeserializer<I> deserializer,
                                      BinarySerializer<O> binarySerializer, BinaryDeserializer<I> binaryDeserializer,
                                      int port) {
        Preconditions.checkArgument(port > 0 && port < 65535, "TCP port must be in range of 0..65535");
        Preconditions.checkArgument(serializer == null && binarySerializer == null ||
                                        serializer != null && binarySerializer == null ||
                                        serializer == null && binarySerializer != null,
                                "JSON and binary serializers/deserializers are mutually exclusive and mandatory.");

        this.binarySerializer = binarySerializer;
        this.binaryDeserializer = binaryDeserializer;
        this.inferenceAdapter = inferenceAdapter;
        this.serializer = serializer;
        this.deserializer = deserializer;
        this.port = port;
    }

    //@Builder
    public SameDiffJsonModelServer(SameDiff sdModel, @NonNull InferenceAdapter<I, O> inferenceAdapter,
                                   JsonSerializer<O> serializer, JsonDeserializer<I> deserializer,
                                   BinarySerializer<O> binarySerializer, BinaryDeserializer<I> binaryDeserializer,
                                   int port, String[] orderedInputNodes, @NonNull String[] orderedOutputNodes) {
        this(inferenceAdapter, serializer, deserializer, binarySerializer, binaryDeserializer, port);
        this.sdModel = sdModel;
        this.orderedInputNodes = orderedInputNodes;
        this.orderedOutputNodes = orderedOutputNodes;

        // TODO: both lists of nodes should be validated, to make sure nodes specified here exist in actual model
        if (orderedInputNodes != null) {
            // input nodes list might be null. strange but ok
        }

        Preconditions.checkArgument(orderedOutputNodes != null && orderedOutputNodes.length > 0, "SameDiff serving requires at least 1 output node");
    }

    protected void start(int port, @NonNull ModelServingServlet<I, O> servlet) throws Exception {
        val context = new ServletContextHandler(ServletContextHandler.SESSIONS);
        context.setContextPath("/");

        server = new Server(port);
        server.setHandler(context);

        val jerseyServlet = context.addServlet(org.glassfish.jersey.servlet.ServletContainer.class, "/*");
        jerseyServlet.setInitOrder(0);
        jerseyServlet.setServlet(servlet);

        server.start();
    }

    public void start() throws Exception {
        Preconditions.checkArgument(sdModel != null, "SameDiff model wasn't defined");

        servingServlet = SameDiffServlet.<I, O>builder()
                .sdModel(sdModel)
                .serializer(serializer)
                .deserializer(deserializer)
                .inferenceAdapter(inferenceAdapter)
                .orderedInputNodes(orderedInputNodes)
                .orderedOutputNodes(orderedOutputNodes)
                .build();

        start(port, servingServlet);
    }

    public void join() throws InterruptedException {
        Preconditions.checkArgument(server != null, "Model server wasn't started yet");

        server.join();
    }

    public void stop() throws Exception {
        //Preconditions.checkArgument(server != null, "Model server wasn't started yet");

        server.stop();
    }


    public static class Builder<I,O> {
        private SameDiff sameDiff;
        private String[] orderedInputNodes;
        private String[] orderedOutputNodes;
        private InferenceAdapter<I, O> inferenceAdapter;
        private JsonSerializer<O> serializer;
        private JsonDeserializer<I> deserializer;
        private int port;

        private InputAdapter<I> inputAdapter;
        private OutputAdapter<O> outputAdapter;

        public Builder() {}

        public Builder<I,O> sdModel(@NonNull SameDiff sameDiff) {
            this.sameDiff = sameDiff;
            return this;
        }

        /**
         * This method defines InferenceAdapter implementation, which will be used to convert object of Input type to the set of INDArray(s), and for conversion of resulting INDArray(s) into object of Output type
         * @param inferenceAdapter
         * @return
         */
        public Builder<I,O> inferenceAdapter(InferenceAdapter<I,O> inferenceAdapter) {
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
         * This method allows you to specify OutputAdapter to be used for inference
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
         * This method defines JsonSerializer instance to be used to convert object of output type into JSON format, so it could be sent over the wire
         *
         * @param serializer
         * @return
         */
        public Builder<I,O> outputSerializer(@NonNull JsonSerializer<O> serializer) {
            this.serializer = serializer;
            return this;
        }

        /**
         * This method defines JsonDeserializer instance to be used to convert JSON passed through HTTP into actual object of input type, that will be fed into SameDiff model
         *
         * @param deserializer
         * @return
         */
        public Builder<I,O> inputDeserializer(@NonNull JsonDeserializer<I> deserializer) {
            this.deserializer = deserializer;
            return this;
        }

        /**
         * This method defines the order of placeholders to be filled with INDArrays provided by Deserializer
         *
         * @param args
         * @return
         */
        public Builder<I,O> orderedInputNodes(String... args) {
            orderedInputNodes = args;
            return this;
        }

        /**
         * This method defines the order of placeholders to be filled with INDArrays provided by Deserializer
         *
         * @param args
         * @return
         */
        public Builder<I,O> orderedInputNodes(@NonNull List<String> args) {
            orderedInputNodes = args.toArray(new String[args.size()]);
            return this;
        }

        /**
         * This method defines list of graph nodes to be extracted after feed-forward pass and used as OutputAdapter input
         * @param args
         * @return
         */
        public Builder<I,O> orderedOutputNodes(String... args) {
            Preconditions.checkArgument(args != null && args.length > 0, "OutputNodes should contain at least 1 element");
            orderedOutputNodes = args;
            return this;
        }

        /**
         * This method defines list of graph nodes to be extracted after feed-forward pass and used as OutputAdapter input
         * @param args
         * @return
         */
        public Builder<I,O> orderedOutputNodes(@NonNull List<String> args) {
            Preconditions.checkArgument(args.size() > 0, "OutputNodes should contain at least 1 element");
            orderedOutputNodes = args.toArray(new String[args.size()]);
            return this;
        }

        /**
         * This method allows to configure HTTP port used for serving
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
         * This method builds SameDiffJsonModelServer instance
         * @return
         */
        public SameDiffJsonModelServer<I, O> build() {
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
            return new SameDiffJsonModelServer<I,O>(sameDiff, inferenceAdapter, serializer, deserializer, null, null, port, orderedInputNodes, orderedOutputNodes);
        }
    }
}
