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

package org.deeplearning4j.remote;

import lombok.*;
import lombok.extern.slf4j.Slf4j;
import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.api.Model;
import org.deeplearning4j.nn.api.NeuralNetwork;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.parallelism.ParallelInference;
import org.nd4j.adapters.InferenceAdapter;
import org.nd4j.base.Preconditions;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.remote.clients.serde.JsonDeserializer;
import org.nd4j.remote.clients.serde.JsonSerializer;
import org.nd4j.adapters.InferenceAdapter;
import org.nd4j.remote.serving.SameDiffServlet;

import javax.servlet.http.HttpServletRequest;
import javax.servlet.http.HttpServletResponse;
import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;

/**
 *
 * @author astoyakin
 */
@Slf4j
@NoArgsConstructor
public class DL4jServlet<I,O> extends SameDiffServlet<I,O> {

    protected ParallelInference parallelInference;
    protected Model model;
    protected boolean parallelEnabled = true;

    public DL4jServlet(@NonNull ParallelInference parallelInference, @NonNull InferenceAdapter<I, O> inferenceAdapter,
                       @NonNull JsonSerializer<O> serializer, @NonNull JsonDeserializer<I> deserializer) {
        super(inferenceAdapter, serializer, deserializer);
        this.parallelInference = parallelInference;
        this.model = null;
        this.parallelEnabled = true;
    }

    public DL4jServlet(@NonNull Model model, @NonNull InferenceAdapter<I, O> inferenceAdapter,
                       @NonNull JsonSerializer<O> serializer, @NonNull JsonDeserializer<I> deserializer) {
        super(inferenceAdapter, serializer, deserializer);
        this.model = model;
        this.parallelInference = null;
        this.parallelEnabled = false;
    }

    @Override
    protected void doPost(HttpServletRequest request, HttpServletResponse response) throws IOException {
        String processorReturned = "";
        String path = request.getPathInfo();
        if (path.equals(SERVING_ENDPOINT)) {
            val contentType = request.getContentType();
            if (validateRequest(request,response)) {
                val stream = request.getInputStream();
                val bufferedReader = new BufferedReader(new InputStreamReader(stream));
                char[] charBuffer = new char[128];
                int bytesRead = -1;
                val buffer = new StringBuilder();
                while ((bytesRead = bufferedReader.read(charBuffer)) > 0) {
                    buffer.append(charBuffer, 0, bytesRead);
                }
                val requestString = buffer.toString();

                val mds = inferenceAdapter.apply(deserializer.deserialize(requestString));

                O result = null;
                if (parallelEnabled) {
                    // process result
                    result = inferenceAdapter.apply(parallelInference.output(mds.getFeatures(), mds.getFeaturesMaskArrays()));
                }
                else {
                    synchronized(this) {
                        if (model instanceof ComputationGraph)
                            result = inferenceAdapter.apply(((ComputationGraph)model).output(false, mds.getFeatures(), mds.getFeaturesMaskArrays()));
                        else if (model instanceof MultiLayerNetwork) {
                            Preconditions.checkArgument(mds.getFeatures().length > 1 || (mds.getFeaturesMaskArrays() != null && mds.getFeaturesMaskArrays().length > 1),
                                    "Input data for MultilayerNetwork is invalid!");
                            result = inferenceAdapter.apply(((MultiLayerNetwork) model).output(mds.getFeatures()[0], false,
                                    mds.getFeaturesMaskArrays() != null ? mds.getFeaturesMaskArrays()[0] : null, null));
                        }
                    }
                }
                processorReturned = serializer.serialize(result);
            }
        } else {
            // we return error otherwise
            sendError(request.getRequestURI(), response);
        }
        try {
            val out = response.getWriter();
            out.write(processorReturned);
        } catch (IOException e) {
            log.error(e.getMessage());
        }
    }

    /**
     * Creates servlet to serve models
     *
     * @param <I> type of Input class
     * @param <O> type of Output class
     *
     * @author raver119@gmail.com
     * @author astoyakin
     */
    public static class Builder<I,O> {

        private ParallelInference pi;
        private Model model;

        private InferenceAdapter<I, O> inferenceAdapter;
        private JsonSerializer<O> serializer;
        private JsonDeserializer<I> deserializer;
        private int port;
        private boolean parallelEnabled = true;

        public Builder(@NonNull ParallelInference pi) {
            this.pi = pi;
        }

        public Builder(@NonNull Model model) {
            this.model = model;
        }

        public Builder<I,O> inferenceAdapter(@NonNull InferenceAdapter<I,O> inferenceAdapter) {
            this.inferenceAdapter = inferenceAdapter;
            return this;
        }

        /**
         * This method is required to specify serializer
         *
         * @param serializer
         * @return
         */
        public Builder<I,O> serializer(@NonNull JsonSerializer<O> serializer) {
            this.serializer = serializer;
            return this;
        }

        /**
         * This method allows to specify deserializer
         *
         * @param deserializer
         * @return
         */
        public Builder<I,O> deserializer(@NonNull JsonDeserializer<I> deserializer) {
            this.deserializer = deserializer;
            return this;
        }

        /**
         * This method allows to specify port
         *
         * @param port
         * @return
         */
        public Builder<I,O> port(int port) {
            this.port = port;
            return this;
        }

        /**
         * This method activates parallel inference
         *
         * @param parallelEnabled
         * @return
         */
        public Builder<I,O> parallelEnabled(boolean parallelEnabled) {
            this.parallelEnabled = parallelEnabled;
            return this;
        }

        public DL4jServlet<I,O> build() {
            return parallelEnabled ? new DL4jServlet<I, O>(pi, inferenceAdapter, serializer, deserializer) :
                    new DL4jServlet<I, O>(model, inferenceAdapter, serializer, deserializer);
        }
    }
}




