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

package org.nd4j.graph;

import com.google.flatbuffers.FlatBufferBuilder;
import io.grpc.CallOptions;
import io.grpc.ManagedChannel;
import io.grpc.ManagedChannelBuilder;
import lombok.NonNull;
import lombok.extern.slf4j.Slf4j;
import lombok.val;
import org.nd4j.autodiff.execution.conf.ExecutorConfiguration;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.primitives.Pair;

import java.util.ArrayList;
import java.util.concurrent.TimeUnit;

/**
 * This class is a wrapper over GraphServer gRPC complex
 *
 * @author raver119@gmail.com
 */
@Slf4j
public class GraphInferenceClient {
    private final ManagedChannel channel;
    private final GraphInferenceServerGrpc.GraphInferenceServerBlockingStub blockingStub;

    /**
     * This method creates new GraphInferenceClient, with plain text connection
     * @param host
     * @param port
     */
    public GraphInferenceClient(@NonNull String host, int port) {
        this(host, port, false);
    }

    /**
     * This method creates new GraphInferenceClient, with optional TLS support
     * @param host
     * @param port
     */
    public GraphInferenceClient(@NonNull String host, int port, boolean useTLS) {
        this(useTLS ? ManagedChannelBuilder.forAddress(host, port).build()
                : ManagedChannelBuilder.forAddress(host, port).usePlaintext().build());
    }

    /**
     * This method creates new GraphInferenceClient over given ManagedChannel
     * @param channel
     */
    public GraphInferenceClient(@NonNull ManagedChannel channel) {
        this.channel = channel;
        this.blockingStub =  GraphInferenceServerGrpc.newBlockingStub(this.channel);
    }

    /**
     * This method shuts down gRPC connection
     *
     * @throws InterruptedException
     */
    public void shutdown() throws InterruptedException {
            this.channel.shutdown().awaitTermination(10, TimeUnit.SECONDS);
    }

    /**
     * This method adds given graph to the GraphServer storage
     * @param graph
     */
    public void registerGraph(@NonNull SameDiff graph) {
        blockingStub.registerGraph(graph.asFlatGraph());
    }

    /**
     * This method adds given graph to the GraphServer storage
     *
     * PLEASE NOTE: You don't need to register graph more then once
     * PLEASE NOTE: You don't need to register graph if GraphServer was used with -f argument
     * @param graph
     */
    public void registerGraph(@NonNull SameDiff graph, ExecutorConfiguration configuration) {
        blockingStub.registerGraph(graph.asFlatGraph(configuration));
    }

    /**
     * This method sends inference request to the GraphServer instance, and returns result as array of INDArrays
     *
     * PLEASE NOTE: This call will be routed to default graph with id 0
     * @param inputs graph inputs with their string ides
     * @return
     */
    public INDArray[] output(Pair<String, INDArray>... inputs) {
        return output(0, inputs);
    }

    /**
     * This method sends inference request to the GraphServer instance, and returns result as array of INDArrays
     * @param graphId id of the graph
     * @param inputs graph inputs with their string ides
     * @return
     */
    public INDArray[] output(long graphId, Pair<String, INDArray>... inputs) {
        val result = new ArrayList<INDArray>();
        val builder = new FlatBufferBuilder(1024);

        val ins = new int[inputs.length];

        int cnt = 0;
        for (val input: inputs) {
            val id = input.getFirst();
            val array = input.getSecond();

            val arrOff = array.toFlatArray(builder);
            val nameOff = builder.createString(id);
            val varOff = FlatVariable.createFlatVariable(builder, 0, nameOff, 0, arrOff, 0);
            ins[cnt++] = varOff;
        }

        val varsOff = FlatInferenceRequest.createVariablesVector(builder, ins);

        val off = FlatInferenceRequest.createFlatInferenceRequest(builder, graphId, varsOff, 0);
        builder.finish(off);

        val req = FlatInferenceRequest.getRootAsFlatInferenceRequest(builder.dataBuffer());

        val flatresults = blockingStub.inferenceRequest(req);

        return result.toArray(new INDArray[0]);
    }

    /**
     * This method allows to remove graph from the GraphServer instance
     * @param graphId
     */
    public void dropGraph(long graphId) {
        val builder = new FlatBufferBuilder(128);

        val off = FlatDropRequest.createFlatDropRequest(builder, graphId);
        builder.finish(off);

        val req = FlatDropRequest.getRootAsFlatDropRequest(builder.dataBuffer());

        blockingStub.forgetGraph(req);
    }
}
