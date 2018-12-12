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
import io.grpc.ManagedChannel;
import io.grpc.ManagedChannelBuilder;
import lombok.NonNull;
import lombok.extern.slf4j.Slf4j;
import lombok.val;
import org.nd4j.autodiff.execution.conf.ExecutorConfiguration;
import org.nd4j.autodiff.execution.input.OperandsAdapter;
import org.nd4j.autodiff.execution.input.Operands;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.autodiff.samediff.serde.FlatBuffersMapper;
import org.nd4j.graph.grpc.GraphInferenceServerGrpc;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.exception.ND4JIllegalStateException;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.primitives.Pair;

import java.util.ArrayList;
import java.util.concurrent.TimeUnit;

/**
 * This class is a wrapper over GraphServer gRPC complex
 *
 * @author raver119@gmail.com
 */
@Slf4j
public class GraphInferenceGrpcClient {
    private final ManagedChannel channel;
    private final GraphInferenceServerGrpc.GraphInferenceServerBlockingStub blockingStub;

    /**
     * This method creates new GraphInferenceGrpcClient, with plain text connection
     * @param host
     * @param port
     */
    public GraphInferenceGrpcClient(@NonNull String host, int port) {
        this(host, port, false);
    }

    /**
     * This method creates new GraphInferenceGrpcClient, with optional TLS support
     * @param host
     * @param port
     */
    public GraphInferenceGrpcClient(@NonNull String host, int port, boolean useTLS) {
        this(useTLS ? ManagedChannelBuilder.forAddress(host, port).build()
                : ManagedChannelBuilder.forAddress(host, port).usePlaintext().build());
    }

    /**
     * This method creates new GraphInferenceGrpcClient over given ManagedChannel
     * @param channel
     */
    public GraphInferenceGrpcClient(@NonNull ManagedChannel channel) {
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
     * @param graphId id of the graph, if not 0 - should be used in subsequent output() requests
     * @param graph
     *
     */
    public void registerGraph(long graphId, @NonNull SameDiff graph, ExecutorConfiguration configuration) {
        val g = graph.asFlatGraph(graphId, configuration);
        val v = blockingStub.registerGraph(g);
        if (v.status() != 0)
            throw new ND4JIllegalStateException("registerGraph() gRPC call failed");
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
     * This method is suited for use of custom OperandsAdapters
     * @param adapter
     * @param <T>
     * @return
     */
    public <T> T output(long graphId, T value, OperandsAdapter<T> adapter) {
        return adapter.output(this.output(graphId, adapter.input(value)));
    }


    public Operands output(long graphId, @NonNull Operands operands) {
        val result = new ArrayList<INDArray>();
        val builder = new FlatBufferBuilder(1024);

        val ins = new int[operands.size()];

        val col = operands.asCollection();

        int cnt = 0;
        for (val input: col) {
            val id = input.getFirst();
            val array = input.getSecond();

            val idPair = IntPair.createIntPair(builder, id.getId(), id.getIndex());
            val nameOff = id.getName() != null ? builder.createString(id.getName()) : 0;

            val arrOff = array.toFlatArray(builder);
            byte variableType = 0;  //TODO is this OK here?
            val varOff = FlatVariable.createFlatVariable(builder, idPair, nameOff, FlatBuffersMapper.getDataTypeAsByte(array.dataType()),0,  arrOff, -1, variableType);
            ins[cnt++] = varOff;
        }

        val varsOff = FlatInferenceRequest.createVariablesVector(builder, ins);

        val off = FlatInferenceRequest.createFlatInferenceRequest(builder, graphId, varsOff, 0);
        builder.finish(off);

        val req = FlatInferenceRequest.getRootAsFlatInferenceRequest(builder.dataBuffer());

        val fr = blockingStub.inferenceRequest(req);

        val res = new Operands();

            for (int e = 0; e < fr.variablesLength(); e++) {
                val v = fr.variables(e);

                val array = Nd4j.createFromFlatArray(v.ndarray());
                res.addArgument(v.name(), array);
                res.addArgument(v.id().first(), v.id().second(), array);
                res.addArgument(v.name(), v.id().first(), v.id().second(), array);
            }

        return res;
    }

    /**
     * This method sends inference request to the GraphServer instance, and returns result as array of INDArrays
     * @param graphId id of the graph
     * @param inputs graph inputs with their string ides
     * @return
     */
    public INDArray[] output(long graphId, Pair<String, INDArray>... inputs) {
        val operands = new Operands();
        for (val in:inputs)
            operands.addArgument(in.getFirst(), in.getSecond());

        return output(graphId, operands).asArray();
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

        val v = blockingStub.forgetGraph(req);
        if (v.status() != 0)
            throw new ND4JIllegalStateException("registerGraph() gRPC call failed");
    }
}
