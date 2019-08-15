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

package org.nd4j.remote.clients;

import com.mashape.unirest.http.HttpResponse;
import com.mashape.unirest.http.Unirest;
import com.mashape.unirest.http.exceptions.UnirestException;
import lombok.Builder;
import lombok.NonNull;
import lombok.extern.slf4j.Slf4j;
import lombok.val;
import org.json.JSONObject;
import org.nd4j.remote.clients.serde.JsonDeserializer;
import org.nd4j.remote.clients.serde.JsonSerializer;

import java.io.IOException;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.Future;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.TimeoutException;

/**
 * This class provides remote inference functionality via JSON-powered REST APIs.
 *
 * Basically we assume that there's remote JSON server available (on bare metal or in k8s/swarm/whatever cluster), and with proper serializers/deserializers provided we can issue REST requests and get back responses.
 * So, in this way application logic can be separated from DL logic.
 *
 * You just need to provide serializer/deserializer and address of the REST server, i.e. "http://model:8080/v1/serving"
 *
 * @param <I> type of the input class, i.e. String
 * @param <O> type of the output class, i.e. Sentiment
 *
 * @author raver119@gmail.com
 */
@Slf4j
public class JsonRemoteInference<I, O> {
    private String endpointAddress;
    private JsonSerializer<I> serializer;
    private JsonDeserializer<O> deserializer;

    @Builder
    public JsonRemoteInference(@NonNull String endpointAddress, @NonNull JsonSerializer<I> inputSerializer, @NonNull JsonDeserializer<O> outputDeserializer) {
        this.endpointAddress = endpointAddress;
        this.serializer = inputSerializer;
        this.deserializer = outputDeserializer;
    }

    private O processResponse(HttpResponse<String> response) throws IOException {
        if (response.getStatus() != 200)
            throw new IOException("Inference request returned bad error code: " + response.getStatus());

        O result = deserializer.deserialize(response.getBody());
        if (result == null) {
            throw new IOException("Deserialization failed!");
        }
        return result;
    }

    /**
     * This method does remote inference in a blocking way
     *
     * @param input
     * @return
     * @throws IOException
     */
    public O predict(I input) throws IOException {
        try {
            val stringResult = Unirest.post(endpointAddress)
                    .header("Content-Type", "application/json")
                    .header("Accept", "application/json")
                    .body(new JSONObject(serializer.serialize(input))).asString();

            return processResponse(stringResult);
        } catch (UnirestException e) {
            throw new IOException(e);
        }
    }

    /**
     * This method does remote inference in asynchronous way, returning Future instead
     * @param input
     * @return
     */
    public Future<O> predictAsync(I input) {
        val stringResult = Unirest.post(endpointAddress)
                .header("Content-Type", "application/json")
                .header("Accept", "application/json")
                .body(new JSONObject(serializer.serialize(input))).asStringAsync();
        return new InferenceFuture(stringResult);
    }

    /**
     * This class holds a Future of the object returned by remote inference server
     */
    private class InferenceFuture implements Future<O> {
        private Future<HttpResponse<String>> unirestFuture;

        private InferenceFuture(@NonNull Future<HttpResponse<String>> future) {
            this.unirestFuture = future;
        }

        @Override
        public boolean cancel(boolean mayInterruptIfRunning) {
            return unirestFuture.cancel(mayInterruptIfRunning);
        }

        @Override
        public boolean isCancelled() {
            return unirestFuture.isCancelled();
        }

        @Override
        public boolean isDone() {
            return unirestFuture.isDone();
        }

        @Override
        public O get() throws InterruptedException, ExecutionException {
            val stringResult = unirestFuture.get();

            try {
                return processResponse(stringResult);
            } catch (IOException e) {
                throw new ExecutionException(e);
            }
        }

        @Override
        public O get(long timeout, TimeUnit unit) throws InterruptedException, ExecutionException, TimeoutException {
            val stringResult = unirestFuture.get(timeout, unit);

            try {
                return processResponse(stringResult);
            } catch (IOException e) {
                throw new ExecutionException(e);
            }
        }
    }
}
