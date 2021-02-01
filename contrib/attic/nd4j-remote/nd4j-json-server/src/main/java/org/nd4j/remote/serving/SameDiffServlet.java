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

package org.nd4j.remote.serving;

import lombok.*;
import lombok.extern.slf4j.Slf4j;
import org.apache.commons.lang3.StringUtils;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.remote.clients.serde.BinaryDeserializer;
import org.nd4j.remote.clients.serde.BinarySerializer;
import org.nd4j.remote.clients.serde.JsonDeserializer;
import org.nd4j.remote.clients.serde.JsonSerializer;
import org.nd4j.adapters.InferenceAdapter;

import javax.servlet.*;
import javax.servlet.http.HttpServletRequest;
import javax.servlet.http.HttpServletResponse;
import javax.ws.rs.HttpMethod;
import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.LinkedHashMap;

import static javax.ws.rs.core.MediaType.APPLICATION_JSON;
import static javax.ws.rs.core.MediaType.APPLICATION_OCTET_STREAM;

/**
 * This servlet provides SameDiff model serving capabilities
 *
 * @param <I>
 * @param <O>
 *
 * @author raver119@gmail.com
 */
@NoArgsConstructor
@AllArgsConstructor
@Slf4j
@Builder
public class SameDiffServlet<I, O> implements ModelServingServlet<I, O> {

    protected static final String typeJson = APPLICATION_JSON;
    protected static final String typeBinary = APPLICATION_OCTET_STREAM;

    protected SameDiff sdModel;
    protected JsonSerializer<O> serializer;
    protected JsonDeserializer<I> deserializer;
    protected BinarySerializer<O> binarySerializer;
    protected BinaryDeserializer<I> binaryDeserializer;
    protected InferenceAdapter<I, O> inferenceAdapter;

    protected String[] orderedInputNodes;
    protected String[] orderedOutputNodes;

    protected final static String SERVING_ENDPOINT = "/v1/serving";
    protected final static String LISTING_ENDPOINT = "/v1";
    protected final static int PAYLOAD_SIZE_LIMIT = 10 * 1024 * 1024; // TODO: should be customizable

    protected SameDiffServlet(@NonNull InferenceAdapter<I, O> inferenceAdapter, JsonSerializer<O> serializer, JsonDeserializer<I> deserializer){
        this.serializer = serializer;
        this.deserializer = deserializer;
        this.inferenceAdapter = inferenceAdapter;
    }

    protected SameDiffServlet(@NonNull InferenceAdapter<I, O> inferenceAdapter,
                              BinarySerializer<O> serializer, BinaryDeserializer<I> deserializer){
        this.binarySerializer = serializer;
        this.binaryDeserializer = deserializer;
        this.inferenceAdapter = inferenceAdapter;
    }

    protected SameDiffServlet(@NonNull InferenceAdapter<I, O> inferenceAdapter,
                              JsonSerializer<O> jsonSerializer, JsonDeserializer<I> jsonDeserializer,
                              BinarySerializer<O> binarySerializer, BinaryDeserializer<I> binaryDeserializer){

        this.serializer = jsonSerializer;
        this.deserializer = jsonDeserializer;
        this.binarySerializer = binarySerializer;
        this.binaryDeserializer = binaryDeserializer;
        this.inferenceAdapter = inferenceAdapter;

        if (serializer != null && binarySerializer != null || serializer == null && binarySerializer == null)
            throw new IllegalStateException("Binary and JSON serializers/deserializers are mutually exclusive and mandatory.");
    }


    @Override
    public void init(ServletConfig servletConfig) throws ServletException {
        //
    }

    @Override
    public ServletConfig getServletConfig() {
        return null;
    }

    @Override
    public void service(ServletRequest servletRequest, ServletResponse servletResponse) throws ServletException, IOException {
        // we'll parse request here, and do model serving
        val httpRequest = (HttpServletRequest) servletRequest;
        val httpResponse = (HttpServletResponse) servletResponse;

        if (httpRequest.getMethod().equals(HttpMethod.GET)) {
            doGet(httpRequest, httpResponse);
        }
        else if (httpRequest.getMethod().equals(HttpMethod.POST)) {
            doPost(httpRequest, httpResponse);
        }

    }

    protected void sendError(String uri, HttpServletResponse response) throws IOException {
        val msg = "Requested endpoint [" + uri + "] not found";
        response.setStatus(404, msg);
        response.sendError(404, msg);
    }

    protected void sendBadContentType(String actualContentType, HttpServletResponse response) throws IOException {
        val msg = "Content type [" + actualContentType + "] not supported";
        response.setStatus(415, msg);
        response.sendError(415, msg);
    }

    protected boolean validateRequest(HttpServletRequest request, HttpServletResponse response)
                                                                            throws IOException{
        val contentType = request.getContentType();
        if (!StringUtils.equals(contentType, typeJson)) {
            sendBadContentType(contentType, response);
            int contentLength = request.getContentLength();
            if (contentLength > PAYLOAD_SIZE_LIMIT) {
                response.sendError(500, "Payload size limit violated!");
            }
            return false;
        }
        return true;
    }

    protected void doGet(HttpServletRequest request, HttpServletResponse response) throws IOException {
        val processor = new ServingProcessor();
        String processorReturned = "";
        String path = request.getPathInfo();
        if (path.equals(LISTING_ENDPOINT)) {
            val contentType = request.getContentType();
            if (!StringUtils.equals(contentType, typeJson)) {
                sendBadContentType(contentType, response);
            }
            processorReturned = processor.listEndpoints();
        }
        else {
            sendError(request.getRequestURI(), response);
        }
        try {
            val out = response.getWriter();
            out.write(processorReturned);
        } catch (IOException e) {
            log.error(e.getMessage());
        }
    }

    protected void doPost(HttpServletRequest request, HttpServletResponse response) throws IOException {
        val processor = new ServingProcessor();
        String processorReturned = "";
        String path = request.getPathInfo();
        if (path.equals(SERVING_ENDPOINT)) {
            val contentType = request.getContentType();
            /*Preconditions.checkArgument(StringUtils.equals(contentType, typeJson),
                    "Content type is " + contentType);*/
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
                val map = new LinkedHashMap<String, INDArray>();

                // optionally define placeholders with names provided in server constructor
                if (orderedInputNodes != null && orderedInputNodes.length > 0) {
                    int cnt = 0;
                    for (val n : orderedInputNodes)
                        map.put(n, mds.getFeatures(cnt++));
                }

                val output = sdModel.output(map, orderedOutputNodes);
                val arrays = new INDArray[output.size()];

                // now we need to get ordered output arrays, as specified in server constructor
                int cnt = 0;
                for (val n : orderedOutputNodes)
                    arrays[cnt++] = output.get(n);

                // process result
                val result = inferenceAdapter.apply(arrays);
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

    @Override
    public String getServletInfo() {
        return null;
    }

    @Override
    public void destroy() {
        //
    }
}
