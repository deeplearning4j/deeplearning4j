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

package org.datavec.spark.inference.server;

import com.beust.jcommander.Parameter;
import com.fasterxml.jackson.databind.JsonNode;
import org.datavec.spark.inference.model.model.Base64NDArrayBody;
import org.datavec.spark.inference.model.model.BatchCSVRecord;
import org.datavec.spark.inference.model.service.DataVecTransformService;
import org.nd4j.shade.jackson.databind.ObjectMapper;
import play.mvc.Http;
import play.server.Server;

/**
 * Created by kepricon on 17. 6. 20.
 */
public abstract class SparkTransformServer implements DataVecTransformService {
    @Parameter(names = {"-j", "--jsonPath"}, arity = 1)
    protected String jsonPath = null;
    @Parameter(names = {"-dp", "--dataVecPort"}, arity = 1)
    protected int port = 9000;
    @Parameter(names = {"-dt", "--dataType"}, arity = 1)
    private TransformDataType transformDataType = null;
    protected Server server;
    protected static ObjectMapper objectMapper = new ObjectMapper();
    protected static String contentType = "application/json";

    public abstract void runMain(String[] args) throws Exception;

    /**
     * Stop the server
     */
    public void stop() {
        if (server != null)
            server.stop();
    }

    protected boolean isSequence(Http.Request request) {
        return request.hasHeader(SEQUENCE_OR_NOT_HEADER)
                && request.header(SEQUENCE_OR_NOT_HEADER).get().equalsIgnoreCase("true");
    }

    protected String getJsonText(Http.Request request) {
        JsonNode tryJson = request.body().asJson();
        if (tryJson != null)
            return tryJson.toString();
        else
            return request.body().asText();
    }

    public abstract Base64NDArrayBody transformSequenceArrayIncremental(BatchCSVRecord singleCsvRecord);
}
