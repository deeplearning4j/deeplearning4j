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

package org.nd4j.tensorflow.conversion.graphrunner;

import org.nd4j.TFGraphRunnerService;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.tensorflow.conversion.TensorDataType;

import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class GraphRunnerServiceProvider implements TFGraphRunnerService {

    private GraphRunner graphRunner;
    Map<String, INDArray> inputs;

    @Override
    public TFGraphRunnerService init(
            List<String> inputNames,
            List<String> outputNames,
            byte[] graphBytes,
            Map<String, INDArray> constants,
            Map<String, String> inputDataTypes){
        if (inputNames.size() != inputDataTypes.size()){
            throw new IllegalArgumentException("inputNames.size() != inputDataTypes.size()");
        }
        Map<String, TensorDataType> convertedDataTypes = new HashMap<>();
        for (int i = 0; i < inputNames.size(); i++){
            convertedDataTypes.put(inputNames.get(i), TensorDataType.fromProtoValue(inputDataTypes.get(inputNames.get(i))));
        }
        Map<String, INDArray> castConstants = new HashMap<>();
        for (Map.Entry<String, INDArray> e: constants.entrySet()) {
            DataType requiredDtype = TensorDataType.toNd4jType(TensorDataType.fromProtoValue(inputDataTypes.get(e.getKey())));
            castConstants.put(e.getKey(), e.getValue().castTo(requiredDtype));
        }
        this.inputs = castConstants;
        graphRunner = GraphRunner.builder().inputNames(inputNames)
                .outputNames(outputNames).graphBytes(graphBytes)
                .inputDataTypes(convertedDataTypes).build();
        return this;

    }

    @Override
    public Map<String, INDArray> run(Map<String, INDArray> inputs){
        if (graphRunner == null){
            throw new RuntimeException("GraphRunner not initialized.");
        }
        this.inputs.putAll(inputs);
        return graphRunner.run(this.inputs);
    }
}
