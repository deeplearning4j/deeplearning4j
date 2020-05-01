/*******************************************************************************
 * Copyright (c) 2020 Konduit K.K.
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

package org.deeplearning4j.nn.modelimport.keras.layers;

import lombok.Data;
import lombok.extern.slf4j.Slf4j;
import org.apache.commons.lang3.ArrayUtils;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.gradient.Gradient;
import org.deeplearning4j.nn.layers.AbstractLayer;
import org.deeplearning4j.nn.workspace.LayerWorkspaceMgr;
import org.nd4j.TFGraphRunnerService;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.common.primitives.Pair;
import org.tensorflow.framework.AttrValue;
import org.tensorflow.framework.GraphDef;
import org.tensorflow.framework.NodeDef;
import com.google.gson.Gson;
import org.nd4j.shade.protobuf.Message;
import org.nd4j.shade.protobuf.TextFormat;

import java.util.*;
import java.util.List;


@Slf4j
@Data
public class TFOpLayerImpl extends AbstractLayer<TFOpLayer> {


    private Map nodeDef;
    private Map constants;
    private List<String> inputNames;
    TFGraphRunnerService graphRunnerService;

    public TFOpLayerImpl(Map nodeDef, Map constants, NeuralNetConfiguration conf, DataType dtype){
        super(conf, dtype);
        this.nodeDef = nodeDef;
        this.constants = constants;
        setGraphRunner();
    }

    @Override
    public Pair<Gradient, INDArray> backpropGradient(INDArray epsilon, LayerWorkspaceMgr workspaceMgr){
        throw new RuntimeException("Backprop through TFOpLayerImpl is not supported yet." +
                " TFOpLayerImpl is created when importing TensorFlow 2.0 Keras models " +
                "(tf.keras) into DL4J, that contains TensorFlow operations not just Keras layers.");
    }

    /**
     * Converts a Map representation of Nodedef to a singleton TF Graph and instantiates a GraphRunner.
     */
    private void setGraphRunner() {
        try{
            String json = new Gson().toJson(nodeDef);
            NodeDef.Builder builder = NodeDef.newBuilder();
            org.nd4j.shade.protobuf.util.JsonFormat.parser().merge(json, builder);
            NodeDef nodeDef = builder.build();
            List<String> allInputNames = new ArrayList<>(); // including constants
            Map<String, String> inputDataTypes = new HashMap<>();
            Map<String, INDArray> constArrays = new HashMap();
            this.inputNames = new ArrayList<>();
            List<String> outputNames = Arrays.asList(nodeDef.getName());
            Map<String, AttrValue> attrMap = nodeDef.getAttrMap();
            for (int i = 0; i < nodeDef.getInputCount(); i++){
                String inputName = nodeDef.getInput(i);
                String[] split = inputName.split("/");
                String attrKey;
                if (split.length == 1){
                    attrKey = "T";
                }
                else{
                    attrKey = "T" + split[split.length - 1];
                }
                allInputNames.add(nodeDef.getInput(i));
                inputDataTypes.put(nodeDef.getInput(i), attrMap.get(attrKey).getType().toString());
                if (constants.containsKey(String.valueOf(i))){
                    constArrays.put(nodeDef.getInput(i), Nd4j.create((List<Number>)constants.get(String.valueOf(i))));
                }
                else{
                    this.inputNames.add(nodeDef.getInput(i));
                }
            }
            String graph = "node{\n" + nodeDef.toString() + "\n}\nversions {\n producer: 22\n}";
            for (int i = 0; i < allInputNames.size(); i++){
                String inpName = allInputNames.get(i);
                String dtype = inputDataTypes.get(inpName);
                graph = "node{\nname: \"" + inpName + "\"\nop: \"Placeholder\"\nattr{\nkey: \"dtype\"\n value {\n type: " + dtype + "}\n}\n}\n" + graph;
            }
            log.info(graph);
            GraphDef.Builder graphDefBuilder = GraphDef.newBuilder();
            TextFormat.getParser().merge(graph, graphDefBuilder);
            GraphDef graphDef = graphDefBuilder.build();
            org.nd4j.shade.protobuf.ByteString serialized = graphDef.toByteString();
            byte[] graphBytes = serialized.toByteArray();

            ServiceLoader<TFGraphRunnerService> sl = ServiceLoader.load(TFGraphRunnerService.class);
            Iterator<TFGraphRunnerService> iter = sl.iterator();
            if (!iter.hasNext()){
                throw new RuntimeException("The model contains a Tensorflow Op, which requires the nd4j-tensorflow dependency to execute.");
            }

            this.graphRunnerService = iter.next().init(allInputNames, outputNames, graphBytes, constArrays, inputDataTypes);
        }
        catch (Exception e){
            throw new RuntimeException("Error parsing protobuf", e);
        }

    }

    private INDArray runGraph(INDArray input){
        Map<String, INDArray> inputMap = new HashMap<>();
        inputMap.put(inputNames.get(0), input);
        INDArray out = graphRunnerService.run(inputMap).values().toArray(new INDArray[0])[0];
        return out;
    }

    public long[] getOutputShape(long[] inputShape){
        long[] shape = ArrayUtils.clone(inputShape);
        for(int i = 0; i < shape.length; i++){
            if (shape[i] < 0){
                shape[i] = 1;
            }
        }
        INDArray dummyArr = Nd4j.zeros(shape);
        return runGraph(dummyArr).shape();
    }

    @Override
    public INDArray activate(boolean training, LayerWorkspaceMgr workspaceMgr){
        return runGraph(input);
    }


    @Override
    public boolean isPretrainLayer(){
        return false;
    }

    @Override
    public void clearNoiseWeightParams(){

    }

}
