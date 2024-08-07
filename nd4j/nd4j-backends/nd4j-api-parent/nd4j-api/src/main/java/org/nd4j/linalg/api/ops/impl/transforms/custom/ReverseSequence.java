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

package org.nd4j.linalg.api.ops.impl.transforms.custom;

import lombok.val;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.imports.NoOpNameFoundException;
import org.nd4j.imports.descriptors.properties.PropertyMapping;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.DynamicCustomOp;
import org.tensorflow.framework.AttrValue;
import org.tensorflow.framework.GraphDef;
import org.tensorflow.framework.NodeDef;

import java.util.*;

public class ReverseSequence extends DynamicCustomOp {


    int seqDim;
    int batchDim;



    public ReverseSequence(SameDiff sameDiff, SDVariable i_v, SDVariable seqLengths, int seqDim, int batchDim) {
        super(null, sameDiff, new SDVariable[]{i_v, seqLengths}, false);

        this.seqDim = seqDim;
        this.batchDim = batchDim;
        addArguments();

    }

    public ReverseSequence(SameDiff sameDiff, SDVariable i_v, SDVariable seqLengths) {
        super(null, sameDiff, new SDVariable[]{i_v, seqLengths}, false);
        this.seqDim = 1;
        this.batchDim = 0;
        addArguments();
    }

    public ReverseSequence(INDArray x, INDArray seq_lengths, int seqDim, int batchDim){
        super(new INDArray[]{x, seq_lengths}, null);
        this.seqDim = seqDim;
        this.batchDim = batchDim;
        addArguments();
    }

    public ReverseSequence(INDArray x, INDArray seq_lengths){
        this(x, seq_lengths, 1, 0);
    }

    private void addArguments(){
        addIArgument(seqDim);
        addIArgument(batchDim);
    }

    public ReverseSequence() {
    }

    @Override
    public String opName() {
        return "reverse_sequence";

    }

    @Override
    public void initFromTensorFlow(NodeDef nodeDef, SameDiff initWith, Map<String, AttrValue> attributesForNode, GraphDef graph) {
        throw new UnsupportedOperationException("Use the new Tensorflow Importer instead. This method is now removed.");

    }

    @Override
    public Map<String, Map<String, PropertyMapping>> mappingsForFunction() {
        Map<String, Map<String, PropertyMapping>> ret = new HashMap<>();
        Map<String, PropertyMapping> attrs = new LinkedHashMap<>();
        val seqDim = PropertyMapping.builder()
                .propertyNames(new String[]{"seqDim"})
                .tfAttrName("seq_dim")
                .build();
        val batchDim = PropertyMapping.builder()
                .propertyNames(new String[]{"batchDim"})
                .tfAttrName("batch_dim")
                .build();
        attrs.put("seqDim", seqDim);
        attrs.put("batchDim", batchDim);
        ret.put(tensorflowName(), attrs);
        return ret;
    }

    @Override
    public String onnxName() {
        throw new NoOpNameFoundException("No onnx op opName found for " + opName());
    }

    @Override
    public String tensorflowName() {
        return "ReverseSequence";
    }


    @Override
    public List<SDVariable> doDiff(List<SDVariable> f1) {
        SDVariable ret = sameDiff.reverseSequence(f1.get(0), arg(1), seqDim, batchDim);
        return Arrays.asList(ret, sameDiff.zerosLike(arg(1)));
    }

    @Override
    public List<DataType> calculateOutputDataTypes(List<DataType> dataTypes){
        return Collections.singletonList(dataTypes.get(0));
    }

}
