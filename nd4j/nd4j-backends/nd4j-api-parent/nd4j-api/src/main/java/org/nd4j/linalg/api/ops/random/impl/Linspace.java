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

package org.nd4j.linalg.api.ops.random.impl;

import lombok.NonNull;
import lombok.val;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.imports.NoOpNameFoundException;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.imports.graphmapper.tf.TFGraphMapper;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.random.BaseRandomOp;
import org.nd4j.linalg.api.shape.LongShapeDescriptor;
import org.nd4j.linalg.factory.Nd4j;
import org.tensorflow.framework.AttrValue;
import org.tensorflow.framework.GraphDef;
import org.tensorflow.framework.NodeDef;

import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * Linspace/arange Op implementation, generates from..to distribution within Z
 *
 * @author raver119@gmail.com
 */
public class Linspace extends BaseRandomOp {
    private double from;
    private double to;
    private long length;

    public Linspace() {
        // no-op
    }

    public Linspace(double from, double to, int length, DataType dataType) {
        this(Nd4j.createUninitialized(dataType, new long[] {1, length}, Nd4j.order()), from, to);
    }

    public Linspace(@NonNull INDArray z, double from, double to) {
        this.from = from;
        this.to = to;
        this.length = z.length();
        init(null, null, z, z.lengthLong());
        this.extraArgs = new Object[] {from, to};
    }

    public Linspace(SameDiff sd, double from, double to, long length){
        super(sd, new long[]{length});
        this.sameDiff = sd;
        this.from = from;
        this.to = to;
        this.length = length;
        this.extraArgs = new Object[] {from, to};
    }


    @Override
    public int opNum() {
        return 4;
    }

    @Override
    public String opName() {
        return "linspace";
    }

    @Override
    public String onnxName() {
        throw new NoOpNameFoundException("No onnx op opName found for " +  opName());
    }

    @Override
    public String tensorflowName() {
        return "LinSpace";
    }

    @Override
    public void initFromTensorFlow(NodeDef nodeDef, SameDiff initWith, Map<String, AttrValue> attributesForNode, GraphDef graph) {
        String thisName = nodeDef.getName();
        INDArray start = null;
        INDArray stop = null;
        INDArray num = null;
        List<NodeDef> allNodes = graph.getNodeList();
        for(NodeDef nd : allNodes){
            String n = nd.getName();
            if(nodeDef.getInput(0).equals(n)){            //"thisName/start"
                start = TFGraphMapper.getInstance().getNDArrayFromTensor(n, nd, graph);
            } else if(nodeDef.getInput(1).equals(n)){     //"thisName/start"
                stop = TFGraphMapper.getInstance().getNDArrayFromTensor(n, nd, graph);
            } else if(nodeDef.getInput(2).equals(n)){     //"thisName/start"
                num = TFGraphMapper.getInstance().getNDArrayFromTensor(n, nd, graph);
            }

            if(start != null && stop != null && num != null)
                break;
        }
        if(start == null || stop == null || num == null)
            throw new IllegalStateException("Could not find expected input arrays: start=" + start + ", stop=" + stop + ", num=" + num);
        this.from = start.getDouble(0);
        this.to = stop.getDouble(0);
        this.length = (long)num.getDouble(0);
        this.extraArgs = new Object[]{from, to};
    }

    @Override
    public void resolvePropertiesFromSameDiffBeforeExecution() {
        //Workaround for: https://github.com/deeplearning4j/deeplearning4j/issues/6723
        super.resolvePropertiesFromSameDiffBeforeExecution();

        boolean update = false;
        if(x != null) {
            this.from = x.getDouble(0);
            update = true;
        }
        if(y != null) {
            this.to = y.getDouble(0);
            update = true;
        }
        if(z != null && z.length() == 1) {
            this.length = (long) z.getDouble(0);
        }
        if(update) {
            this.extraArgs = new Object[]{from, to};
        }

        x = null;
        y = null;
    }

    @Override
    public INDArray x(){
        //Workaround/hack for: https://github.com/deeplearning4j/deeplearning4j/issues/6723
        //If x or y is present, can't execute this op properly (wrong signature is used)
        return null;
    }

    @Override
    public INDArray y(){
        //Workaround/hack for: https://github.com/deeplearning4j/deeplearning4j/issues/6723
        //If x or y is present, can't execute this op properly (wrong signature is used)
        return null;
    }

    @Override
    public void setX(INDArray x){
        //Workaround/hack for: https://github.com/deeplearning4j/deeplearning4j/issues/6723
        //If x or y is present, can't execute this op properly (wrong signature is used)
        this.x = null;
    }

    @Override
    public void setY(INDArray y){
        //Workaround for: https://github.com/deeplearning4j/deeplearning4j/issues/6723
        //If x or y is present, can't execute this op properly (wrong signature is used)
        this.y = null;
    }

    @Override
    public List<LongShapeDescriptor> calculateOutputShape() {
        return Collections.singletonList(LongShapeDescriptor.fromShape(new long[]{length}, DataType.FLOAT));      //TODO Don't hardcode float!
    }


    @Override
    public List<SDVariable> doDiff(List<SDVariable> f1) {
        //No inputs
        return Collections.emptyList();
    }
}
