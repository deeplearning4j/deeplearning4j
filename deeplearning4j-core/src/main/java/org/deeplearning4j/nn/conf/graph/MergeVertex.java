/*
 *
 *  * Copyright 2016 Skymind,Inc.
 *  *
 *  *    Licensed under the Apache License, Version 2.0 (the "License");
 *  *    you may not use this file except in compliance with the License.
 *  *    You may obtain a copy of the License at
 *  *
 *  *        http://www.apache.org/licenses/LICENSE-2.0
 *  *
 *  *    Unless required by applicable law or agreed to in writing, software
 *  *    distributed under the License is distributed on an "AS IS" BASIS,
 *  *    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  *    See the License for the specific language governing permissions and
 *  *    limitations under the License.
 *
 */

package org.deeplearning4j.nn.conf.graph;


import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.inputs.InvalidInputTypeException;
import org.deeplearning4j.nn.graph.ComputationGraph;

/** A MergeVertex is used to combine the activations of two or more layers/GraphVertex by means of concatenation/merging.<br>
 * Exactly how this is done depends on the type of input.<br>
 * For 2d (feed forward layer) inputs: MergeVertex([numExamples,layerSize1],[numExamples,layerSize2]) -> [numExamples,layerSize1 + layerSize2]<br>
 * For 3d (time series) inputs: MergeVertex([numExamples,layerSize1,timeSeriesLength],[numExamples,layerSize2,timeSeriesLength])
 *      -> [numExamples,layerSize1 + layerSize2,timeSeriesLength]<br>
 * For 4d (convolutional) inputs: MergeVertex([numExamples,depth1,width,height],[numExamples,depth2,width,height])
 *      -> [numExamples,depth1 + depth2,width,height]<br>
 * @author Alex Black
 */
public class MergeVertex extends GraphVertex {

    @Override
    public MergeVertex clone() {
        return new MergeVertex();
    }

    @Override
    public boolean equals(Object o) {
        return o instanceof MergeVertex;
    }

    @Override
    public int hashCode(){
        return 433682566;
    }

    @Override
    public org.deeplearning4j.nn.graph.vertex.GraphVertex instantiate(ComputationGraph graph, String name, int idx) {
        return new org.deeplearning4j.nn.graph.vertex.impl.MergeVertex(graph,name,idx);
    }

    @Override
    public InputType getOutputType(InputType... vertexInputs) throws InvalidInputTypeException {
        if(vertexInputs.length == 1) return vertexInputs[0];
        InputType first = vertexInputs[0];
        if(first.getType() != InputType.Type.CNN){
            //FF or RNN data inputs
            int size = 0;
            boolean ff = true;
            for( int i=0; i<vertexInputs.length; i++ ){
                if(vertexInputs[i].getType() != first.getType()){
                    throw new InvalidInputTypeException("Invalid input: MergeVertex cannot merge activations of different types:"
                            + " first type = " + first.getType() + ", input type " + (i+1) + " = " + vertexInputs[i].getType());
                }

                int thisSize;
                if(vertexInputs[i] instanceof InputType.InputTypeFeedForward) thisSize = ((InputType.InputTypeFeedForward)vertexInputs[i]).getSize();
                else{
                    thisSize = ((InputType.InputTypeRecurrent)vertexInputs[i]).getSize();
                    ff = false;
                }
                if(thisSize <= 0){//Size is not defined
                    size = -1;
                } else {
                    size += thisSize;
                }
            }

            if(size > 0){
                //Size is specified
                if(ff) return InputType.feedForward(size);
                else return InputType.recurrent(size);
            } else {
                //size is unknown
                if(ff) return InputType.feedForward(-1);
                else return InputType.recurrent(-1);
            }
        } else {
            //CNN inputs... also check that the depth, width and heights match:
            InputType.InputTypeConvolutional firstConv = (InputType.InputTypeConvolutional)first;
            int fd = firstConv.getDepth();
            int fw = firstConv.getWidth();
            int fh = firstConv.getHeight();

            int depthSum = fd;

            for( int i=1; i<vertexInputs.length; i++ ){
                if(vertexInputs[i].getType() != InputType.Type.CNN){
                    throw new InvalidInputTypeException("Invalid input: MergeVertex cannot process activations of different types:"
                            + " first type = " + InputType.Type.CNN + ", input type " + (i+1) + " = " + vertexInputs[i].getType());
                }

                InputType.InputTypeConvolutional otherConv = (InputType.InputTypeConvolutional) vertexInputs[i];

                int od = otherConv.getDepth();
                int ow = otherConv.getWidth();
                int oh = otherConv.getHeight();

                if(fw != ow || fh != oh){
                    throw new InvalidInputTypeException("Invalid input: MergeVertex cannot merge CNN activations of different width/heights:"
                            + "first [depth,width,height] = [" + fd + "," + fw + "," + fh + "], input " + i + " = [" + od + "," + ow + "," + oh + "]");
                }

                depthSum += od;
            }

            return InputType.convolutional(fh,fw,depthSum);
        }
    }
}
