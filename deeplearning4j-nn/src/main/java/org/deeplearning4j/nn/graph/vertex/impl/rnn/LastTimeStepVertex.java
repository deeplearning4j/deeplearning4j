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

package org.deeplearning4j.nn.graph.vertex.impl.rnn;

import org.deeplearning4j.berkeley.Pair;
import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.gradient.Gradient;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.graph.vertex.BaseGraphVertex;
import org.deeplearning4j.nn.graph.vertex.VertexIndices;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.INDArrayIndex;
import org.nd4j.linalg.indexing.NDArrayIndex;

/** LastTimeStepVertex is used in the context of recurrent neural network activations, to go from 3d (time series)
 * activations to 2d activations, by extracting out the last time step of activations for each example.<br>
 * This can be used for example in sequence to sequence architectures, and potentially for sequence classification.
 * <b>NOTE</b>: Because RNNs may have masking arrays (to allow for examples/time series of different lengths in the same
 * minibatch), it is necessary to provide the same of the network input that has the corresponding mask array. If this
 * input does not have a mask array, the last time step of the input will be used for all examples; otherwise, the time
 * step of the last non-zero entry in the mask array (for each example separately) will be used.
 * @author Alex Black
 */
public class LastTimeStepVertex extends BaseGraphVertex {

    private String inputName;
    private int inputIdx;
    /** Shape of the forward pass activations */
    private int[] fwdPassShape;
    /** Indexes of the time steps that were extracted, for each example */
    private int[] fwdPassTimeSteps;

    public LastTimeStepVertex(ComputationGraph graph, String name, int vertexIndex, String inputName) {
        this(graph, name, vertexIndex, null, null, inputName);
    }


    public LastTimeStepVertex(ComputationGraph graph, String name, int vertexIndex, VertexIndices[] inputVertices,
                              VertexIndices[] outputVertices, String inputName ) {
        super(graph, name, vertexIndex, inputVertices, outputVertices);
        this.inputName = inputName;
        this.inputIdx = graph.getConfiguration().getNetworkInputs().indexOf(inputName);
        if(inputIdx == -1) throw new IllegalArgumentException("Invalid input name: \"" + inputName + "\" not found in list "
            + "of network inputs (" + graph.getConfiguration().getNetworkInputs() + ")");
    }

    @Override
    public boolean hasLayer() {
        return false;
    }

    @Override
    public boolean isOutputVertex() {
        return false;
    }

    @Override
    public Layer getLayer() {
        return null;
    }

    @Override
    public INDArray doForward(boolean training) {
        //First: get the mask arrays for the given input, if any
        INDArray[] inputMaskArrays = graph.getInputMaskArrays();
        INDArray mask = (inputMaskArrays != null ? inputMaskArrays[inputIdx] : null);

        //Then: work out, from the mask array, which time step of activations we want, extract activations
        //Also: record where they came from (so we can do errors later)
        fwdPassShape = inputs[0].shape();

        INDArray out;
        if(mask == null){
            //No mask array -> extract same (last) column for all
            int lastTS = inputs[0].size(2)-1;
            out = inputs[0].get(NDArrayIndex.all(),NDArrayIndex.all(),NDArrayIndex.point(lastTS));
            fwdPassTimeSteps = null;    //Null -> last time step for all examples
        } else {
            int[] outShape = new int[]{inputs[0].size(0),inputs[0].size(1)};
            out = Nd4j.create(outShape);

            //Want the index of the last non-zero entry in the mask array.
            //Check a little here by using mulRowVector([0,1,2,3,...]) and argmax
            int maxTsLength = fwdPassShape[2];
            INDArray row = Nd4j.linspace(0, maxTsLength - 1, maxTsLength);
            INDArray temp = mask.mulRowVector(row);
            INDArray lastElementIdx = Nd4j.argMax(temp,1);
            fwdPassTimeSteps = new int[fwdPassShape[0]];
            for( int i=0; i<fwdPassTimeSteps.length; i++ ){
                fwdPassTimeSteps[i] = (int)lastElementIdx.getDouble(i);
            }

            //Now, get and assign the corresponding subsets of 3d activations:
            for( int i=0; i<fwdPassTimeSteps.length; i++){
                out.putRow(i,inputs[0].get(NDArrayIndex.point(i),NDArrayIndex.all(),NDArrayIndex.point(fwdPassTimeSteps[i])));
            }
        }

        return out;
    }

    @Override
    public Pair<Gradient, INDArray[]> doBackward(boolean tbptt) {

        //Allocate the appropriate sized array:
        INDArray epsilonsOut = Nd4j.create(fwdPassShape);

        if(fwdPassTimeSteps == null){
            //Last time step for all examples
            epsilonsOut.put(new INDArrayIndex[]{NDArrayIndex.all(),NDArrayIndex.all(),NDArrayIndex.point(fwdPassShape[2]-1)},
                    epsilons[0]);
        } else {
            //Different time steps were extracted for each example
            for( int i=0; i<fwdPassTimeSteps.length; i++ ){
                epsilonsOut.put(new INDArrayIndex[]{NDArrayIndex.point(i),NDArrayIndex.all(),
                        NDArrayIndex.point(fwdPassTimeSteps[i])}, epsilons[0].getRow(i));
            }
        }
        return new Pair<>(null,new INDArray[]{epsilonsOut});
    }

    @Override
    public void setBackpropGradientsViewArray(INDArray backpropGradientsViewArray) {
        if(backpropGradientsViewArray != null) throw new RuntimeException("Vertex does not have gradients; gradients view array cannot be set here");
    }

    @Override
    public String toString(){
        return "LastTimeStepVertex(inputName="+inputName+")";
    }
}
