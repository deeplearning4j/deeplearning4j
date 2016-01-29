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

import com.fasterxml.jackson.annotation.JsonProperty;
import lombok.Data;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.inputs.InvalidInputTypeException;
import org.deeplearning4j.nn.graph.ComputationGraph;

import java.util.Arrays;

/** SubsetVertex is used to select a subset of the activations out of another GraphVertex.<br>
 * For example, a subset of the activations out of a layer.<br>
 * Note that this subset is specifying by means of an interval of the original activations.
 * For example, to get the first 10 activations of a layer (or, first 10 features out of a CNN layer) use
 * new SubsetVertex(0,9)
 * @author Alex Black
 */
@Data
public class SubsetVertex extends GraphVertex {

    private int from;
    private int to;

    /**
     * @param from The first column index, inclusive
     * @param to The last column index, inclusive
     */
    public SubsetVertex(@JsonProperty("from") int from, @JsonProperty("to") int to) {
        this.from = from;
        this.to = to;
    }

    @Override
    public SubsetVertex clone() {
        return new SubsetVertex(from,to);
    }

    @Override
    public boolean equals(Object o){
        if(!(o instanceof SubsetVertex)) return false;
        SubsetVertex s = (SubsetVertex)o;
        return s.from == from && s.to == to;
    }

    @Override
    public int hashCode(){
        return new Integer(from).hashCode() ^ new Integer(to).hashCode();
    }

    @Override
    public org.deeplearning4j.nn.graph.vertex.GraphVertex instantiate(ComputationGraph graph, String name, int idx) {
        return new org.deeplearning4j.nn.graph.vertex.impl.SubsetVertex(graph,name,idx,from,to);
    }

    @Override
    public InputType getOutputType(InputType... vertexInputs) throws InvalidInputTypeException {
        if(vertexInputs.length != 1){
            throw new InvalidInputTypeException("SubsetVertex expects single input type. Received: " + Arrays.toString(vertexInputs));
        }

        if(vertexInputs[0].getType() == InputType.Type.CNN){
            InputType.InputTypeConvolutional conv = (InputType.InputTypeConvolutional)vertexInputs[0];
            int depth = conv.getDepth();
            if(to >= depth){
                throw new InvalidInputTypeException("Invalid range: Cannot select depth subset [" + from + "," + to + "] inclusive from CNN activations with "
                    + " [depth,width,height] = [" + depth + "," + conv.getWidth() + "," + conv.getHeight() + "]" );
            }
            return InputType.convolutional(from-to+1,conv.getWidth(),conv.getHeight());
        } else {
            //FF or RNN inputs
            return vertexInputs[0];
        }
    }
}
