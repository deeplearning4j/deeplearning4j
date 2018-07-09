package org.deeplearning4j.nn.conf.layers.samediff;

import lombok.Data;
import org.nd4j.base.Preconditions;

import java.util.Arrays;
import java.util.List;

/**
 * SDVertexParams is used to define the inputs - and the parameters - for a SameDiff layer
 *
 * @author Alex Black
 */
@Data
public class SDVertexParams extends SDLayerParams {

    protected List<String> inputs;

    /**
     * Define the inputs to the DL4J SameDiff Vertex with specific names
     * @param inputNames Names of the inputs. Number here also defines the number of vertex inputs
     * @see #defineInputs(int)
     */
    public void defineInputs(String... inputNames){
        Preconditions.checkArgument(inputNames != null && inputNames.length > 0,
                "Input names must not be null, and must have length > 0: got %s", inputNames);
        this.inputs = Arrays.asList(inputNames);
    }

    /**
     * Define the inputs to the DL4J SameDiff vertex with generated names. Names will have format "input_0", "input_1", etc
     *
     * @param numInputs Number of inputs to the vertex.
     */
    public void defineInputs(int numInputs){
        Preconditions.checkArgument(numInputs > 0, "Number of inputs must be > 0: Got %s", numInputs);
        String[] inputNames = new String[numInputs];
        for( int i=0; i<numInputs; i++ ){
            inputNames[i] = "input_" + i;
        }
    }

}
