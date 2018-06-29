package org.deeplearning4j.nn.conf.layers.samediff;

import lombok.Data;
import org.nd4j.base.Preconditions;

import java.util.Arrays;
import java.util.List;

@Data
public class SDVertexParams extends SDLayerParams {

    protected List<String> inputs;

    public void defineInputs(String... inputNames){
        Preconditions.checkArgument(inputNames != null && inputNames.length > 0,
                "Input names must not be null, and must have length > 0: got %s", inputNames);
        this.inputs = Arrays.asList(inputNames);
    }

    public void defineInputs(int numInputs){
        Preconditions.checkArgument(numInputs > 0, "Number of inputs must be > 0: Got %s", numInputs);
        String[] inputNames = new String[numInputs];
        for( int i=0; i<numInputs; i++ ){
            inputNames[i] = "input_" + i;
        }
    }

}
