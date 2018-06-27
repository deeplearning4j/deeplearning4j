package org.deeplearning4j.nn.conf.layers.samediff;

import lombok.AllArgsConstructor;
import lombok.NoArgsConstructor;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.base.Preconditions;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.util.HashMap;
import java.util.Map;

//TODO is is possible to have a Lambda class that works as BOTH a Layer and a Vertex?
//i.e., works for both, but fail at runtime if user tries to do multiple inputs in CompGraph
public abstract class SameDiffLambdaVertex extends SameDiffVertex {

    protected VertexInputs inputs;

    public abstract SDVariable defineVertex(SameDiff sameDiff, VertexInputs inputs);

    @Override
    public SDVariable defineVertex(SameDiff sameDiff, Map<String, SDVariable> layerInput, Map<String, SDVariable> paramTable) {
        return defineVertex(sameDiff, inputs);
    }

    @Override
    public void defineParametersAndInputs(SDVertexParams params) {
        //No op, for lamda vertex
    }

    @Override
    public void initializeParameters(Map<String, INDArray> params) {
        //No op, for lambda vertex
    }

    protected VertexInputs getInputs(SameDiff sd){
        if(inputs == null){
            inputs = new VertexInputs(sd);
        }
        return inputs;
    }

    public class VertexInputs {

        private SameDiff sameDiff;
        private Map<Integer,SDVariable> map = new HashMap<>();

        protected VertexInputs(SameDiff sd){
            this.sameDiff = sd;
        }

        public SDVariable getInput(int inputNum){
            Preconditions.checkArgument(inputNum >= 0, "Input number must be >= 0." +
                    "Got: %s", inputNum);

            if(!map.containsKey(inputNum)){
                //Lazily define extra input variable as required
                SDVariable var = sameDiff.var("var_" + inputNum, (long[])null);
                map.put(inputNum, var);
            }

            return map.get(inputNum);
        }
    }
}
