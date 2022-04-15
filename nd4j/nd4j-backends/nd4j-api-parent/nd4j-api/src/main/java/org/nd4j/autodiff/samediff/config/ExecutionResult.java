package org.nd4j.autodiff.samediff.config;

import lombok.Builder;
import lombok.Data;
import org.nd4j.autodiff.functions.DifferentialFunction;
import org.nd4j.common.base.Preconditions;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.OpContext;
import org.nd4j.linalg.factory.Nd4j;

import java.util.*;

@Builder
@Data
public class ExecutionResult {

    private Map<String,INDArray> outputs;
    private Map<String,SDValue> valueOutputs;




    public INDArray arrayWithName(String name) {
        return outputs.get(name);
    }

    public static  ExecutionResult createFrom(List<String> names,List<INDArray> input) {
        Preconditions.checkState(names.size() == input.size(),"Inputs and names must be equal size!");
        Map<String,INDArray> outputs = new LinkedHashMap<>();
        for(int i = 0; i < input.size(); i++) {
            outputs.put(names.get(i),input.get(i));
        }
        return ExecutionResult.builder()
                .outputs(outputs)
                .build();
    }

    public static ExecutionResult createValue(String name,SDValue inputs) {
        return ExecutionResult.builder()
                .valueOutputs(Collections.singletonMap(name,inputs))
                .build();
    }


    public static ExecutionResult createValue(String name,List inputs) {
        return ExecutionResult.builder()
                .valueOutputs(Collections.singletonMap(name,SDValue.create(inputs)))
                .build();
    }

    public static  ExecutionResult createFrom(String name,INDArray input) {
        return createFrom(Arrays.asList(name),Arrays.asList(input));
    }


    public static  ExecutionResult createFrom(DifferentialFunction func, OpContext opContext) {
        return createFrom(Arrays.asList(func.outputVariablesNames())
                ,opContext.getOutputArrays().toArray(new INDArray[opContext.getOutputArrays().size()]));
    }

    public static  ExecutionResult createFrom(List<String> names,INDArray[] input) {
        Preconditions.checkState(names.size() == input.length,"Inputs and names must be equal size!");
        Map<String,INDArray> outputs = new LinkedHashMap<>();
        for(int i = 0; i < input.length; i++) {
            outputs.put(names.get(i),input[i]);
        }
        return ExecutionResult.builder()
                .outputs(outputs)
                .build();
    }

    public INDArray[] outputsToArray(List<String> inputs) {
        if(valueOutputs != null) {
            INDArray[] ret =  new INDArray[valueOutputs.size()];
            int count = 0;
            for(Map.Entry<String,SDValue> entry : valueOutputs.entrySet()) {
                if(entry.getValue() != null)
                    ret[count++] = entry.getValue().getTensorValue();
            }
            return ret;
        } else if(outputs != null) {
            INDArray[] ret =  new INDArray[inputs.size()];
            for(int i = 0; i < inputs.size(); i++) {
                ret[i] = outputs.get(inputs.get(i));
            }

            return ret;
        } else {
            throw new IllegalStateException("No outputs to be converted.");
        }

    }


    public boolean hasValues() {
        return valueOutputs != null;
    }

    public boolean hasSingle() {
        return outputs != null;
    }


    public int numResults() {
        if(outputs != null && !outputs.isEmpty())
            return outputs.size();
        else if(valueOutputs != null && !valueOutputs.isEmpty())
            return valueOutputs.size();
        return 0;
    }




    public boolean valueExistsAtIndex(int index) {
        if (outputs != null)
            return resultAt(index) != null;
        else if (valueOutputs != null) {
            SDValue value = valueWithKey(valueAtIndex(index));
            if (value != null) {
                switch (value.getSdValueType()) {
                    case TENSOR:
                        return value.getTensorValue() != null;
                    case LIST:
                        return value.getListValue() != null;
                }
            }

        }

        return false;

    }


    public boolean isNull() {
        return valueOutputs == null && outputs == null;
    }


    public INDArray resultOrValueAt(int index, boolean returnDummy) {
        if(hasValues()) {
            SDValue sdValue = valueWithKeyAtIndex(index, returnDummy);
            if(sdValue != null)
                return sdValue.getTensorValue();
            return null;
        }
        else
            return resultAt(index);
    }


    private String valueAtIndex(int index) {
        Set<String> keys = valueOutputs != null ? valueOutputs.keySet() : outputs.keySet();
        int count = 0;
        for(String value : keys) {
            if(count == index)
                return value;
            count++;
        }

        return null;
    }

    public SDValue valueWithKeyAtIndex(int index, boolean returnDummy) {
        if(valueOutputs == null)
            return null;
        String key = valueAtIndex(index);
        if(valueOutputs.containsKey(key)) {
            SDValue sdValue = valueOutputs.get(key);
            if(sdValue != null && sdValue.getSdValueType() == SDValueType.LIST && returnDummy)
                return SDValue.create(Nd4j.empty(DataType.FLOAT));
            else
                return sdValue;
        }
        return valueOutputs.get(key);
    }

    public SDValue valueWithKey(String name) {
        if(valueOutputs == null)
            return null;
        return valueOutputs.get(name);
    }

    public INDArray resultAt(int index) {
        if(outputs == null) {
            return null;
        }

        String name = this.valueAtIndex(index);
        return outputs.get(name);
    }



}
