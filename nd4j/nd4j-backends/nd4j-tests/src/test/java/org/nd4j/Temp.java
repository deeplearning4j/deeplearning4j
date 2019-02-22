package org.nd4j;

import org.junit.Ignore;
import org.junit.Test;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.autodiff.samediff.VariableType;
import org.nd4j.base.Preconditions;
import org.nd4j.imports.graphmapper.tf.TFGraphMapper;
import org.nd4j.imports.tensorflow.TFImportOverride;
import org.nd4j.imports.tensorflow.TFImportStatus;
import org.nd4j.imports.tensorflow.TensorFlowImportValidator;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.transforms.pairwise.arithmetic.Axpy;
import org.nd4j.linalg.api.ops.impl.transforms.same.Identity;
import org.nd4j.linalg.api.ops.impl.transforms.strict.GELU;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.ops.transforms.Transforms;
import org.tensorflow.framework.AttrValue;
import org.tensorflow.framework.GraphDef;
import org.tensorflow.framework.NodeDef;

import java.io.File;
import java.util.*;

import static org.junit.Assert.assertEquals;

@Ignore
public class Temp {

    @Test
    public void testBert(){
        File f = new File("C:\\Temp\\TF_Graphs\\BERT_multi_cased_L-12_H-768_A-12\\BERT_multi_cased_L-12_H-768_A-12_frozen.pb");
        SameDiff sd = TFGraphMapper.getInstance().importGraph(f);
        System.out.println("INPUTS: " + sd.inputs());
        System.out.println("OUTPUTS: " + sd.outputs());

        for(String s : sd.inputs()){
            System.out.println(s + ": " + sd.getVariable(s));
        }
    }

    @Test
    public void testBert2() throws Exception {
        File f = new File("C:\\Temp\\TF_Graphs\\mrpc_output\\BERT_uncased_L-12_H-768_A-12_mrpc_frozen.pb");
        TFImportStatus status = TensorFlowImportValidator.checkModelForImport(f);
        System.out.println(status.getUnsupportedOpNames());
        System.out.println("SUPPORTED: " + status.getImportSupportedOpNames());

        SameDiff sd = TFGraphMapper.getInstance().importGraph(f);
        System.out.println("INPUTS: " + sd.inputs());
        System.out.println("OUTPUTS: " + sd.outputs());

        for(String s : sd.inputs()){
            System.out.println(s + ": " + sd.getVariable(s));
        }
    }

    @Test
    public void testGpt2() throws Exception {
        File f = new File("C:\\Temp\\TF_Graphs\\gpt-2\\models\\117M\\gpt-2_117M_frozen.pb");

        TFImportStatus status = TensorFlowImportValidator.checkModelForImport(f);
        System.out.println("UNSUPPORTED: " + status.getUnsupportedOpNames());
        System.out.println("OK: " + status.getImportSupportedOpNames());

        Map<String,TFImportOverride> map = new HashMap<>();
        map.put("convert_gradient_to_tensor_8rf1PQaJCP4", new ConvertToGradOverride());

        SameDiff sd = TFGraphMapper.getInstance().importGraph(f, map);
        System.out.println("INPUTS: " + sd.inputs());
        System.out.println("OUTPUTS: " + sd.outputs());

        for(String s : sd.inputs()){
            System.out.println(s + ": " + sd.getVariable(s));
        }
    }

//    public static class ConvertToGradOverride implements TFImportOverride {
//        @Override
//        public List<SDVariable> initFromTensorFlow(List<SDVariable> inputs, List<SDVariable> controlDepInputs, NodeDef nodeDef, SameDiff initWith, Map<String, AttrValue> attributesForNode, GraphDef graph) {
//            Preconditions.checkState(inputs != null && inputs.size() == 1, "Expected exactly 1 input");
//            Preconditions.checkState(controlDepInputs == null || controlDepInputs.isEmpty(), "Expected no control dependency inputs");
//            String baseName = nodeDef.getName();
//            String name = baseName;
//            int count = 1;
//            while(initWith.hasVariable(name)){
//                name = baseName + "_" + (count++);
//            }
//            SDVariable identity = initWith.identity(name,inputs.get(0));
//            return Collections.singletonList(identity);
//        }
//    }

    public static class ConvertToGradOverride implements TFImportOverride {
        @Override
        public List<SDVariable> initFromTensorFlow(List<SDVariable> inputs, List<SDVariable> controlDepInputs, NodeDef nodeDef, SameDiff initWith, Map<String, AttrValue> attributesForNode, GraphDef graph) {
            Preconditions.checkState(inputs != null && inputs.size() == 1, "Expected exactly 1 input");
            Preconditions.checkState(controlDepInputs == null || controlDepInputs.isEmpty(), "Expected no control dependency inputs");
            String baseName = nodeDef.getName();

            SDVariable out;
            if(initWith.hasVariable(baseName)){
                out = initWith.getVariable(baseName);
            } else {
                out = initWith.var(baseName, VariableType.ARRAY, null, inputs.get(0).dataType(), inputs.get(0).placeholderShape());
            }

            Identity i = new Identity(null, inputs.get(0));
            i.setOwnName(baseName);
            initWith.putFunctionForId(baseName, i);
            initWith.getOps().get(baseName).setOutputsOfOp(Collections.singletonList(out.getVarName()));
            initWith.getOps().get(baseName).setInputsToOp(Collections.singletonList(inputs.get(0).getVarName()));

            return Collections.singletonList(out);
        }
    }
}
