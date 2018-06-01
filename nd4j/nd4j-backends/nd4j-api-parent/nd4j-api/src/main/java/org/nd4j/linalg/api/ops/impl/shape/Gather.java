package org.nd4j.linalg.api.ops.impl.shape;

import lombok.NoArgsConstructor;
import lombok.val;
import onnx.OnnxProto3;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.imports.descriptors.properties.PropertyMapping;
import org.nd4j.imports.graphmapper.onnx.OnnxGraphMapper;
import org.nd4j.imports.graphmapper.tf.TFGraphMapper;
import org.nd4j.linalg.api.ops.DynamicCustomOp;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.util.ArrayUtil;
import org.tensorflow.framework.AttrValue;
import org.tensorflow.framework.GraphDef;
import org.tensorflow.framework.NodeDef;

import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * Gather op
 */
@NoArgsConstructor
public class Gather extends DynamicCustomOp {

    protected int[] indices;
    protected int axis = 0;


    public Gather(SameDiff sameDiff, SDVariable input, int[] indices, int axis, boolean inPlace) {
        super(null, sameDiff, new SDVariable[] {input}, inPlace);

        addIArgument(axis);
        addIArgument(indices);
        this.axis = axis;
        this.indices = indices;
    }

    public Gather(SameDiff sameDiff, SDVariable input, SDVariable indices, int axis, boolean inPlace) {
        super(null, sameDiff, new SDVariable[] {input, indices}, inPlace);
        addIArgument(axis);
        this.axis = axis;
    }

    @Override
    public String onnxName() {
        return "Gather";
    }


    @Override
    public String[] tensorflowNames() {
        return new String[]{"Gather", "GatherV2"};
    }

    @Override
    public void initFromTensorFlow(NodeDef nodeDef, SameDiff initWith, Map<String, AttrValue> attributesForNode, GraphDef graph) {
        TFGraphMapper.getInstance().initFunctionFromProperties(nodeDef.getOp(), this, attributesForNode, nodeDef, graph);

    }

    @Override
    public void initFromOnnx(OnnxProto3.NodeProto node, SameDiff initWith, Map<String, OnnxProto3.AttributeProto> attributesForNode, OnnxProto3.GraphProto graph) {
        OnnxGraphMapper.getInstance().initFunctionFromProperties(node.getOpType(), this, attributesForNode, node, graph);
    }


    @Override
    public void resolvePropertiesFromSameDiffBeforeExecution() {
        super.resolvePropertiesFromSameDiffBeforeExecution();
        if (indices != null && numInputArguments() < 2) {
            if (numInputArguments() == 0) {
                addInputArgument(args()[0].getArr(), Nd4j.create(ArrayUtil.toFloats(indices)).reshape(indices.length));

            } else if (numInputArguments() == 1) {
                addInputArgument(Nd4j.create(ArrayUtil.toFloats(indices)));
            }

        }

        if (numIArguments() < 1) {
            addIArgument(axis);
        }

        if (numOutputArguments() < getDescriptor().getNumOutputs()) {
            val outputs = outputVariables();
            //Check that ALL variables have an array before setting
            for(SDVariable v : outputs){
                if(v.getArr() == null){
                    return;
                }
            }

            for (int i = 0; i < outputs.length; i++) {
                val output = outputs[i].getArr();
                addOutputArgument(output);
            }
        }


    }

    @Override
    public Map<String, Map<String, PropertyMapping>> mappingsForFunction() {
        Map<String, Map<String, PropertyMapping>> ret = new HashMap<>();
        Map<String, PropertyMapping> map = new HashMap<>();
        val broadcast = PropertyMapping.builder()
                .onnxAttrName("broadcast")
                .tfInputPosition(1)
                .propertyNames(new String[]{"broadcast"}).build();

        map.put("broadcast", broadcast);

        ret.put(tensorflowNames()[0], map);
        ret.put(onnxName(), map);

        Map<String, PropertyMapping> map2 = new HashMap<>();
        val broadcast2 = PropertyMapping.builder()
                .tfInputPosition(1)
                .propertyNames(new String[]{"broadcast"}).build();
        map2.put("broadcast", broadcast2);

        val axis2 = PropertyMapping.builder()
                .tfInputPosition(2)
                .propertyNames(new String[]{"axis"}).build();
        map2.put("axis", axis2);

        ret.put("GatherV2", map2);


        return ret;
    }

    @Override
    public String opName() {
        return "gather";
    }

    @Override
    public List<SDVariable> doDiff(List<SDVariable> gradOut){
        //2 args: input and indices. Plus integer dimension arg
        //Gather backprop is just scatter add

        SDVariable indicesGrad = sameDiff.zerosLike(arg(1));
        SDVariable inputGrad = sameDiff.zerosLike(arg(0));

        if(axis == 0){
            inputGrad = sameDiff.scatterAdd(inputGrad, arg(1), gradOut.get(0));
        } else {
            throw new UnsupportedOperationException("Gather backprop for axis > 0 not yet implemented");
        }

        return Arrays.asList(inputGrad, indicesGrad);
    }
}
