package org.nd4j.linalg.api.ops.impl.shape;

import lombok.NoArgsConstructor;
import lombok.val;
import onnx.OnnxProto3;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.imports.descriptors.properties.PropertyMapping;
import org.nd4j.imports.graphmapper.onnx.OnnxGraphMapper;
import org.nd4j.imports.graphmapper.tf.TFGraphMapper;
import org.nd4j.linalg.api.ndarray.INDArray;
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
                INDArray a = Nd4j.create(ArrayUtil.toFloats(indices));
                if (indices.length > 1)
                    a = a.reshape(indices.length);
                else
                    a = a.reshape(new int[]{});

                addInputArgument(args()[0].getArr(), a);
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
                .onnxAttrName("indices")
                .tfInputPosition(1)
                .propertyNames(new String[]{"indices"}).build();

        map.put("indices", broadcast);

        ret.put(tensorflowNames()[0], map);
        ret.put(onnxName(), map);

        Map<String, PropertyMapping> map2 = new HashMap<>();
        val broadcast2 = PropertyMapping.builder()
                .tfInputPosition(1)
                .propertyNames(new String[]{"indices"}).build();
        map2.put("indices", broadcast2);

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
    public List<SDVariable> doDiff(List<SDVariable> i_v){
        //2 args: input and indices. Plus integer dimension arg
        //Gather backprop is just scatter add

        SDVariable indicesGrad = sameDiff.zerosLike(arg(1));
        SDVariable inputGrad = sameDiff.zerosLike(arg(0));

        int ndim = arg(0).getShape().length;
        if(axis < 0){
            axis += ndim;
        }

        if(axis == 0){
            inputGrad = sameDiff.scatterAdd(inputGrad, arg(1), i_v.get(0));
        } else {
            int ndim = arg(0).getShape().length;
            int a = this.axis;
            if (a < 0) {
                a += ndim;
            }
            int[] permDims = new int[ndim];
            permDims[0] = a;
            for(int i=0; i<a; i++){
                permDims[i+1] = i;
            }
            for(int i=a+1; i<ndim; i++){
                permDims[i] = i;
            }
            inputGrad = sameDiff.permute(inputGrad, permDims);
            SDVariable i_v_transposed = sameDiff.permute(i_v.get(0), permDims);
            inputGrad = sameDiff.scatterAdd(inputGrad, arg(1), i_v_transposed);
            int[] reverseDims = new int[ndim];
            for(int i=0; i<ndim; i++){
                reverseDims[permDims[i]] = i;
            }
            inputGrad = sameDiff.permute(inputGrad, reverseDims);
        }

        return Arrays.asList(inputGrad, indicesGrad);
    }
}
