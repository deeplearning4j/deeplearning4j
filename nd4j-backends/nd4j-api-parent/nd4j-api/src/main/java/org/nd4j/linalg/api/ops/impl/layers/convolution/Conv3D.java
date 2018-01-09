package org.nd4j.linalg.api.ops.impl.layers.convolution;

import lombok.Builder;
import lombok.Getter;
import lombok.extern.slf4j.Slf4j;
import lombok.val;
import org.nd4j.autodiff.functions.DifferentialFunction;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.imports.descriptors.properties.PropertyMapping;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.DynamicCustomOp;
import org.nd4j.linalg.api.ops.impl.layers.convolution.config.Conv3DConfig;
import org.nd4j.linalg.util.ArrayUtil;

import java.util.*;


/**
 * Conv3D operation
 */
@Slf4j
@Getter
public class Conv3D extends DynamicCustomOp {

    protected Conv3DConfig config;

    public Conv3D() {}

    @Builder(builderMethodName = "builder")
    public Conv3D(SameDiff sameDiff, SDVariable[] inputFunctions,INDArray[] inputs, INDArray[] outputs,Conv3DConfig conv3DConfig) {
        super(null,sameDiff, inputFunctions, false);
        setSameDiff(sameDiff);

        if(inputs != null)
            addInputArgument(inputs);
        if(outputs != null)
            addOutputArgument(outputs);
        this.config = conv3DConfig;
        addArgs();

    }


    private void addArgs() {
        addIArgument(new int[]{getConfig().getDT(),
                getConfig().getDW(),
                getConfig().getDH(),
                getConfig().getPT(),
                getConfig().getPW(),
                getConfig().getPH(),
                getConfig().getDilationT(),
                getConfig().getDilationW(),
                getConfig().getDilationH(),
                getConfig().getAT(),
                getConfig().getAW(),
                getConfig().getAH(),
                ArrayUtil.fromBoolean(getConfig().isBiasUsed())});

    }

    @Override
    public Map<String, Object> propertiesForFunction() {
        if(config == null) {
            return Collections.emptyMap();
        }
        return config.toProperties();
    }

    @Override
    public String opName() {
        return "conv3d";
    }



    @Override
    public Map<String, Map<String, PropertyMapping>> mappingsForFunction() {
        Map<String,Map<String,PropertyMapping>> ret = new HashMap<>();
        Map<String,PropertyMapping> map = new HashMap<>();
        val strideMapping = PropertyMapping.builder()
                .tfAttrName("strides")
                .onnxAttrName("strides")
                .propertyNames(new String[]{"sx","sy"})
                .build();



        val kernelMapping = PropertyMapping.builder()
                .propertyNames(new String[]{"kh","kw"})
                .tfInputPosition(1)
                .onnxAttrName("kernel_shape")
                .build();

        val dilationMapping = PropertyMapping.builder()
                .onnxAttrName("dilations")
                .propertyNames(new String[]{"dw","dh"})
                .tfAttrName("rates")
                .build();



        val sameMode = PropertyMapping.builder()
                .onnxAttrName("auto_pad")
                .propertyNames(new String[]{"isSameMode"})
                .tfAttrName("padding")
                .build();

        val paddingWidthHeight = PropertyMapping.builder()
                .onnxAttrName("padding")
                .propertyNames(new String[]{"ph","pw"})
                .build();


        map.put("sx", strideMapping);
        map.put("sy", strideMapping);
        map.put("kh", kernelMapping);
        map.put("kw", kernelMapping);
        map.put("dw", dilationMapping);
        map.put("dh", dilationMapping);
        map.put("isSameMode",sameMode);
        map.put("ph", paddingWidthHeight);
        map.put("pw", paddingWidthHeight);

        ret.put(onnxName(),map);
        ret.put(tensorflowName(),map);
        return ret;
    }



    @Override
    public List<SDVariable> doDiff(List<SDVariable> f1) {
        List<SDVariable> ret = new ArrayList<>();
        List<DifferentialFunction> inputs = new ArrayList<>();
        inputs.addAll(Arrays.asList(args()));
        inputs.add(f1.get(0));
        Conv3DDerivative conv3DDerivative = Conv3DDerivative.derivativeBuilder()
                .conv3DConfig(config)
                .inputFunctions(args())
                .outputs(outputArguments())
                .inputFunctions(inputs.toArray(new SDVariable[inputs.size()]))
                .sameDiff(sameDiff)
                .build();
        ret.addAll(Arrays.asList(conv3DDerivative.outputVariables()));
        return ret;
    }



    @Override
    public String onnxName() {
        return "Conv";
    }

    @Override
    public String tensorflowName() {
        return "Conv3D";
    }
}
