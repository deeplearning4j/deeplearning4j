package org.nd4j.linalg.api.ops.impl.layers.convolution;

import lombok.Builder;
import lombok.Getter;
import lombok.extern.slf4j.Slf4j;
import lombok.val;
import org.nd4j.autodiff.functions.DifferentialFunction;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.imports.descriptors.properties.AttributeAdapter;
import org.nd4j.imports.descriptors.properties.PropertyMapping;
import org.nd4j.imports.descriptors.properties.adapters.IntArrayIntIndexAdpater;
import org.nd4j.imports.graphmapper.tf.TFGraphMapper;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.DynamicCustomOp;
import org.nd4j.linalg.api.ops.impl.layers.convolution.config.Conv3DConfig;
import org.nd4j.linalg.util.ArrayUtil;
import org.tensorflow.framework.AttrValue;
import org.tensorflow.framework.GraphDef;
import org.tensorflow.framework.NodeDef;

import java.lang.reflect.Field;
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
        addIArgument(new int[]{
                ArrayUtil.fromBoolean(getConfig().isBiasUsed()),
                getConfig().getDT(),
                getConfig().getDW(),
                getConfig().getDH(),
                getConfig().getPT(),
                getConfig().getPW(),
                getConfig().getPH(),
                getConfig().getDilationT(),
                getConfig().getDilationW(),
                getConfig().getDilationH()});

    }


    @Override
    public void setValueFor(Field target, Object value) {
        if(config == null) {
            config = Conv3DConfig.builder().build();
        }

        if(target != null)
            config.setValueFor(target,value);
    }


    @Override
    public Map<String, Map<String, AttributeAdapter>> attributeAdaptersForFunction() {
        Map<String,Map<String,AttributeAdapter>> ret = new LinkedHashMap<>();
        Map<String,AttributeAdapter> tfAdapters = new LinkedHashMap<>();

        tfAdapters.put("dT", new IntArrayIntIndexAdpater(1));
        tfAdapters.put("dW",  new IntArrayIntIndexAdpater(2));
        tfAdapters.put("dH",new IntArrayIntIndexAdpater(3));


        tfAdapters.put("pT", new IntArrayIntIndexAdpater(1));
        tfAdapters.put("pW",  new IntArrayIntIndexAdpater(2));
        tfAdapters.put("pH",new IntArrayIntIndexAdpater(3));

        ret.put(tensorflowName(),tfAdapters);

        return ret;
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
                .propertyNames(new String[]{"dT","dW","dH"})
                .build();



        val dilationMapping = PropertyMapping.builder()
                .onnxAttrName("dilations")
                .propertyNames(new String[]{"dilationT","dilationH","dilationW"})
                .tfAttrName("rates")
                .build();



        val sameMode = PropertyMapping.builder()
                .onnxAttrName("auto_pad")
                .propertyNames(new String[]{"isSameMode"})
                .tfAttrName("padding")
                .build();

        val paddingWidthHeight = PropertyMapping.builder()
                .onnxAttrName("padding")
                .propertyNames(new String[]{"pT","pW","pH"})
                .build();

        val dataFormat = PropertyMapping.builder()
                .onnxAttrName("data_format")
                .tfAttrName("data_format")
                .propertyNames(new String[]{"dataFormat"})
                .build();


        val outputPadding = PropertyMapping.builder()
                .propertyNames(new String[]{"aT","aH","aW"})
                .build();


        val biasUsed = PropertyMapping.builder()
                .propertyNames(new String[]{"biasUsed"})
                .build();




        for(val propertyMapping : new PropertyMapping[] {
                strideMapping,
                dilationMapping,
                sameMode,
                paddingWidthHeight,
                dataFormat,
                outputPadding,biasUsed}) {
            for(val keys : propertyMapping.getPropertyNames())
                map.put(keys,propertyMapping);

        }

        ret.put(onnxName(),map);
        ret.put(tensorflowName(),map);
        return ret;
    }


    @Override
    public void initFromTensorFlow(NodeDef nodeDef, SameDiff initWith, Map<String, AttrValue> attributesForNode, GraphDef graph) {
        TFGraphMapper.getInstance().initFunctionFromProperties(nodeDef.getOp(), this, attributesForNode, nodeDef, graph);

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
    public void resolvePropertiesFromSameDiffBeforeExecution() {
        if(numIArguments() < 1) {
            addArgs();
        }

        if(numInputArguments() < getDescriptor().getNumIArgs()) {
            populateInputsAndOutputsFromSameDiff();
        }


    }

    @Override
    public boolean isConfigProperties() {
        return true;
    }

    @Override
    public String configFieldName() {
        return "config";
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
