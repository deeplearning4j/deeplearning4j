package org.nd4j.linalg.api.ops.impl.layers.convolution;

import lombok.Builder;
import lombok.Getter;
import lombok.extern.slf4j.Slf4j;
import lombok.val;
import org.nd4j.autodiff.functions.DifferentialFunction;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.imports.converters.DifferentialFunctionClassHolder;
import org.nd4j.imports.descriptors.properties.AttributeAdapter;
import org.nd4j.imports.descriptors.properties.PropertyMapping;
import org.nd4j.imports.descriptors.properties.adapters.ConditionalFieldValueNDArrayShapeAdapter;
import org.nd4j.imports.descriptors.properties.adapters.IntArrayIntIndexAdpater;
import org.nd4j.imports.descriptors.properties.adapters.StringEqualsAdapter;
import org.nd4j.imports.graphmapper.tf.TFGraphMapper;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.DynamicCustomOp;
import org.nd4j.linalg.api.ops.impl.layers.convolution.config.Conv3DConfig;
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

    public Conv3D() {
    }

    @Builder(builderMethodName = "builder")
    public Conv3D(SameDiff sameDiff, SDVariable[] inputFunctions, INDArray[] inputs, INDArray[] outputs,
                  Conv3DConfig conv3DConfig) {
        super(null, sameDiff, inputFunctions, false);
        setSameDiff(sameDiff);

        if (inputs != null)
            addInputArgument(inputs);
        if (outputs != null)
            addOutputArgument(outputs);
        this.config = conv3DConfig;
        addArgs();


        //for (val arg: iArgs())
        //  System.out.println(getIArgument(arg));
    }


    private void addArgs() {
        addIArgument(
                // TODO: support bias terms
//                ArrayUtil.fromBoolean(getConfig().isBiasUsed()),
                getConfig().getKT(),
                getConfig().getKH(),
                getConfig().getKW(),

                getConfig().getDT(),
                getConfig().getDH(),
                getConfig().getDW(),

                getConfig().getPT(),
                getConfig().getPH(),
                getConfig().getPW(),

                getConfig().getdD(),
                getConfig().getdH(),
                getConfig().getdW(),

                getConfig().isValidMode() ? 0 : 1,
                getConfig().isNCDHW() ? 0 : 1
        );

    }


    @Override
    public Object getValue(Field property) {
        if (config == null) {
            config = Conv3DConfig.builder().build();
        }

        return config.getValue(property);
    }

    @Override
    public void setValueFor(Field target, Object value) {
        if (config == null) {
            config = Conv3DConfig.builder().build();
        }

        if (target != null)
            config.setValueFor(target, value);
    }

    @Override
    public long[] iArgs() {
        if (iArguments.size() == 0)
            addArgs();

        return super.iArgs();
    }

    @Override
    public Map<String, Map<String, AttributeAdapter>> attributeAdaptersForFunction() {
        Map<String, Map<String, AttributeAdapter>> ret = new LinkedHashMap<>();
        Map<String, AttributeAdapter> tfAdapters = new LinkedHashMap<>();
        val fields = DifferentialFunctionClassHolder.getInstance().getFieldsForFunction(this);

        tfAdapters.put("kT", new ConditionalFieldValueNDArrayShapeAdapter("NDHWC", 0, 2, fields.get("dataFormat")));
        tfAdapters.put("kH", new ConditionalFieldValueNDArrayShapeAdapter("NDHWC", 1, 3, fields.get("dataFormat")));
        tfAdapters.put("kW", new ConditionalFieldValueNDArrayShapeAdapter("NDHWC", 2, 4, fields.get("dataFormat")));

        tfAdapters.put("dT", new IntArrayIntIndexAdpater(1));
        tfAdapters.put("dH", new IntArrayIntIndexAdpater(2));
        tfAdapters.put("dW", new IntArrayIntIndexAdpater(3));

        tfAdapters.put("pT", new IntArrayIntIndexAdpater(1));
        tfAdapters.put("pH", new IntArrayIntIndexAdpater(2));
        tfAdapters.put("pW", new IntArrayIntIndexAdpater(3));


        tfAdapters.put("isValidMode", new StringEqualsAdapter("VALID"));
        tfAdapters.put("isNCDHW", new StringEqualsAdapter("NCDHW"));

        ret.put(tensorflowName(), tfAdapters);

        return ret;
    }

    @Override
    public Map<String, Object> propertiesForFunction() {
        if (config == null) {
            return Collections.emptyMap();
        }
        return config.toProperties();
    }

    @Override
    public String opName() {
        return "conv3dnew";
    }


    @Override
    public Map<String, Map<String, PropertyMapping>> mappingsForFunction() {
        Map<String, Map<String, PropertyMapping>> ret = new HashMap<>();
        Map<String, PropertyMapping> map = new HashMap<>();


        val kernelMapping = PropertyMapping.builder()
                .propertyNames(new String[]{"kT", "kW", "kH"})
                .tfInputPosition(1)
                .onnxAttrName("kernel_shape")
                .build();

        val strideMapping = PropertyMapping.builder()
                .tfAttrName("strides")
                .onnxAttrName("strides")
                .propertyNames(new String[]{"dT", "dW", "dH"})
                .build();

        val dilationMapping = PropertyMapping.builder()
                .onnxAttrName("dilations")
                .propertyNames(new String[]{"dD", "dH", "dW"})
                .tfAttrName("rates")
                .build();

        val sameMode = PropertyMapping.builder()
                .onnxAttrName("auto_pad")
                .propertyNames(new String[]{"isValidMode"})
                .tfAttrName("padding")
                .build();

        val paddingWidthHeight = PropertyMapping.builder()
                .onnxAttrName("padding")
                .propertyNames(new String[]{"pT", "pW", "pH"})
                .build();

        val dataFormat = PropertyMapping.builder()
                .onnxAttrName("data_format")
                .tfAttrName("data_format")
                .propertyNames(new String[]{"dataFormat"})
                .build();


        val outputPadding = PropertyMapping.builder()
                .propertyNames(new String[]{"aT", "aH", "aW"})
                .build();


        val biasUsed = PropertyMapping.builder()
                .propertyNames(new String[]{"biasUsed"})
                .build();


        for (val propertyMapping : new PropertyMapping[]{
                kernelMapping,
                strideMapping,
                dilationMapping,
                sameMode,
                paddingWidthHeight,
                dataFormat,
                outputPadding, biasUsed}) {
            for (val keys : propertyMapping.getPropertyNames())
                map.put(keys, propertyMapping);
        }

        ret.put(onnxName(), map);
        ret.put(tensorflowName(), map);
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
        if (numIArguments() < 1) {
            addArgs();
        }

        if (numInputArguments() < getDescriptor().getNumIArgs()) {
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
