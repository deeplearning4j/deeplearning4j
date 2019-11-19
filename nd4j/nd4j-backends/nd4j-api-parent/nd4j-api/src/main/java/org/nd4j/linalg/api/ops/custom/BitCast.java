package org.nd4j.linalg.api.ops.custom;

import lombok.val;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.base.Preconditions;
import org.nd4j.imports.graphmapper.tf.TFGraphMapper;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.DynamicCustomOp;
import org.nd4j.linalg.api.shape.options.ArrayOptionsHelper;
import org.tensorflow.framework.AttrValue;
import org.tensorflow.framework.GraphDef;
import org.tensorflow.framework.NodeDef;

import java.util.Collections;
import java.util.List;
import java.util.Map;

public class BitCast extends DynamicCustomOp {
    public BitCast() {}

    public BitCast(INDArray in, DataType dataType, INDArray out) {
        this(in, dataType.toInt(), out);
    }

    public BitCast(INDArray in, int dataType, INDArray out) {
        inputArguments.add(in);
        outputArguments.add(out);
        iArguments.add(Long.valueOf(dataType));
    }

    public BitCast(INDArray in, DataType dataType) {
        this(in, dataType.toInt());
    }

    public BitCast(INDArray in, int dataType) {
        inputArguments.add(in);
        iArguments.add(Long.valueOf(dataType));
    }

    public BitCast(SameDiff sameDiff, SDVariable in, SDVariable dataType) {
        super("", sameDiff, new SDVariable[]{in, dataType});
    }

    @Override
    public void initFromTensorFlow(NodeDef nodeDef, SameDiff initWith, Map<String, AttrValue> attributesForNode, GraphDef graph) {
        TFGraphMapper.initFunctionFromProperties(nodeDef.getOp(), this, attributesForNode, nodeDef, graph);
        val t = nodeDef.getAttrOrDefault("type", null);
        val type = ArrayOptionsHelper.convertToDataType(t.getType());
        addIArgument(type.toInt());
    }

    @Override
    public String opName() {
        return "bitcast";
    }

    @Override
    public String tensorflowName() {
        return "Bitcast";
    }

    @Override
    public List<DataType> calculateOutputDataTypes(List<DataType> inputDataTypes){
        int n = args().length;
        Preconditions.checkState(inputDataTypes != null && inputDataTypes.size() == n, "Expected %s input data types for %s, got %s", n, getClass(), inputDataTypes);
        return Collections.singletonList(inputDataTypes.get(0));
    }
}