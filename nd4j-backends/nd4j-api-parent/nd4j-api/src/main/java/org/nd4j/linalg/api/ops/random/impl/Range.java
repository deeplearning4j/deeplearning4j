package org.nd4j.linalg.api.ops.random.impl;

import lombok.NonNull;
import lombok.val;
import onnx.OnnxProto3;
import org.nd4j.autodiff.functions.DifferentialFunction;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.imports.graphmapper.tf.TFGraphMapper;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.Op;
import org.nd4j.linalg.api.ops.random.BaseRandomOp;
import org.nd4j.linalg.factory.Nd4j;
import org.tensorflow.framework.AttrValue;
import org.tensorflow.framework.GraphDef;
import org.tensorflow.framework.NodeDef;

import java.util.List;
import java.util.Map;

/**
 * Range Op implementation, generates from..to distribution within Z
 *
 * @author raver119@gmail.com
 */
public class Range extends BaseRandomOp {
    private double from;
    private double to;
    private double delta;
    //used for initWithArrays when there are place holder
    //values that need to be resolved
    private int[] fromVertexId,toVertexId,deltaVertexId;
    public Range() {
        // no-op
    }

    public Range(double from, double to, int length) {
        this(Nd4j.createUninitialized(new int[] {1, length}, Nd4j.order()), from, to);
    }

    public Range(@NonNull INDArray z, double from, double to) {
        this.from = from;
        this.to = to;
        init(null, null, z, z.lengthLong());
        this.extraArgs = new Object[] {from, to};
    }

    @Override
    public int opNum() {
        return 4;
    }

    @Override
    public String opName() {
        return "range";
    }

    @Override
    public String onnxName() {
        return "Range";
    }

    @Override
    public String tensorflowName() {
        return "Range";
    }


    @Override
    public List<int[]> calculateOutputShape() {
        return super.calculateOutputShape();
    }

    @Override
    public void initFromTensorFlow(NodeDef nodeDef, SameDiff initWith, Map<String, AttrValue> attributesForNode, GraphDef graph) {
        super.initFromTensorFlow(nodeDef, initWith, attributesForNode, graph);

        NodeDef startNode = null,endNode = null,deltaNode = null;
        for(val  node : graph.getNodeList()) {
            if(node.getName().equals(nodeDef.getInput(0))) {
                startNode = node;
            }
            if(node.getName().equals(nodeDef.getInput(1))) {
                endNode = node;
            }
            if(node.getName().equals(nodeDef.getInput(2))) {
                deltaNode = node;
            }

            if(startNode != null && endNode != null && deltaNode != null)
                break;
        }

        val start = TFGraphMapper.getInstance().getNDArrayFromTensor("value",startNode,graph);
        val end = TFGraphMapper.getInstance().getNDArrayFromTensor("value",endNode,graph);
        val delta = TFGraphMapper.getInstance().getNDArrayFromTensor("value",deltaNode,graph);
        if(start != null && end != null && delta != null) {
            this.from = start.getDouble(0);
            this.to = end.getDouble(0);
            this.delta = delta.getDouble(0);

        }

        this.fromVertexId = sameDiff.getVariable(startNode.getName()).resultVertexId();
        this.toVertexId = sameDiff.getVariable(endNode.getName()).getVertexId();
        this.deltaVertexId = sameDiff.getVariable(deltaNode.getName()).getVertexId();





    }

    @Override
    public void initFromOnnx(OnnxProto3.NodeProto node, SameDiff initWith, Map<String, OnnxProto3.AttributeProto> attributesForNode, OnnxProto3.GraphProto graph) {
        super.initFromOnnx(node, initWith, attributesForNode, graph);
    }

    @Override
    public int[] getResultShape() {
        return new int[] {1, Math.max(1,(int) (from - to))};
    }

    @Override
    public void initWithArrays(Map<String, INDArray> arrayMap) {
        super.initWithArrays(arrayMap);
        val start = sameDiff.getVariableForVertexId(fromVertexId).getArr();
        val end = sameDiff.getVariableForVertexId(toVertexId).getArr();
        val delta = sameDiff.getVariableForVertexId(deltaVertexId).getArr();
        if(start != null && end != null && delta != null) {
            this.from = start.getDouble(0);
            this.to = end.getDouble(0);
            this.delta = delta.getDouble(0);
        }

    }

    @Override
    public List<DifferentialFunction> doDiff(List<DifferentialFunction> f1) {
        throw new UnsupportedOperationException("Unable to differentiate array creation routine");
    }

    @Override
    public Op.Type opType() {
        return Op.Type.RETURN;
    }
}
