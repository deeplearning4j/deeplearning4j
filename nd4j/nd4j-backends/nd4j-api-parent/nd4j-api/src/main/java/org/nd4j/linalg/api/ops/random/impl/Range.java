package org.nd4j.linalg.api.ops.random.impl;

import lombok.val;
import onnx.OnnxProto3;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.imports.graphmapper.tf.TFGraphMapper;
import org.nd4j.linalg.api.ops.DynamicCustomOp;
import org.nd4j.linalg.api.ops.Op;
import org.nd4j.linalg.factory.Nd4j;
import org.tensorflow.framework.AttrValue;
import org.tensorflow.framework.GraphDef;
import org.tensorflow.framework.NodeDef;

import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Map;

/**
 * Range Op implementation, generates from..to distribution within Z
 *
 * @author raver119@gmail.com
 */
public class Range extends DynamicCustomOp {
    private Double from;
    private Double to;
    private Double delta;
    //used for initWithArrays when there are place holder
    //values that need to be resolved
    private String fromVertexId,toVertexId,deltaVertexId;
    public Range() {
        // no-op
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
            val outputVars = outputVariables();
            this.from = start.getDouble(0);
            this.to = end.getDouble(0);
            this.delta = delta.getDouble(0);
            addTArgument(this.from,this.to,this.delta);
            val outputVertexId = outputVars[0].getVarName();
            if(sameDiff.getArrForVarName(outputVertexId) == null) {
                if(outputVars[0].getShape() == null) {
                    val calcShape = calculateOutputShape();
                    sameDiff.putShapeForVarName(outputVars[0].getVarName(),calcShape.get(0));
                }


                val arr = Nd4j.create(outputVars[0].getShape());
                initWith.putArrayForVarName(outputVertexId, arr);
                addOutputArgument(arr);

            }
        }

        val fromVar = initWith.getVariable(TFGraphMapper.getInstance().getNodeName(startNode.getName()));
        val toVar = initWith.getVariable(TFGraphMapper.getInstance().getNodeName(endNode.getName()));
        val deltaVar =  initWith.getVariable(TFGraphMapper.getInstance().getNodeName(deltaNode.getName()));

        this.fromVertexId = fromVar.getVarName();
        this.toVertexId = toVar.getVarName();
        this.deltaVertexId = deltaVar.getVarName();

    }



    @Override
    public void initFromOnnx(OnnxProto3.NodeProto node, SameDiff initWith, Map<String, OnnxProto3.AttributeProto> attributesForNode, OnnxProto3.GraphProto graph) {
        super.initFromOnnx(node, initWith, attributesForNode, graph);
    }




    @Override
    public List<long[]> calculateOutputShape() {
        val iArgs = iArgs();
        val tArgs = tArgs();
        val inputArgs = inputArguments();
        int cnt = 0;

        if (iArgs.length > 0) {
            int start = (int) iArgs[0];
            int stop = (int) iArgs[1];
            int step = (int) iArgs[2];

            double e = (double) start;
            if (start > stop) {
                while (e > (double) stop) {
                    cnt++;
                    e = (double) step > 0.0 ? e - step : e + step;
                }
            } else {
                while (e < (double) stop) {
                    cnt++;
                    e += step;
                }
            }

            return Arrays.asList(new long[]{cnt});
        }

        else if (tArgs.length > 0) {
            double start = tArgs[0];
            double stop = tArgs[1];
            double step = tArgs[2];

            double e = start;
            if (start > stop) {
                while (e > stop) {
                    cnt++;
                    e = step > 0.0 ? e - step : e + step;
                }
            } else {
                while (e < stop) {
                    cnt++;
                    e += step;
                }
            }

            return Arrays.asList(new long[]{cnt});
        }

        else if(inputArgs.length > 0) {
            double start = inputArgs[0].getDouble(0);
            double stop = inputArgs[1].getDouble(0);
            double step = inputArgs[2].getDouble(0);

            double e = start;
            if (start > stop) {
                while (e > stop) {
                    cnt++;
                    e = step > 0.0 ? e - step : e + step;
                }
            } else {
                while (e < stop) {
                    cnt++;
                    e += step;
                }
            }

            return Arrays.asList(new long[]{cnt});
        }


       return Collections.emptyList();

    }

    @Override
    public List<SDVariable> doDiff(List<SDVariable> f1) {
        throw new UnsupportedOperationException("Unable to differentiate array creation routine");
    }

    @Override
    public Op.Type opType() {
        return Op.Type.CUSTOM;
    }
}
