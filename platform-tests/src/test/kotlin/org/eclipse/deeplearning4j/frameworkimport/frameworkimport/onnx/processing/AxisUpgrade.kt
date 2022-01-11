package org.eclipse.deeplearning4j.frameworkimport.frameworkimport.onnx.processing

import onnx.Onnx
import org.nd4j.linalg.api.buffer.DataType
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.samediff.frameworkimport.hooks.NodePreProcessorHook
import org.nd4j.samediff.frameworkimport.hooks.annotations.NodePreProcessor
import org.nd4j.samediff.frameworkimport.ir.IRGraph
import org.nd4j.samediff.frameworkimport.ir.IRNode
import org.nd4j.samediff.frameworkimport.onnx.ir.OnnxIRGraph
import org.nd4j.shade.guava.primitives.Ints
import org.nd4j.shade.protobuf.GeneratedMessageV3
import org.nd4j.shade.protobuf.ProtocolMessageEnum

@NodePreProcessor(nodeTypes = arrayOf("MeanVarianceNormalization","ReduceL1",
    "ReduceL2","ReduceLogSum","ReduceLogSumExp","ReduceMax","ReduceMean","ReduceMin","ReduceProd","ReduceSum",
    "ReduceSumSquare","Slice","Squeeze","Unsqueeze"),frameworkName = "onnx")
class AxisUpgrade: NodePreProcessorHook<Onnx.NodeProto,Onnx.TensorProto,Onnx.AttributeProto,Onnx.AttributeProto,Onnx.TensorProto.DataType> {

    override fun modifyNode(
        node: IRNode<Onnx.NodeProto, Onnx.TensorProto, Onnx.AttributeProto, Onnx.AttributeProto, Onnx.TensorProto.DataType>,
        graph: IRGraph<GeneratedMessageV3, GeneratedMessageV3, GeneratedMessageV3, GeneratedMessageV3, GeneratedMessageV3, GeneratedMessageV3, ProtocolMessageEnum>
    ): IRNode<Onnx.NodeProto, Onnx.TensorProto, Onnx.AttributeProto, Onnx.AttributeProto, Onnx.TensorProto.DataType> {
        val irGraph = graph as OnnxIRGraph
        if(node.hasAttribute("axes")) {
            val attrValue = node.removeAttribute("axes")
            val ints = attrValue.intsList
            node.addInput("${node.nodeName()}_axes")
            irGraph.addConstantNode("${node.nodeName()}_axes", Nd4j.create(Nd4j.createBuffer(Ints.toArray(ints)))
                .reshape(ints.size.toLong()).castTo(
                DataType.INT64))
            irGraph.updateNode(node)
        }

        return node
    }
}