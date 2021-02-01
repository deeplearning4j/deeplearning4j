package org.nd4j.codegen.ir.tensorflow

import org.nd4j.codegen.ir.BaseNDArrayMappingRule
import org.nd4j.codegen.ir.MultiInputIndexMappingRule
import org.nd4j.codegen.ir.findOp
import org.nd4j.codegen.ir.nd4jOpDescriptors
import org.nd4j.ir.OpNamespace
import org.nd4j.ir.TensorNamespace
import org.tensorflow.framework.*

class NDArrayMappingRule(mappingNamesToPerform: MutableMap<String, String>,
                         transformerArgs: Map<String, List<OpNamespace.ArgDescriptor>> = emptyMap()):
        BaseNDArrayMappingRule<GraphDef, OpDef, NodeDef, OpDef.AttrDef, AttrValue, TensorProto, DataType>(mappingNamesToPerform = mappingNamesToPerform, transformerArgs = transformerArgs) {



    override fun createTensorProto(input: TensorProto): TensorNamespace.TensorProto {
        return TensorflowIRTensor(input).toArgTensor()
    }


    override fun isInputTensorName(inputName: String): Boolean {
        val tfOp = tensorflowOps.findOp(mappingProcess!!.inputFrameworkOpName())
        return tfOp.inputArgList.map { input -> input.name }.contains(inputName)
    }

    override fun isOutputTensorName(outputName: String): Boolean {
        val nd4jOpDescriptor = nd4jOpDescriptors.findOp(mappingProcess!!.opName())
        return nd4jOpDescriptor.argDescriptorList.filter { inputDescriptor -> inputDescriptor.argType == OpNamespace.ArgDescriptor.ArgType.INPUT_TENSOR }
                .map {inputDescriptor -> inputDescriptor.name }.contains(outputName)
    }
}

fun mappingNDArrayInputs(inputs: MutableMap<String, String>) : NDArrayMappingRule {
    return NDArrayMappingRule(
        mappingNamesToPerform = inputs
    )
}

//MultiInputIndexMappingRule
class TensorflowMultiInputIndexMappingRule(mappingNamesToPerform: MutableMap<String, String>,
                         transformerArgs: Map<String, List<OpNamespace.ArgDescriptor>> = emptyMap()):
    MultiInputIndexMappingRule<GraphDef, OpDef, NodeDef, OpDef.AttrDef, AttrValue, TensorProto, DataType>(mappingNamesToPerform = mappingNamesToPerform, transformerArgs = transformerArgs) {



    override fun createTensorProto(input: TensorProto): TensorNamespace.TensorProto {
        return TensorflowIRTensor(input).toArgTensor()
    }


    override fun isInputTensorName(inputName: String): Boolean {
        val tfOp = tensorflowOps.findOp(mappingProcess!!.inputFrameworkOpName())
        return tfOp.inputArgList.map { input -> input.name }.contains(inputName)
    }

    override fun isOutputTensorName(outputName: String): Boolean {
        val nd4jOpDescriptor = nd4jOpDescriptors.findOp(mappingProcess!!.opName())
        return nd4jOpDescriptor.argDescriptorList.filter { inputDescriptor -> inputDescriptor.argType == OpNamespace.ArgDescriptor.ArgType.INPUT_TENSOR }
            .map {inputDescriptor -> inputDescriptor.name }.contains(outputName)
    }
}

fun mappingListNDArrays(inputs: MutableMap<String, String>) : TensorflowMultiInputIndexMappingRule {
    return TensorflowMultiInputIndexMappingRule(
        mappingNamesToPerform = inputs
    )
}