package org.nd4j.codegen.ir.tensorflow

import org.nd4j.codegen.ir.AbstractMappingProcess
import org.nd4j.codegen.ir.AttributeMappingRule
import org.nd4j.codegen.ir.AttributeValueType
import org.nd4j.codegen.ir.TensorMappingRule
import org.nd4j.codegen.ir.registry.OpMappingRegistry
import org.nd4j.codegen.ir.registry.OpRegistryHolder
import org.nd4j.common.base.Preconditions
import org.tensorflow.framework.*

open class TensorflowMappingProcess(inputFramework: String = "tensorflow",
                                    frameworkVersion: String = "2.3",
                                    inputFrameworkOpName: String,
                                    opName: String,
                                    opMappingRegistry: OpMappingRegistry<GraphDef,
                                            NodeDef,OpDef,
                                            TensorProto,DataType, OpDef.AttrDef,AttrValue>,
                                    tensorMappingRules: List<TensorMappingRule<GraphDef,
                                            OpDef, NodeDef,
                                            OpDef.AttrDef,
                                            AttrValue, TensorProto, DataType>> = emptyList(),
                                    attributeMappingRules: List<AttributeMappingRule<GraphDef,
                                            OpDef, NodeDef,
                                            OpDef.AttrDef,
                                            AttrValue,
                                            TensorProto, DataType>> = emptyList(),
                                    inputIndexOverrides: Map<Int,Int> = emptyMap())
    : AbstractMappingProcess<GraphDef,OpDef, NodeDef, TensorProto, OpDef.AttrDef,
        AttrValue, DataType>(
    inputFramework,
    frameworkVersion,
    inputFrameworkOpName,
    inputIndexOverrides,
    opName,
    opMappingRegistry,
    tensorMappingRules,
    attributeMappingRules) {
    override fun inputOpDefValueTypes(): Map<String, AttributeValueType> {
        Preconditions.checkNotNull(inputFrameworkOpName,"No input framework op def name found!")
        val opDef = opMappingRegistry.lookupInputFrameworkOpDef(inputFrameworkOpName)
        val retMap = HashMap<String,AttributeValueType>()
        opDef.attrList.forEach { attrDef ->
            retMap[attrDef.name] = attributeValueTypeForTensorflowAttribute(attrDef)
        }

        return retMap

    }

}


