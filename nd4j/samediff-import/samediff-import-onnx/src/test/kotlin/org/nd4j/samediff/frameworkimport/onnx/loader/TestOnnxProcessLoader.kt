package org.nd4j.samediff.frameworkimport.onnx.loader

import junit.framework.Assert
import onnx.Onnx
import org.junit.jupiter.api.Test
import org.nd4j.samediff.frameworkimport.onnx.definitions.registry
import org.nd4j.samediff.frameworkimport.onnx.process.OnnxMappingProcessLoader
import org.nd4j.samediff.frameworkimport.opdefs.OpDescriptorLoaderHolder
import org.nd4j.samediff.frameworkimport.registry.OpMappingRegistry

class TestOnnxProcessLoader {

    @Test
    fun testLoader() {
        val onnxOpMappingRegistry = OpMappingRegistry<Onnx.GraphProto, Onnx.NodeProto,
                Onnx.NodeProto, Onnx.TensorProto,
                Onnx.TensorProto.DataType, Onnx.AttributeProto, Onnx.AttributeProto>(
            "onnx", OpDescriptorLoaderHolder.nd4jOpDescriptor)

        val loader = OnnxMappingProcessLoader(onnxOpMappingRegistry)
        println(loader)
        registry().inputFrameworkOpNames().forEach { name ->
            if(registry().hasMappingOpProcess(name)) {
                val process = registry().lookupOpMappingProcess(name)
                val serialized = process.serialize()
                val created = loader.createProcess(serialized)
                Assert.assertEquals(
                    "Op name $name failed with process tensor rules ${process.tensorMappingRules()} and created tensor rules ${created.tensorMappingRules()} with attributes ${process.attributeMappingRules()} and created attribute rules ${created.attributeMappingRules()}",
                    process,
                    created
                )
            }

        }
    }

    @Test
    fun saveTest() {
        registry().saveProcessesAndRuleSet()
    }
}