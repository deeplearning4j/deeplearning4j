package org.nd4j.samediff.frameworkimport.tensorflow.loader

import junit.framework.Assert.assertEquals
import org.junit.Test
import org.nd4j.samediff.frameworkimport.opdefs.OpDescriptorLoaderHolder
import org.nd4j.samediff.frameworkimport.registry.OpMappingRegistry
import org.nd4j.samediff.frameworkimport.tensorflow.definitions.registry
import org.nd4j.samediff.frameworkimport.tensorflow.process.TensorflowMappingProcessLoader
import org.tensorflow.framework.*

class TestTensorflowProcessLoader {

    @Test
    fun testLoader() {
        val tensorflowOpMappingRegistry = OpMappingRegistry<GraphDef, NodeDef, OpDef, TensorProto, DataType, OpDef.AttrDef, AttrValue>(
           "tensorflow", OpDescriptorLoaderHolder.nd4jOpDescriptor)

        val loader = TensorflowMappingProcessLoader(tensorflowOpMappingRegistry)
        println(loader)
        registry().inputFrameworkOpNames().forEach { name ->
            if(registry().hasMappingOpProcess(name)) {
                val process = registry().lookupOpMappingProcess(name)
                val serialized = process.serialize()
                val created = loader.createProcess(serialized)
                assertEquals("Op name $name failed with process tensor rules ${process.tensorMappingRules()} and created tensor rules ${created.tensorMappingRules()} with attributes ${process.attributeMappingRules()} and created attribute rules ${created.attributeMappingRules()}",process,created)
            }

        }

    }

    @Test
    fun saveTest() {
        registry().saveProcessesAndRuleSet()
    }

}