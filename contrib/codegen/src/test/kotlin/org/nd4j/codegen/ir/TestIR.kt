package org.nd4j.codegen.ir

import com.google.common.reflect.TypeToken
import org.junit.jupiter.api.Test
import kotlin.test.assertTrue
import org.apache.commons.lang3.reflect.TypeUtils
import org.nd4j.codegen.ir.registry.OpMappingRegistry
import org.tensorflow.framework.*


class TestIR {
    @Test
    fun testLoadOpDescriptors() {
        val outputType = TypeUtils.parameterize(OpMappingRegistry::class.java,GraphDef::class.java, NodeDef::class.java, OpDef::class.java,
            TensorProto::class.java,DataType::class.java, OpDef.AttrDef::class.java,AttrValue::class.java)
        val rawType = TypeToken.of(outputType).rawType
        println(rawType)
        val createdRegistry = rawType.getConstructor(String::class.java).newInstance("tensorflow")
        println(createdRegistry)
    }
}

