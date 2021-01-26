package org.nd4j.codegen.ir

import com.google.common.reflect.TypeToken
import org.apache.commons.lang3.reflect.TypeUtils
import org.nd4j.autodiff.samediff.SameDiff
import org.nd4j.codegen.ir.registry.OpMappingRegistry
import org.tensorflow.framework.*
import kotlin.reflect.KClass

class FrameworkImporter(inputFrameworkName: String,nodeType: String,graphType: String,opDefType: String,tensorType: String,dataType: String,attributeType: String,attributeValueType: String) {

    val graphType = graphType
    val nodeType = nodeType
    val tensorType = tensorType
    val dataType = dataType
    val attributeType = attributeType
    val attributeValueType = attributeValueType
    val opDefType = opDefType
    val inputFrameworkName = inputFrameworkName

    fun runImport(): SameDiff {
        val outputType = TypeUtils.parameterize(
            OpMappingRegistry::class.java, Class.forName(graphType), Class.forName(nodeType), Class.forName(opDefType),
            Class.forName(tensorType), Class.forName(dataType), Class.forName(attributeType), Class.forName(attributeValueType))

        val importGraphParameterized = TypeUtils.parameterize(ImportGraph::class.java,
            Class.forName(graphType),
            Class.forName(nodeType),
            Class.forName(opDefType),
            Class.forName(tensorType),
            Class.forName(attributeType),
            Class.forName(attributeValueType),
            Class.forName(dataType))

        val rawRegistryType = TypeToken.of(outputType).rawType
        val rawRegistryInstance = rawRegistryType.getConstructor(String::class.java).newInstance(inputFrameworkName)
        val importGraphType = TypeToken.of(importGraphParameterized).rawType
        val importGraphInstance = importGraphType.getConstructor().newInstance()
        return SameDiff.create()
    }


}