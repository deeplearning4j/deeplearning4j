/*
 *  ******************************************************************************
 *  *
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
 *  *  See the NOTICE file distributed with this work for additional
 *  *  information regarding copyright ownership.
 *  * Unless required by applicable law or agreed to in writing, software
 *  * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 *  * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 *  * License for the specific language governing permissions and limitations
 *  * under the License.
 *  *
 *  * SPDX-License-Identifier: Apache-2.0
 *  *****************************************************************************
 */

package org.nd4j.samediff.frameworkimport.onnx.definitions

import onnx.Onnx
import org.nd4j.ir.OpNamespace
import org.nd4j.samediff.frameworkimport.ArgDescriptor
import org.nd4j.samediff.frameworkimport.onnx.*
import org.nd4j.samediff.frameworkimport.onnx.process.OnnxMappingProcess
import org.nd4j.samediff.frameworkimport.onnx.rule.tensor.NDArrayMappingRule
import org.nd4j.samediff.frameworkimport.opdefs.OpDescriptorLoaderHolder
import org.nd4j.samediff.frameworkimport.registry.OpMappingRegistry
import org.nd4j.samediff.frameworkimport.registry.OpRegistryHolder

/**
 * Microsoft ONNX Extensions Registry
 * Handles registration and processing of Microsoft-specific ONNX operators
 * that are not part of the standard ONNX specification.
 * 
 * Reference: https://github.com/microsoft/onnxruntime/blob/main/docs/ContribOperators.md
 */
object MicrosoftOnnxExtensions {
    
    // Registry of Microsoft extension operators with their domains
    private val microsoftExtensionOps = mapOf(
        // Attention and Transformer Operations
        "EmbedLayerNormalization" to "com.microsoft",
        "Attention" to "com.microsoft", 
        "MultiHeadAttention" to "com.microsoft",
        "SkipLayerNormalization" to "com.microsoft",
        "LayerNormalization" to "com.microsoft",
        "SimplifiedLayerNormalization" to "com.microsoft",
        "PackedAttention" to "com.microsoft",
        "PackedMultiHeadAttention" to "com.microsoft",
        
        // Activation Functions
        "FastGelu" to "com.microsoft",
        "Gelu" to "com.microsoft",
        "BiasGelu" to "com.microsoft",
        "QuickGelu" to "com.microsoft",
        "BiasSoftmax" to "com.microsoft",
        
        // Utility Operations
        "BiasAdd" to "com.microsoft",
        "RelativePositionBias" to "com.microsoft",
        "RemovePadding" to "com.microsoft",
        "RestorePadding" to "com.microsoft",
        "FusedConv" to "com.microsoft",
        
        // Quantization Operations
        "QLinearAdd" to "com.microsoft",
        "QLinearMul" to "com.microsoft",
        "DynamicQuantizeMatMul" to "com.microsoft",
        "MatMulInteger16" to "com.microsoft",
        
        // Beam Search Operations
        "BeamSearch" to "com.microsoft",
        "GreedySearch" to "com.microsoft",
        
        // Miscellaneous
        "Crop" to "com.microsoft",
        "ImageScaler" to "com.microsoft",
        "ParametricSoftplus" to "com.microsoft",
        "ScaledTanh" to "com.microsoft",
        "ThresholdedRelu" to "com.microsoft"
    )
    
    /**
     * Check if an operation name is a Microsoft extension
     */
    fun isMicrosoftExtension(opName: String): Boolean {
        return microsoftExtensionOps.containsKey(opName)
    }
    
    /**
     * Get the domain for a Microsoft extension operation
     */
    fun getDomainForOp(opName: String): String? {
        return microsoftExtensionOps[opName]
    }
    
    /**
     * Create a dummy op descriptor for Microsoft extensions that don't have
     * standard ONNX definitions. This allows the import process to handle them.
     */
    fun createExtensionOpDescriptor(opName: String, domain: String = "com.microsoft"): Onnx.NodeProto {
        return Onnx.NodeProto.newBuilder()
            .setOpType(opName)
            .setDomain(domain)
            .setName("${opName}_extension")
            .addAllInput(emptyList()) // Will be populated dynamically during import
            .addAllOutput(emptyList()) // Will be populated dynamically during import
            .build()
    }
    
    /**
     * Register Microsoft extension operations with the ONNX registry
     */
    fun registerMicrosoftExtensions(registry: OpMappingRegistry<Onnx.GraphProto, Onnx.NodeProto, Onnx.NodeProto, Onnx.TensorProto, Onnx.TensorProto.DataType, Onnx.AttributeProto, Onnx.AttributeProto>) {
        
        // Register mapping processes for Microsoft extensions
        registerExtensionMappings(registry)
        
        // Register op descriptors for extensions that don't have standard equivalents
        registerExtensionOpDescriptors(registry)
    }
    
    private fun registerExtensionMappings(registry: OpMappingRegistry<Onnx.GraphProto, Onnx.NodeProto, Onnx.NodeProto, Onnx.TensorProto, Onnx.TensorProto.DataType, Onnx.AttributeProto, Onnx.AttributeProto>) {
        
        // FastGelu mapping
        val fastGelu = OnnxMappingProcess(
            inputFrameworkOpName = "FastGelu",
            opName = "noop", // Handled by PreImportHook
            opMappingRegistry = registry,
            tensorMappingRules = listOf(),
            attributeMappingRules = listOf()
        )
        
        // Gelu mapping  
        val gelu = OnnxMappingProcess(
            inputFrameworkOpName = "Gelu",
            opName = "gelu",
            opMappingRegistry = registry,
            tensorMappingRules = listOf(mappingNDArrayInputs(mutableMapOf("input" to "X"))),
            attributeMappingRules = booleanConstant(inputName = "inPlace", constantValue = false, argumentIndex = 0)
        )
        
        // BiasGelu mapping
        val biasGelu = OnnxMappingProcess(
            inputFrameworkOpName = "BiasGelu",
            opName = "noop", // Handled by PreImportHook
            opMappingRegistry = registry,
            tensorMappingRules = listOf(),
            attributeMappingRules = listOf()
        )
        
        // QuickGelu mapping
        val quickGelu = OnnxMappingProcess(
            inputFrameworkOpName = "QuickGelu",
            opName = "noop", // Handled by PreImportHook
            opMappingRegistry = registry,
            tensorMappingRules = listOf(),
            attributeMappingRules = listOf()
        )
        
        // SkipLayerNormalization mapping
        val skipLayerNorm = OnnxMappingProcess(
            inputFrameworkOpName = "SkipLayerNormalization",
            opName = "noop", // Handled by PreImportHook
            opMappingRegistry = registry,
            tensorMappingRules = listOf(),
            attributeMappingRules = listOf()
        )
        
        // Attention mapping
        val attention = OnnxMappingProcess(
            inputFrameworkOpName = "Attention",
            opName = "noop", // Handled by PreImportHook
            opMappingRegistry = registry,
            tensorMappingRules = listOf(),
            attributeMappingRules = listOf()
        )
        
        // MultiHeadAttention mapping
        val multiHeadAttention = OnnxMappingProcess(
            inputFrameworkOpName = "MultiHeadAttention",
            opName = "noop", // Handled by PreImportHook
            opMappingRegistry = registry,
            tensorMappingRules = listOf(),
            attributeMappingRules = listOf()
        )
        
        // BiasAdd mapping
        val biasAdd = OnnxMappingProcess(
            inputFrameworkOpName = "BiasAdd",
            opName = "noop", // Handled by PreImportHook
            opMappingRegistry = registry,
            tensorMappingRules = listOf(),
            attributeMappingRules = listOf()
        )
        
        // LayerNormalization mapping (Microsoft version)
        val layerNormalization = OnnxMappingProcess(
            inputFrameworkOpName = "LayerNormalization",
            opName = "noop", // Handled by PreImportHook  
            opMappingRegistry = registry,
            tensorMappingRules = listOf(),
            attributeMappingRules = listOf()
        )
    }
    
    private fun registerExtensionOpDescriptors(registry: OpMappingRegistry<Onnx.GraphProto, Onnx.NodeProto, Onnx.NodeProto, Onnx.TensorProto, Onnx.TensorProto.DataType, Onnx.AttributeProto, Onnx.AttributeProto>) {
        microsoftExtensionOps.forEach { (opName, domain) ->
            val opDescriptor = createExtensionOpDescriptor(opName, domain)
            registry.registerInputFrameworkOpDef(opName, opDescriptor)
        }
    }
    
    /**
     * Get all registered Microsoft extension operation names
     */
    fun getAllExtensionOpNames(): Set<String> {
        return microsoftExtensionOps.keys
    }
    
    /**
     * Get all registered Microsoft extension domains
     */
    fun getAllExtensionDomains(): Set<String> {
        return microsoftExtensionOps.values.toSet()
    }
}

// Extension function to check if a node uses Microsoft extensions
fun Onnx.NodeProto.isMicrosoftExtension(): Boolean {
    return MicrosoftOnnxExtensions.isMicrosoftExtension(this.opType) ||
           this.domain == "com.microsoft"
}

// Extension function to get the Microsoft domain for a node
fun Onnx.NodeProto.getMicrosoftDomain(): String? {
    return if (this.isMicrosoftExtension()) {
        if (this.domain.isNotEmpty()) this.domain else MicrosoftOnnxExtensions.getDomainForOp(this.opType)
    } else null
}