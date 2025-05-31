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

package org.eclipse.deeplearning4j.contrib

import org.nd4j.samediff.frameworkimport.tensorflow.definitions.registry as tensorflowRegistry
import org.nd4j.samediff.frameworkimport.onnx.definitions.registry as onnxRegistry
import org.nd4j.samediff.frameworkimport.tensorflow.process.TensorflowMappingProcessLoader
import org.nd4j.samediff.frameworkimport.onnx.process.OnnxMappingProcessLoader
import org.nd4j.samediff.frameworkimport.opdefs.OpDescriptorLoaderHolder
import org.nd4j.samediff.frameworkimport.registry.OpMappingRegistry
import org.nd4j.samediff.frameworkimport.reflect.ImportReflectionCache
import org.nd4j.samediff.frameworkimport.hooks.PreImportHook
import org.nd4j.samediff.frameworkimport.hooks.annotations.PreHookRule
import org.tensorflow.framework.*
import onnx.Onnx
import kotlin.system.exitProcess

/**
 * Standalone tool for updating framework import configuration files.
 * This replaces the need to run individual test methods for updating op configurations.
 * 
 * Usage:
 * java -jar op-registry-updater.jar [--framework=<framework>] [--validate-only] [--debug] [--help]
 * 
 * Or via Maven:
 * mvn exec:java -Dexec.args="--framework=all"
 * 
 * Options:
 * --framework=<framework>  Update specific framework only (tensorflow|onnx|all). Default: all
 * --validate-only          Only validate existing configs without saving. Default: false
 * --debug                  Enable debug output for PreImportHook discovery. Default: false
 * --help                   Show this help message
 */
object OpRegistryUpdater {

    private const val DEFAULT_FRAMEWORK = "all"
    private val SUPPORTED_FRAMEWORKS = setOf("tensorflow", "onnx", "all")

    @JvmStatic
    fun main(args: Array<String>) {
        val options = parseArguments(args)
        
        if (options.showHelp) {
            showHelp()
            return
        }

        try {
            println("=======================================================")
            println("OP Registry Updater - Standalone Tool")
            println("=======================================================")
            println("Framework: ${options.framework}")
            println("Validate only: ${options.validateOnly}")
            println("Debug mode: ${options.debug}")
            println()

            if (options.debug) {
                debugPreImportHooks()
            }

            when (options.framework.lowercase()) {
                "tensorflow" -> updateTensorflowRegistry(options.validateOnly, options.debug)
                "onnx" -> updateOnnxRegistry(options.validateOnly, options.debug)
                "all" -> {
                    updateTensorflowRegistry(options.validateOnly, options.debug)
                    println()
                    updateOnnxRegistry(options.validateOnly, options.debug)
                }
                else -> {
                    System.err.println("Unsupported framework: ${options.framework}")
                    exitProcess(1)
                }
            }

            println()
            println("=======================================================")
            println("OP Registry Update Process completed successfully!")
            println("=======================================================")

        } catch (e: Exception) {
            System.err.println("Error during OP Registry Update: ${e.message}")
            e.printStackTrace()
            exitProcess(1)
        }
    }

    private fun debugPreImportHooks() {
        println("🔍 Debug: Checking PreImportHook discovery...")
        println("-------------------------------------------------------")
        
        // Force cache reload if needed
        try {
            ImportReflectionCache.load()
        } catch (e: Exception) {
            println("  ⚠ Warning: Error loading ImportReflectionCache: ${e.message}")
            e.printStackTrace()
        }
        
        // Check what's in the cache
        val allPreHooks = ImportReflectionCache.preProcessRuleImplementationsByOp
        println("  Total PreImportHook entries in cache: ${allPreHooks.size()}")
        
        // Show all frameworks found
        val allFrameworks = allPreHooks.rowKeySet()
        println("  Frameworks found in cache: ${allFrameworks}")
        
        allFrameworks.forEach { framework ->
            val frameworkHooks = allPreHooks.row(framework)
            println("  $framework framework hooks: ${frameworkHooks.size}")
            frameworkHooks.forEach { (opName, hooks) ->
                println("    Op: $opName -> ${hooks.size} hooks")
                hooks.forEach { hook ->
                    println("      Hook class: ${hook.javaClass.name}")
                }
            }
        }
        
        // Try to manually scan for ONNX PreImportHook classes
        println("  🔍 Manual scan for ONNX PreImportHook classes...")
        val expectedOnnxHooks = listOf(
            "org.nd4j.samediff.frameworkimport.onnx.definitions.implementations.EmbedLayerNormalization",
            "org.nd4j.samediff.frameworkimport.onnx.definitions.implementations.BatchNormalization",
            "org.nd4j.samediff.frameworkimport.onnx.definitions.implementations.Gemm",
            "org.nd4j.samediff.frameworkimport.onnx.definitions.implementations.GlobalAveragePooling",
            "org.nd4j.samediff.frameworkimport.onnx.definitions.implementations.GlobalMaxPooling",
            "org.nd4j.samediff.frameworkimport.onnx.definitions.implementations.PRelu",
            "org.nd4j.samediff.frameworkimport.onnx.definitions.implementations.Reshape",
            "org.nd4j.samediff.frameworkimport.onnx.definitions.implementations.Transpose",
            "org.nd4j.samediff.frameworkimport.onnx.definitions.implementations.Unsqueeze",
            "org.nd4j.samediff.frameworkimport.onnx.definitions.implementations.Slice",
            "org.nd4j.samediff.frameworkimport.onnx.definitions.implementations.Split",
            "org.nd4j.samediff.frameworkimport.onnx.definitions.implementations.Constant",
            "org.nd4j.samediff.frameworkimport.onnx.definitions.implementations.ConstantOfShape",
            "org.nd4j.samediff.frameworkimport.onnx.definitions.implementations.Cast",
            "org.nd4j.samediff.frameworkimport.onnx.definitions.implementations.Clip",
            "org.nd4j.samediff.frameworkimport.onnx.definitions.implementations.Dropout",
            "org.nd4j.samediff.frameworkimport.onnx.definitions.implementations.Expand",
            "org.nd4j.samediff.frameworkimport.onnx.definitions.implementations.Minimum",
            "org.nd4j.samediff.frameworkimport.onnx.definitions.implementations.Maximum",
            "org.nd4j.samediff.frameworkimport.onnx.definitions.implementations.Resize",
            "org.nd4j.samediff.frameworkimport.onnx.definitions.implementations.NonZero"
        )
        
        var foundHooks = 0
        var notFoundHooks = 0
        
        expectedOnnxHooks.forEach { className ->
            try {
                val clazz = Class.forName(className)
                val annotation = clazz.getAnnotation(PreHookRule::class.java)
                if (annotation != null) {
                    foundHooks++
                    val opNames = annotation.opNames
                    println("    ✓ Found: ${clazz.simpleName} -> ops: ${opNames.contentToString()}")
                } else {
                    println("    ❌ No @PreHookRule: ${clazz.simpleName}")
                }
            } catch (e: ClassNotFoundException) {
                notFoundHooks++
                println("    ❌ Class not found: ${className.substringAfterLast('.')}")
            } catch (e: Exception) {
                println("    ❌ Error loading: ${className.substringAfterLast('.')}: ${e.message}")
            }
        }
        
        println("  📊 Manual scan results: $foundHooks found, $notFoundHooks not found")
        println()
    }

    private fun updateTensorflowRegistry(validateOnly: Boolean, debug: Boolean) {
        println("Processing TensorFlow Registry...")
        println("-------------------------------------------------------")
        
        try {
            val tensorflowOpMappingRegistry = OpMappingRegistry<GraphDef, NodeDef, OpDef, TensorProto, DataType, OpDef.AttrDef, AttrValue>(
                "tensorflow", OpDescriptorLoaderHolder.nd4jOpDescriptor
            )
            
            val loader = TensorflowMappingProcessLoader(tensorflowOpMappingRegistry)
            
            // Validate existing mappings
            val frameworkOpNames = tensorflowRegistry().inputFrameworkOpNames()
            println("  Found ${frameworkOpNames.size} TensorFlow ops to process")
            
            var validatedCount = 0
            var errorCount = 0
            var noopCount = 0
            var preHookCount = 0
            
            frameworkOpNames.forEach { name ->
                try {
                    if (tensorflowRegistry().hasMappingOpProcess(name)) {
                        val process = tensorflowRegistry().lookupOpMappingProcess(name)
                        
                        // Check if this is a noop (which indicates PreImportHook usage)
                        if (process.opName() == "noop") {
                            noopCount++
                            
                            // Check if there are actual PreImportHooks for this op
                            val preHooks = ImportReflectionCache.preProcessRuleImplementationsByOp.get("tensorflow", name)
                            if (preHooks != null && preHooks.isNotEmpty()) {
                                preHookCount++
                                if (debug) {
                                    println("  📝 TensorFlow op '$name' mapped to noop with ${preHooks.size} PreImportHook(s)")
                                    preHooks.forEach { hook ->
                                        println("    Hook: ${hook.javaClass.simpleName}")
                                    }
                                }
                            } else if (debug) {
                                println("  ⚠ TensorFlow op '$name' mapped to noop but no PreImportHooks found!")
                            }
                        } else if (debug && process.opName() != name) {
                            println("  🔄 TensorFlow op '$name' -> nd4j op '${process.opName()}'")
                        }
                        
                        val serialized = process.serialize()
                        val created = loader.createProcess(serialized)
                        
                        if (process != created) {
                            System.err.println("  WARNING: Validation failed for TensorFlow op '$name'")
                            if (debug) {
                                System.err.println("    Original tensor rules: ${process.tensorMappingRules()}")
                                System.err.println("    Created tensor rules: ${created.tensorMappingRules()}")
                                System.err.println("    Original attribute rules: ${process.attributeMappingRules()}")
                                System.err.println("    Created attribute rules: ${created.attributeMappingRules()}")
                            }
                            errorCount++
                        } else {
                            validatedCount++
                        }
                    } else if (debug) {
                        // Check if this op has PreImportHook rules
                        val preHooks = ImportReflectionCache.preProcessRuleImplementationsByOp.get("tensorflow", name)
                        if (preHooks != null && preHooks.isNotEmpty()) {
                            println("  📝 TensorFlow op '$name' has ${preHooks.size} PreImportHook(s) but no mapping process")
                        }
                    }
                } catch (e: Exception) {
                    System.err.println("  ERROR: Failed to process TensorFlow op '$name': ${e.message}")
                    errorCount++
                }
            }
            
            println("  ✓ Validated: $validatedCount ops")
            println("  📝 Noop mappings: $noopCount ops")
            println("  🔧 PreImportHooks: $preHookCount ops")
            if (errorCount > 0) {
                println("  ⚠ Errors: $errorCount ops")
            }
            
            if (!validateOnly) {
                println("  💾 Saving TensorFlow processes and rule set...")
                tensorflowRegistry().saveProcessesAndRuleSet()
                println("  ✓ TensorFlow registry saved successfully")
            } else {
                println("  ℹ Validation only - no files saved")
            }
            
        } catch (e: Exception) {
            throw RuntimeException("Failed to update TensorFlow registry", e)
        }
    }

    private fun updateOnnxRegistry(validateOnly: Boolean, debug: Boolean) {
        println("Processing ONNX Registry...")
        println("-------------------------------------------------------")
        
        try {
            val onnxOpMappingRegistry = OpMappingRegistry<Onnx.GraphProto, Onnx.NodeProto,
                    Onnx.NodeProto, Onnx.TensorProto,
                    Onnx.TensorProto.DataType, Onnx.AttributeProto, Onnx.AttributeProto>(
                "onnx", OpDescriptorLoaderHolder.nd4jOpDescriptor
            )
            
            val loader = OnnxMappingProcessLoader(onnxOpMappingRegistry)
            
            // Validate existing mappings
            val frameworkOpNames = onnxRegistry().inputFrameworkOpNames()
            println("  Found ${frameworkOpNames.size} ONNX ops to process")
            
            var validatedCount = 0
            var errorCount = 0
            var noopCount = 0
            var preHookCount = 0
            val noopOpsWithoutHooks = mutableListOf<String>()
            val noopOpsWithHooks = mutableListOf<String>()
            
            // Special debug for EmbedLayerNormalization
            if (debug) {
                println("  🔍 Special debug for EmbedLayerNormalization:")
                val embedHooks = ImportReflectionCache.preProcessRuleImplementationsByOp.get("onnx", "EmbedLayerNormalization")
                println("    Cache lookup result: ${embedHooks?.size ?: 0} hooks")
                if (embedHooks != null) {
                    embedHooks.forEach { hook ->
                        println("    Hook instance: ${hook.javaClass.name} @ ${System.identityHashCode(hook)}")
                    }
                }
                
                // Check if registry has the mapping
                val hasMapping = onnxRegistry().hasMappingOpProcess("EmbedLayerNormalization")
                println("    Registry has mapping: $hasMapping")
                if (hasMapping) {
                    val process = onnxRegistry().lookupOpMappingProcess("EmbedLayerNormalization")
                    println("    Process op name: ${process.opName()}")
                    println("    Process input framework name: ${process.inputFrameworkOpName()}")
                }
                println()
            }
            
            frameworkOpNames.forEach { name ->
                try {
                    if (onnxRegistry().hasMappingOpProcess(name)) {
                        val process = onnxRegistry().lookupOpMappingProcess(name)
                        
                        // Check if this is a noop (which indicates PreImportHook usage)
                        if (process.opName() == "noop") {
                            noopCount++
                            
                            // Special debug for EmbedLayerNormalization
                            if (debug && name == "EmbedLayerNormalization") {
                                println("  🔍 Processing EmbedLayerNormalization:")
                                println("    Op name: $name")
                                println("    Process op name: ${process.opName()}")
                                
                                // Check cache lookup
                                val preHooks = ImportReflectionCache.preProcessRuleImplementationsByOp.get("onnx", name)
                                println("    Cache lookup result: ${preHooks?.size ?: 0} hooks")
                                if (preHooks != null) {
                                    preHooks.forEach { hook ->
                                        println("    Hook: ${hook.javaClass.name}")
                                    }
                                } else {
                                    println("    ❌ No hooks found in cache for $name")
                                    
                                    // Debug cache contents
                                    val allOnnxOps = ImportReflectionCache.preProcessRuleImplementationsByOp.row("onnx")
                                    println("    Available ops in cache: ${allOnnxOps.keys}")
                                    
                                    // Check if there's a case sensitivity issue
                                    allOnnxOps.keys.forEach { cacheOpName ->
                                        if (cacheOpName.equals(name, ignoreCase = true)) {
                                            println("    Found case-insensitive match: '$cacheOpName' vs '$name'")
                                        }
                                    }
                                }
                            }
                            
                            // Check if there are actual PreImportHooks for this op
                            val preHooks = ImportReflectionCache.preProcessRuleImplementationsByOp.get("onnx", name)
                            if (preHooks != null && preHooks.isNotEmpty()) {
                                preHookCount++
                                noopOpsWithHooks.add(name)
                                if (debug && name != "EmbedLayerNormalization") {
                                    println("  📝 ONNX op '$name' mapped to noop with ${preHooks.size} PreImportHook(s)")
                                    preHooks.forEach { hook ->
                                        println("    Hook: ${hook.javaClass.simpleName}")
                                    }
                                }
                            } else {
                                noopOpsWithoutHooks.add(name)
                                if (debug) {
                                    println("  ⚠ ONNX op '$name' mapped to noop but no PreImportHooks found!")
                                }
                            }
                        } else if (debug && process.opName() != name) {
                            println("  🔄 ONNX op '$name' -> nd4j op '${process.opName()}'")
                        }
                        
                        val serialized = process.serialize()
                        val created = loader.createProcess(serialized)
                        
                        if (process != created) {
                            System.err.println("  WARNING: Validation failed for ONNX op '$name'")
                            if (debug) {
                                System.err.println("    Original tensor rules: ${process.tensorMappingRules()}")
                                System.err.println("    Created tensor rules: ${created.tensorMappingRules()}")
                                System.err.println("    Original attribute rules: ${process.attributeMappingRules()}")
                                System.err.println("    Created attribute rules: ${created.attributeMappingRules()}")
                            }
                            errorCount++
                        } else {
                            validatedCount++
                        }
                    } else if (debug) {
                        // Check if this op has PreImportHook rules
                        val preHooks = ImportReflectionCache.preProcessRuleImplementationsByOp.get("onnx", name)
                        if (preHooks != null && preHooks.isNotEmpty()) {
                            println("  📝 ONNX op '$name' has ${preHooks.size} PreImportHook(s) but no mapping process")
                        }
                    }
                } catch (e: Exception) {
                    System.err.println("  ERROR: Failed to process ONNX op '$name': ${e.message}")
                    if (debug && name == "EmbedLayerNormalization") {
                        e.printStackTrace()
                    }
                    errorCount++
                }
            }
            
            println("  ✓ Validated: $validatedCount ops")
            println("  📝 Noop mappings: $noopCount ops")
            println("  🔧 PreImportHooks: $preHookCount ops")
            if (errorCount > 0) {
                println("  ⚠ Errors: $errorCount ops")
            }
            
            // Show specific noop analysis
            if (noopOpsWithoutHooks.isNotEmpty()) {
                println("  ❌ Noop ops WITHOUT PreImportHooks (${noopOpsWithoutHooks.size}):")
                noopOpsWithoutHooks.sorted().forEach { opName ->
                    println("    - $opName")
                }
            }
            
            if (noopOpsWithHooks.isNotEmpty() && debug) {
                println("  ✅ Noop ops WITH PreImportHooks (${noopOpsWithHooks.size}):")
                noopOpsWithHooks.sorted().forEach { opName ->
                    println("    - $opName")
                }
            }
            
            if (!validateOnly) {
                println("  💾 Saving ONNX processes and rule set...")
                onnxRegistry().saveProcessesAndRuleSet()
                
                // Additional step: reload from file to verify it works
                println("  🔍 Verifying ONNX processes can be loaded from file...")
                onnxRegistry().loadFromFile("onnx-processes.pbtxt", loader)
                println("  ✓ ONNX registry saved and verified successfully")
            } else {
                println("  ℹ Validation only - no files saved")
            }
            
        } catch (e: Exception) {
            throw RuntimeException("Failed to update ONNX registry", e)
        }
    }

    private fun parseArguments(args: Array<String>): CommandLineOptions {
        var framework = DEFAULT_FRAMEWORK
        var validateOnly = false
        var debug = false
        var showHelp = false

        for (arg in args) {
            when {
                arg.startsWith("--framework=") -> {
                    framework = arg.substringAfter("=")
                    if (framework !in SUPPORTED_FRAMEWORKS) {
                        System.err.println("Error: Unsupported framework '$framework'. Supported: ${SUPPORTED_FRAMEWORKS.joinToString(", ")}")
                        exitProcess(1)
                    }
                }
                arg == "--validate-only" -> validateOnly = true
                arg == "--debug" -> debug = true
                arg == "--help" || arg == "-h" -> showHelp = true
                else -> {
                    System.err.println("Error: Unknown argument '$arg'")
                    showHelp = true
                }
            }
        }

        return CommandLineOptions(framework, validateOnly, debug, showHelp)
    }

    private fun showHelp() {
        println("""
            |=======================================================
            |OP Registry Updater - Standalone Tool
            |=======================================================
            |
            |Updates the available ops for import configuration files for framework import.
            |This standalone tool replaces the need to run individual test methods for updating
            |op configurations, and eliminates dependencies on the platform-tests module.
            |
            |Usage:
            |  java -jar op-registry-updater.jar [options]
            |  
            |  Or via Maven:
            |  mvn compile exec:java -Dexec.args="[options]"
            |
            |Options:
            |  --framework=<framework>  Update specific framework only
            |                             Supported: tensorflow, onnx, all (default: all)
            |
            |  --validate-only            Only validate existing configurations without saving
            |                             Useful for testing and verification (default: false)
            |
            |  --debug                    Enable debug output for PreImportHook discovery
            |                             Shows detailed information about hook scanning (default: false)
            |
            |  --help, -h                 Show this help message
            |
            |Examples:
            |  java -jar op-registry-updater.jar --debug
            |      Update all registries with debug output to check hook discovery
            |
            |  java -jar op-registry-updater.jar --framework=onnx --debug --validate-only
            |      Debug ONNX registry and check for missing PreImportHooks
            |
            |  mvn compile exec:java -Dexec.args="--framework=onnx --debug"
            |      Run via Maven to debug ONNX registry issues
            |
            |Requirements:
            |  - Java 11+
            |  - Access to framework import dependencies (tensorflow-java, onnx-java, etc.)
            |  - Write permissions to the target resource directories
            |
            |Output Files:
            |  The tool generates configuration files in the appropriate framework import modules:
            |  - TensorFlow: nd4j/samediff-import/samediff-import-tensorflow/src/main/resources/
            |  - ONNX: nd4j/samediff-import/samediff-import-onnx/src/main/resources/
            |
            |Note: This standalone tool requires the appropriate framework dependencies to be
            |      available in the classpath (tensorflow-java, onnx-java, etc.)
            |=======================================================
        """.trimMargin())
    }

    private data class CommandLineOptions(
        val framework: String,
        val validateOnly: Boolean,
        val debug: Boolean,
        val showHelp: Boolean
    )
}
