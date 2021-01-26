package org.nd4j.samediff.frameworkimport.hooks.annotations

import org.nd4j.autodiff.functions.DifferentialFunction
import org.nd4j.autodiff.samediff.SDVariable

data class HookResult(val outputVariables: Map<String,List<SDVariable>> = emptyMap(),val functions: List<DifferentialFunction> = emptyList())
