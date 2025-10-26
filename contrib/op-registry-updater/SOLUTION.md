# SOLUTION: EmbedLayerNormalization Missing from ONNX Registry

## Root Cause Identified

The issue is that `EmbedLayerNormalization` is **not a standard ONNX operation** - it's a Microsoft-specific extension operator used in transformer models like BERT. Therefore, it's not included in the standard ONNX op descriptor list loaded by `OpDescriptorLoaderHolder.listForFramework<Onnx.NodeProto>("onnx")`.

## Evidence

1. **PreImportHook is discovered**: ✅ Cache shows `Op: EmbedLayerNormalization -> 2 hooks`
2. **Registry mapping exists**: ✅ `embedLayerNormalization` noop mapping is defined
3. **Missing from processing**: ❌ Not in the 186 ONNX ops processed in the main loop
4. **Runtime error**: "No import process defined for EmbedLayerNormalization"

## The Fix

The solution is to ensure that `EmbedLayerNormalization` gets registered as a framework operation even though it's not in the standard ONNX specification.

### Option 1: Manual Op Registration (Recommended)

Add explicit registration for Microsoft ONNX extensions in `OnnxOpDeclarations.kt`:

```kotlin
object OnnxOpDeclarations {
    init {
        // ... existing code ...
        
        // Register Microsoft ONNX extensions that aren't in standard ONNX op descriptors
        val microsoftExtensions = listOf(
            "EmbedLayerNormalization",
            "Attention", 
            "MultiHeadAttention",
            "SkipLayerNormalization",
            "FastGelu",
            "Gelu"
        )
        
        microsoftExtensions.forEach { opName ->
            if (!onnxOpRegistry.inputFrameworkOpNames().contains(opName)) {
                // Create a dummy op descriptor for the extension
                val dummyOpDesc = Onnx.NodeProto.newBuilder()
                    .setOpType(opName)
                    .setDomain("com.microsoft")
                    .build()
                onnxOpRegistry.registerInputFrameworkOpDef(opName, dummyOpDesc)
            }
        }
        
        // ... rest of existing code ...
    }
}
```

### Option 2: Fix the Registry Update Process

Alternatively, modify the registry update process to include operations that have PreImportHooks but aren't in the standard op list:

```kotlin
private fun updateOnnxRegistry(validateOnly: Boolean, debug: Boolean) {
    // ... existing code ...
    
    // Get all ops with PreImportHooks
    val onnxPreHookOps = ImportReflectionCache.preProcessRuleImplementationsByOp.row("onnx").keys
    
    // Combine framework ops with PreImportHook ops
    val allOpsToProcess = (frameworkOpNames + onnxPreHookOps).toSet()
    
    allOpsToProcess.forEach { name ->
        // ... process each op ...
    }
}
```

## Implementation

I recommend **Option 1** because it addresses the root cause by properly registering Microsoft ONNX extensions in the framework registry.

## Files to Modify

1. **`nd4j/samediff-import/samediff-import-onnx/src/main/kotlin/org/nd4j/samediff/frameworkimport/onnx/definitions/OnnxOpDeclarations.kt`**
   - Add Microsoft extension registration in the `init` block

## Expected Result

After this fix:
- `EmbedLayerNormalization` will appear in the `frameworkOpNames` list
- The registry update will process it as a noop operation  
- The PreImportHook will be properly associated with the noop mapping
- Runtime imports will use the `EmbedLayerNormalization` PreImportHook implementation
- The error "No import process defined for EmbedLayerNormalization" will be resolved

## Testing

```bash
cd contrib/op-registry-updater
./update-op-registry.sh --framework onnx --debug --validate-only
```

After the fix, you should see:
```
📝 ONNX op 'EmbedLayerNormalization' mapped to noop with 2 PreImportHook(s)
    Hook: EmbedLayerNormalization
    Hook: EmbedLayerNormalization
```

And in the final summary:
```
✅ Noop ops WITH PreImportHooks (31):
  - EmbedLayerNormalization  # <-- This should now appear
  - BatchNormalization
  - Cast
  # ... etc
```
