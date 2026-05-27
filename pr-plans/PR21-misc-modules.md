# PR21: Miscellaneous Modules

**Estimated files:** ~160
**Merge layer:** varies (can merge independently)
**Complexity:** Low-Medium
**Reviewers:** Module owners

## Description

Smaller modules that don't fit into the major PR categories:
python4j, deeplearning4j (DL4J layers/Keras import), datavec,
contrib, omnihub, codegen (non-pom files), and resources.

## DL4J Core / Keras Import (~110 files)

### deeplearning4j-modelimport (~55)
Keras layer import classes:
- `KerasModelImport.java`
- Layer classes: `KerasAtrousConvolution1D/2D`, `KerasBatchNormalization`,
  `KerasBidirectional`, `KerasConvolution1D/2D/3D`, `KerasCropping*`,
  `KerasDense`, `KerasDropout`, `KerasEmbedding`, `KerasFlatten`,
  `KerasGlobalPooling`, `KerasInput`, `KerasLRN`, `KerasLSTM/SimpleRNN/GRU`,
  `KerasPooling*`, `KerasRepeatVector`, `KerasReshape`, `KerasSubsampling*`,
  `KerasUpsampling*`, `KerasZeroPadding*`

### deeplearning4j-nn (~40)
- Layers: `BatchNormalization`, `ConvolutionLayer`, `EmbeddingLayer`,
  `GlobalPoolingLayer`, `LSTM`, `OutputLayer`, `SubsamplingLayer`
- Configuration: `NeuralNetConfiguration`, `MultiLayerConfiguration`
- Graph: `ComputationGraph`, `ComputationGraphConfiguration`
- Updaters: `Adam`, `SGD`, `Nesterovs`, etc.
- SameDiff converter classes

### deeplearning4j-data (3)
- Data pipeline classes

### deeplearning4j-parallelwrapper (1)

### deeplearning4j-ui-parent (2)
- `module-info.java` files

### Build script (1)
- `buildmultiplescalaversions.sh`

## Python4J (~9 files)
- `Python.java`
- `PythonExecutioner.java`
- `PythonGC.java`
- `PythonGIL.java`
- `PythonObject.java`
- `PythonRefCount.java` (new)
- `PythonTypes.java`
- `UncheckedPythonInterpreter.java`
- `numpy/NumpyArray.java`

## Datavec (~8 files)
- `StringListToCounts.java`
- `StringListToIndices.java`
- `RecordConverter.java`
- `NDArrayWritable.java`
- `ArrowConverter.java`
- `RandomCropTransform.java`
- `datavec-excel/pom.xml`
- `datavec-excel/module-info.java`

## Contrib (~9 files)
- `contrib/benchmarking_nd4j/pom.xml`
- `contrib/blas-lapack-generator/pom.xml`
- `contrib/cpp-dependency-analyzer/` (5 files)
- `contrib/op-registry-updater/OpRegistryUpdater.kt`

## OmniHub (~17 files)
- `pom.xml`
- `BootstrapFromLocal.java`, `Framework.java`
- `HuggingFaceHubDownloader.java`
- `module-info.java`
- Kotlin files: `FinetuneRecipe.kt`, `Model.kt`, `LLMCatalog.kt`, `ModelBuilder.kt`

## Codegen (non-pom) (~13 files)
- `codegen/blas-lapack-generator/` — pom
- `codegen/libnd4j-gen/` — op-ir.proto, ParseOpFile.java
- `codegen/op-codegen/` — Namespace.java, Nd4jNamespaceGenerator.java,
  8 `.kt` op definition files, ConstructionTest.kt

## Keras model config
- `modelimport/keras/configs/bidirectional_last_timeStep.json`

### ADRs (1 — only those actually changed in the diff)
- `ADRs/0076 - OmniHub Model Repository Abstraction.md` — ModelRepository interface with priority-based backend registry

## Review Focus

- Python4J PythonRefCount — new reference counting for Python objects
- Keras import layer changes — verify shape handling
- OmniHub — model download/caching logic
