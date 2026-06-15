# ADR: LlamaCpp, OneDNN, and cuDNN Backend Classifiers

## Status
Accepted

## Date
2025-12-30

## Context

DeepLearning4J's nd4j-native and nd4j-cuda backends support platform-specific classifiers for optimized builds (e.g., `avx2`, `avx512`). There was a need to extend this system to support additional backend capabilities:

1. **llama.cpp/GGML Support**: For loading and running LLM models in GGUF format, enabling integration with the growing ecosystem of quantized language models. Supports both CPU and CUDA acceleration.

2. **OneDNN Support**: Intel's Deep Neural Network Library (formerly MKL-DNN) provides optimized implementations for CPU-based deep learning operations.

3. **cuDNN Support**: NVIDIA's CUDA Deep Neural Network library provides GPU-accelerated primitives for deep learning.

These capabilities require linking against additional native libraries and should be optional to keep the base build lightweight.

## Decision

We have implemented classifier-based extension profiles for `llamacpp`, `onednn`, and `cudnn` backends, following the existing pattern used for `avx2` and `avx512` extensions.

### Architecture

```
nd4j-native (CPU backend - base jar with no native libs)
├── nd4j-native-linux-x86_64.jar (standard classifier)
├── nd4j-native-linux-x86_64-avx2.jar (AVX2 optimized)
├── nd4j-native-linux-x86_64-avx512.jar (AVX512 optimized)
├── nd4j-native-linux-x86_64-onednn.jar (OneDNN enabled)
└── nd4j-native-linux-x86_64-llamacpp.jar (llama.cpp CPU enabled)

nd4j-cuda (CUDA backend - base jar with no native libs)
├── nd4j-cuda-linux-x86_64.jar (standard classifier)
├── nd4j-cuda-linux-x86_64-cudnn.jar (cuDNN enabled)
└── nd4j-cuda-linux-x86_64-llamacpp.jar (llama.cpp CUDA enabled)
```

### Components Modified

#### 1. libnd4j/cmake/Options.cmake
Already contains the helper options:
```cmake
option(HELPERS_onednn "Enable OneDNN helper" OFF)
option(HELPERS_cudnn "Enable cuDNN helper" OFF)
option(HELPERS_llamacpp "Enable llama.cpp/GGML helper" OFF)
```

#### 2. libnd4j/buildnativeoperations.sh
Added extension handling for the build script with CUDA support:
```bash
elif [ "$CHIP_EXTENSION" == "onednn" ]; then
    CHIP_EXTENSION="onednn"
    if [ "$CHIP" == "cpu" ]; then
        ARCH="x86-64"
    fi
    export CMAKE_COMMAND="$CMAKE_COMMAND -DHELPERS_onednn=ON"
elif [ "$CHIP_EXTENSION" == "llamacpp" ]; then
    CHIP_EXTENSION="llamacpp"
    if [ "$CHIP" == "cpu" ]; then
        ARCH="x86-64"
    fi
    export CMAKE_COMMAND="$CMAKE_COMMAND -DHELPERS_llamacpp=ON"
    if [ "$CHIP" == "cuda" ]; then
        # Enable CUDA support in llama.cpp
        export CMAKE_COMMAND="$CMAKE_COMMAND -DLLAMA_CUDA=ON"
    fi
elif [ "$CHIP_EXTENSION" == "cudnn" ]; then
    CHIP_EXTENSION="cudnn"
    if [ "$CHIP" == "cuda" ]; then
        export CMAKE_COMMAND="$CMAKE_COMMAND -DHELPERS_cudnn=ON"
    fi
```

#### 3. nd4j-native/pom.xml
Added Maven profiles for CPU classifier activation:
```xml
<profile>
    <id>llamacpp</id>
    <activation>
        <property>
            <name>libnd4j.extension</name>
            <value>llamacpp</value>
        </property>
    </activation>
    <properties>
        <javacpp.platform.extension>-llamacpp</javacpp.platform.extension>
    </properties>
</profile>
```

#### 4. nd4j-cuda/pom.xml
Added Maven profiles for CUDA classifier activation:
```xml
<profile>
    <id>llamacpp</id>
    <activation>
        <property>
            <name>libnd4j.extension</name>
            <value>llamacpp</value>
        </property>
    </activation>
    <properties>
        <javacpp.platform.extension>-llamacpp</javacpp.platform.extension>
    </properties>
</profile>

<profile>
    <id>cudnn</id>
    <activation>
        <property>
            <name>libnd4j.extension</name>
            <value>cudnn</value>
        </property>
    </activation>
    <properties>
        <javacpp.platform.extension>-cudnn</javacpp.platform.extension>
    </properties>
</profile>
```

#### 5. nd4j-native-platform/pom.xml and nd4j-cuda-platform/pom.xml
Added platform-specific classifier dependencies for multi-platform builds:
```xml
<profile>
    <id>llamacpp</id>
    <dependencies>
        <dependency>
            <artifactId>nd4j-native</artifactId>
            <classifier>${javacpp.platform.linux-x86_64}-llamacpp</classifier>
        </dependency>
        <!-- macosx, windows variants -->
    </dependencies>
</profile>
```

## Usage

### Building with llama.cpp CPU Support

```bash
# Step 1: Build libnd4j with llamacpp helper enabled (CPU)
cd libnd4j
./buildnativeoperations.sh -c cpu -e llamacpp

# Step 2: Build nd4j-native with llamacpp classifier
cd ..
mvn -Pcpu -Dlibnd4j.extension=llamacpp \
    -pl libnd4j,:nd4j-api,:nd4j-cpu-backend-common,:nd4j-native \
    clean install -DskipTests
```

### Building with llama.cpp CUDA Support

```bash
# Step 1: Build libnd4j with llamacpp helper enabled (CUDA)
cd libnd4j
./buildnativeoperations.sh -c cuda -e llamacpp

# Step 2: Build nd4j-cuda with llamacpp classifier
cd ..
mvn -Pcuda -Dlibnd4j.extension=llamacpp \
    -pl libnd4j,:nd4j-api,:nd4j-cuda-preset,:nd4j-cuda \
    clean install -DskipTests
```

### Building with OneDNN Support

```bash
# Step 1: Build libnd4j with onednn helper enabled
cd libnd4j
./buildnativeoperations.sh -c cpu -e onednn

# Step 2: Build nd4j-native with onednn classifier
cd ..
mvn -Pcpu -Dlibnd4j.extension=onednn \
    -pl libnd4j,:nd4j-api,:nd4j-cpu-backend-common,:nd4j-native \
    clean install -DskipTests
```

### Building with cuDNN Support

```bash
# Step 1: Build libnd4j with cudnn helper enabled
cd libnd4j
./buildnativeoperations.sh -c cuda -e cudnn

# Step 2: Build nd4j-cuda with cudnn classifier
cd ..
mvn -Pcuda -Dlibnd4j.extension=cudnn \
    -pl libnd4j,:nd4j-api,:nd4j-cuda-preset,:nd4j-cuda \
    clean install -DskipTests
```

### Using in Applications

```xml
<!-- For llamacpp CPU support -->
<dependency>
    <groupId>org.eclipse.deeplearning4j</groupId>
    <artifactId>nd4j-native</artifactId>
    <version>${dl4j.version}</version>
    <classifier>linux-x86_64-llamacpp</classifier>
</dependency>

<!-- For llamacpp CUDA support (GPU accelerated) -->
<dependency>
    <groupId>org.eclipse.deeplearning4j</groupId>
    <artifactId>nd4j-cuda</artifactId>
    <version>${dl4j.version}</version>
    <classifier>linux-x86_64-llamacpp</classifier>
</dependency>

<!-- For onednn support -->
<dependency>
    <groupId>org.eclipse.deeplearning4j</groupId>
    <artifactId>nd4j-native</artifactId>
    <version>${dl4j.version}</version>
    <classifier>linux-x86_64-onednn</classifier>
</dependency>

<!-- For cudnn support -->
<dependency>
    <groupId>org.eclipse.deeplearning4j</groupId>
    <artifactId>nd4j-cuda</artifactId>
    <version>${dl4j.version}</version>
    <classifier>linux-x86_64-cudnn</classifier>
</dependency>
```

## Consequences

### Positive
- **Modular Design**: Users only include the capabilities they need
- **Smaller Deployments**: Base jar remains lightweight
- **Consistent Pattern**: Follows existing avx2/avx512 classifier pattern
- **LLM Integration**: Enables GGUF model support for running quantized LLMs
- **CPU Optimization**: OneDNN provides significant performance improvements for Intel CPUs

### Negative
- **Build Complexity**: Multiple classifier jars need to be built and published
- **Dependency Management**: Users must explicitly choose the right classifier
- **CI/CD Overhead**: More build matrix combinations for releases

### Neutral
- **Backward Compatible**: No changes to existing functionality
- **Optional Features**: Both extensions are disabled by default

## Related

- **nd4j-ggml module**: Provides Java API for GGML/GGUF model loading (uses llamacpp backend)
- **Platform ops in libnd4j**: OneDNN-optimized operations in `include/ops/declarable/platform/mkldnn/`
- **Existing classifiers**: avx2, avx512 profiles in nd4j-native

## Future Considerations

1. **ARM Support**: Add llamacpp classifiers for ARM platforms (linux-arm64, macosx-arm64)
2. **Auto-detection**: Runtime selection of optimal backend based on available hardware
3. **Combined Classifiers**: Support for multiple extensions (e.g., `onednn-avx512`)
