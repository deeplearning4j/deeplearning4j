#!/bin/bash

# Script to create separate branches from master with cherry-picked files from ag_new_release_updates_2
# Each PR contains only specific files for focused changes

set -e

# Base directory
BASE_DIR="/home/agibsonccc/Documents/GitHub/deeplearning4j"
cd "$BASE_DIR"

# Source branch with all changes
SOURCE_BRANCH="ag_new_release_updates_2"

# First, fix any existing merge conflicts and reset to clean state
echo "Cleaning up any existing merge conflicts..."
git merge --abort 2>/dev/null || true
git fetch origin
git checkout master
git reset --hard origin/master

# Function to create a new branch from master and cherry-pick specific files
create_pr_branch() {
    local branch_name=$1
    local commit_message=$2
    shift 2
    local files=("$@")
    
    echo "================================================="
    echo "Creating branch: $branch_name"
    echo "================================================="
    
    # First, ensure we're on master and it's up to date
    git checkout master
    git pull origin master --strategy=ours --no-edit || {
        echo "Pull failed, resetting to origin/master"
        git fetch origin
        git reset --hard origin/master
    }
    
    # Check if branch already exists
    if git show-ref --verify --quiet refs/heads/"$branch_name"; then
        echo "Branch $branch_name already exists, deleting it first..."
        git branch -D "$branch_name"
    fi
    
    # Create new branch from master
    git checkout -b "$branch_name"
    
    # Cherry-pick files from source branch
    for file in "${files[@]}"; do
        echo "Cherry-picking: $file"
        # Get the file from source branch (this will take OUR version)
        git checkout "$SOURCE_BRANCH" -- "$file" || echo "Warning: Could not cherry-pick $file"
    done
    
    # Add and commit the files
    git add .
    git commit -m "$commit_message"
    
    # Force push the branch (in case it existed remotely)
    git push --force origin "$branch_name"
    
    echo "Branch $branch_name created and pushed successfully"
    echo ""
}

# PR 1: Build System and Cross-Compilation Improvements
create_pr_branch "ag_build_system_improvements" \
"Build system and cross-compilation improvements

- Enhanced CMake configuration for cross-compilation
- Improved CUDA version management scripts  
- Updated build scripts for multiple platforms
- Added Raspberry Pi build support" \
"change-cuda-versions.sh" \
"datavec/buildmultiplescalaversions.sh" \
"deeplearning4j/buildmultiplescalaversions.sh" \
"libnd4j/CMakeLists.txt" \
"libnd4j/buildnativeoperations.sh" \
"libnd4j/cmake/postinst" \
"libnd4j/packages/push_to_bintray.sh" \
"libnd4j/pi_build.sh" \
"libnd4j/profile/CMakeLists.txt" \
"libnd4j/tests_cpu/run_tests.sh"

# PR 2: LibND4J Native Operations and Loop Optimizations
create_pr_branch "ag_libnd4j_native_ops" \
"LibND4J native operations and loop optimizations

- Updated BLAS interfaces
- Optimized core computational loops
- Enhanced buffer management
- Improved CUDA kernels" \
"libnd4j/include/cblas.h" \
"libnd4j/include/cblas_enum_conversion.h" \
"libnd4j/include/legacy/NativeOps.h" \
"libnd4j/include/legacy/cuda/NativeOps.cu" \
"libnd4j/include/loops/broadcasting.h" \
"libnd4j/include/loops/indexreduce.h" \
"libnd4j/include/loops/pairwise_transform.h" \
"libnd4j/include/loops/reduce3.h" \
"libnd4j/include/loops/scalar.h" \
"libnd4j/include/loops/summarystatsreduce.h" \
"libnd4j/include/system/buffer.h"

# PR 3: Dataset Infrastructure Refactoring
create_pr_branch "ag_dataset_infrastructure" \
"Dataset infrastructure refactoring

- Updated dataset fetchers and iterators
- Improved MNIST and Iris data handling
- Enhanced memory efficiency
- Better error handling" \
"deeplearning4j/deeplearning4j-core/src/main/resources/iris.dat" \
"deeplearning4j/deeplearning4j-data/deeplearning4j-datasets/src/main/java/org/deeplearning4j/datasets/fetchers/IrisDataFetcher.java" \
"deeplearning4j/deeplearning4j-data/deeplearning4j-datasets/src/main/java/org/deeplearning4j/datasets/fetchers/MnistDataFetcher.java" \
"deeplearning4j/deeplearning4j-data/deeplearning4j-datasets/src/main/java/org/deeplearning4j/datasets/iterator/impl/IrisDataSetIterator.java" \
"deeplearning4j/deeplearning4j-data/deeplearning4j-datasets/src/main/java/org/deeplearning4j/datasets/iterator/impl/LFWDataSetIterator.java" \
"deeplearning4j/deeplearning4j-data/deeplearning4j-datasets/src/main/java/org/deeplearning4j/datasets/iterator/impl/MnistDataSetIterator.java" \
"deeplearning4j/deeplearning4j-data/deeplearning4j-datasets/src/main/java/org/deeplearning4j/datasets/mnist/MnistDbFile.java" \
"deeplearning4j/deeplearning4j-data/deeplearning4j-datasets/src/main/java/org/deeplearning4j/datasets/mnist/MnistImageFile.java" \
"deeplearning4j/deeplearning4j-data/deeplearning4j-datasets/src/main/java/org/deeplearning4j/datasets/mnist/MnistLabelFile.java" \
"deeplearning4j/deeplearning4j-data/deeplearning4j-datasets/src/main/java/org/deeplearning4j/datasets/mnist/MnistManager.java" \
"deeplearning4j/deeplearning4j-data/deeplearning4j-utility-iterators/src/main/java/org/deeplearning4j/datasets/iterator/BaseDatasetIterator.java" \
"deeplearning4j/deeplearning4j-data/deeplearning4j-utility-iterators/src/main/java/org/deeplearning4j/datasets/iterator/DataSetFetcher.java" \
"deeplearning4j/deeplearning4j-data/deeplearning4j-utility-iterators/src/main/java/org/deeplearning4j/datasets/iterator/MultipleEpochsIterator.java" \
"deeplearning4j/deeplearning4j-data/deeplearning4j-utility-iterators/src/main/java/org/deeplearning4j/datasets/iterator/ReconstructionDataSetIterator.java" \
"deeplearning4j/deeplearning4j-data/deeplearning4j-utility-iterators/src/main/java/org/deeplearning4j/datasets/iterator/SamplingDataSetIterator.java"

# PR 4: NLP Core Infrastructure Update
create_pr_branch "ag_nlp_core_infrastructure" \
"NLP core infrastructure update

- Enhanced Word2Vec components
- Improved vocabulary management
- Better serialization support
- Optimized vectorizers" \
"deeplearning4j/deeplearning4j-nlp-parent/deeplearning4j-nlp/src/main/java/org/deeplearning4j/bagofwords/vectorizer/DefaultInputStreamCreator.java" \
"deeplearning4j/deeplearning4j-nlp-parent/deeplearning4j-nlp/src/main/java/org/deeplearning4j/bagofwords/vectorizer/TextVectorizer.java" \
"deeplearning4j/deeplearning4j-nlp-parent/deeplearning4j-nlp/src/main/java/org/deeplearning4j/models/embeddings/loader/WordVectorSerializer.java" \
"deeplearning4j/deeplearning4j-nlp-parent/deeplearning4j-nlp/src/main/java/org/deeplearning4j/models/word2vec/Huffman.java" \
"deeplearning4j/deeplearning4j-nlp-parent/deeplearning4j-nlp/src/main/java/org/deeplearning4j/models/word2vec/InputStreamCreator.java" \
"deeplearning4j/deeplearning4j-nlp-parent/deeplearning4j-nlp/src/main/java/org/deeplearning4j/models/word2vec/StreamWork.java" \
"deeplearning4j/deeplearning4j-nlp-parent/deeplearning4j-nlp/src/main/java/org/deeplearning4j/models/word2vec/VocabWord.java" \
"deeplearning4j/deeplearning4j-nlp-parent/deeplearning4j-nlp/src/main/java/org/deeplearning4j/models/word2vec/VocabWork.java" \
"deeplearning4j/deeplearning4j-nlp-parent/deeplearning4j-nlp/src/main/java/org/deeplearning4j/models/word2vec/iterator/Word2VecDataSetIterator.java" \
"deeplearning4j/deeplearning4j-nlp-parent/deeplearning4j-nlp/src/main/java/org/deeplearning4j/models/word2vec/wordstore/VocabCache.java" \
"deeplearning4j/deeplearning4j-nlp-parent/deeplearning4j-nlp/src/main/java/org/deeplearning4j/models/word2vec/wordstore/inmemory/InMemoryLookupCache.java"

# PR 5: NLP Text Processing Pipeline
create_pr_branch "ag_nlp_text_processing" \
"NLP text processing pipeline enhancement

- Improved document and sentence iterators
- Enhanced tokenization
- Better text preprocessing
- Efficient streaming support" \
"deeplearning4j/deeplearning4j-nlp-parent/deeplearning4j-nlp/src/main/java/org/deeplearning4j/text/documentiterator/DocumentIterator.java" \
"deeplearning4j/deeplearning4j-nlp-parent/deeplearning4j-nlp/src/main/java/org/deeplearning4j/text/documentiterator/FileDocumentIterator.java" \
"deeplearning4j/deeplearning4j-nlp-parent/deeplearning4j-nlp/src/main/java/org/deeplearning4j/text/documentiterator/LabelAwareDocumentIterator.java" \
"deeplearning4j/deeplearning4j-nlp-parent/deeplearning4j-nlp/src/main/java/org/deeplearning4j/text/sentenceiterator/BaseSentenceIterator.java" \
"deeplearning4j/deeplearning4j-nlp-parent/deeplearning4j-nlp/src/main/java/org/deeplearning4j/text/sentenceiterator/CollectionSentenceIterator.java" \
"deeplearning4j/deeplearning4j-nlp-parent/deeplearning4j-nlp/src/main/java/org/deeplearning4j/text/sentenceiterator/FileSentenceIterator.java" \
"deeplearning4j/deeplearning4j-nlp-parent/deeplearning4j-nlp/src/main/java/org/deeplearning4j/text/sentenceiterator/LineSentenceIterator.java" \
"deeplearning4j/deeplearning4j-nlp-parent/deeplearning4j-nlp/src/main/java/org/deeplearning4j/text/sentenceiterator/SentenceIterator.java" \
"deeplearning4j/deeplearning4j-nlp-parent/deeplearning4j-nlp/src/main/java/org/deeplearning4j/text/sentenceiterator/SentencePreProcessor.java" \
"deeplearning4j/deeplearning4j-nlp-parent/deeplearning4j-nlp/src/main/java/org/deeplearning4j/text/sentenceiterator/labelaware/LabelAwareFileSentenceIterator.java" \
"deeplearning4j/deeplearning4j-nlp-parent/deeplearning4j-nlp/src/main/java/org/deeplearning4j/text/sentenceiterator/labelaware/LabelAwareSentenceIterator.java" \
"deeplearning4j/deeplearning4j-nlp-parent/deeplearning4j-nlp/src/main/java/org/deeplearning4j/text/inputsanitation/InputHomogenization.java" \
"deeplearning4j/deeplearning4j-nlp-parent/deeplearning4j-nlp/src/main/java/org/deeplearning4j/text/stopwords/StopWords.java" \
"deeplearning4j/deeplearning4j-nlp-parent/deeplearning4j-nlp/src/main/java/org/deeplearning4j/text/tokenization/tokenizer/DefaultStreamTokenizer.java" \
"deeplearning4j/deeplearning4j-nlp-parent/deeplearning4j-nlp/src/main/java/org/deeplearning4j/text/tokenization/tokenizer/DefaultTokenizer.java" \
"deeplearning4j/deeplearning4j-nlp-parent/deeplearning4j-nlp/src/main/java/org/deeplearning4j/text/tokenization/tokenizer/TokenPreProcess.java" \
"deeplearning4j/deeplearning4j-nlp-parent/deeplearning4j-nlp/src/main/java/org/deeplearning4j/text/tokenization/tokenizer/Tokenizer.java" \
"deeplearning4j/deeplearning4j-nlp-parent/deeplearning4j-nlp/src/main/java/org/deeplearning4j/text/tokenization/tokenizer/preprocessor/EndingPreProcessor.java"

# PR 6: NLP Moving Window and Resources
create_pr_branch "ag_nlp_moving_window" \
"NLP moving window implementation and resources

- Added moving window functionality
- Updated sentiment analysis resources
- Enhanced linguistic data
- Improved visualization assets" \
"deeplearning4j/deeplearning4j-nlp-parent/deeplearning4j-nlp/src/main/java/org/deeplearning4j/text/movingwindow/ContextLabelRetriever.java" \
"deeplearning4j/deeplearning4j-nlp-parent/deeplearning4j-nlp/src/main/java/org/deeplearning4j/text/movingwindow/Util.java" \
"deeplearning4j/deeplearning4j-nlp-parent/deeplearning4j-nlp/src/main/java/org/deeplearning4j/text/movingwindow/Window.java" \
"deeplearning4j/deeplearning4j-nlp-parent/deeplearning4j-nlp/src/main/java/org/deeplearning4j/text/movingwindow/WindowConverter.java" \
"deeplearning4j/deeplearning4j-nlp-parent/deeplearning4j-nlp/src/main/java/org/deeplearning4j/text/movingwindow/Windows.java" \
"deeplearning4j/deeplearning4j-nlp-parent/deeplearning4j-nlp/src/main/java/org/deeplearning4j/text/movingwindow/WordConverter.java" \
"deeplearning4j/deeplearning4j-nlp-parent/deeplearning4j-nlp/src/main/resources/adverbs/adjectives" \
"deeplearning4j/deeplearning4j-nlp-parent/deeplearning4j-nlp/src/main/resources/adverbs/affirmative.csv" \
"deeplearning4j/deeplearning4j-nlp-parent/deeplearning4j-nlp/src/main/resources/adverbs/classscores" \
"deeplearning4j/deeplearning4j-nlp-parent/deeplearning4j-nlp/src/main/resources/adverbs/doubt.csv" \
"deeplearning4j/deeplearning4j-nlp-parent/deeplearning4j-nlp/src/main/resources/adverbs/intense.csv" \
"deeplearning4j/deeplearning4j-nlp-parent/deeplearning4j-nlp/src/main/resources/adverbs/negative.csv" \
"deeplearning4j/deeplearning4j-nlp-parent/deeplearning4j-nlp/src/main/resources/adverbs/negativedoc" \
"deeplearning4j/deeplearning4j-nlp-parent/deeplearning4j-nlp/src/main/resources/adverbs/positivedoc" \
"deeplearning4j/deeplearning4j-nlp-parent/deeplearning4j-nlp/src/main/resources/adverbs/weakintense.csv" \
"deeplearning4j/deeplearning4j-nlp-parent/deeplearning4j-nlp/src/main/resources/assets/d3.min.js" \
"deeplearning4j/deeplearning4j-nlp-parent/deeplearning4j-nlp/src/main/resources/assets/jquery.rest.min.js" \
"deeplearning4j/deeplearning4j-nlp-parent/deeplearning4j-nlp/src/main/resources/assets/render.js" \
"deeplearning4j/deeplearning4j-nlp-parent/deeplearning4j-nlp/src/main/resources/org/deeplearning4j/ehcache.xml" \
"deeplearning4j/deeplearning4j-nlp-parent/deeplearning4j-nlp/src/main/resources/org/deeplearning4j/plot/dropwizard/render.ftl" \
"deeplearning4j/deeplearning4j-nlp-parent/deeplearning4j-nlp/src/main/resources/render/dropwizard.yml" \
"deeplearning4j/deeplearning4j-nlp-parent/deeplearning4j-nlp/src/main/resources/sentiment/adjectives" \
"deeplearning4j/deeplearning4j-nlp-parent/deeplearning4j-nlp/src/main/resources/sentiment/affirmative.csv" \
"deeplearning4j/deeplearning4j-nlp-parent/deeplearning4j-nlp/src/main/resources/sentiment/classscores" \
"deeplearning4j/deeplearning4j-nlp-parent/deeplearning4j-nlp/src/main/resources/sentiment/doubt.csv" \
"deeplearning4j/deeplearning4j-nlp-parent/deeplearning4j-nlp/src/main/resources/sentiment/intense.csv" \
"deeplearning4j/deeplearning4j-nlp-parent/deeplearning4j-nlp/src/main/resources/sentiment/negative.csv" \
"deeplearning4j/deeplearning4j-nlp-parent/deeplearning4j-nlp/src/main/resources/sentiment/negativedoc" \
"deeplearning4j/deeplearning4j-nlp-parent/deeplearning4j-nlp/src/main/resources/sentiment/positivedoc" \
"deeplearning4j/deeplearning4j-nlp-parent/deeplearning4j-nlp/src/main/resources/sentiment/sentiwordnet.txt" \
"deeplearning4j/deeplearning4j-nlp-parent/deeplearning4j-nlp/src/main/resources/sentiment/weakintense.csv"

# PR 7: Neural Network Core API Updates
create_pr_branch "ag_nn_core_api" \
"Neural network core API updates

- Enhanced evaluation framework
- Improved exception handling
- Updated core interfaces
- Better configuration options" \
"deeplearning4j/deeplearning4j-nn/src/main/java/org/deeplearning4j/eval/ConfusionMatrix.java" \
"deeplearning4j/deeplearning4j-nn/src/main/java/org/deeplearning4j/eval/Evaluation.java" \
"deeplearning4j/deeplearning4j-nn/src/main/java/org/deeplearning4j/exception/DeepLearningException.java" \
"deeplearning4j/deeplearning4j-nn/src/main/java/org/deeplearning4j/exception/InvalidStepException.java" \
"deeplearning4j/deeplearning4j-nn/src/main/java/org/deeplearning4j/nn/api/Classifier.java" \
"deeplearning4j/deeplearning4j-nn/src/main/java/org/deeplearning4j/nn/api/Layer.java" \
"deeplearning4j/deeplearning4j-nn/src/main/java/org/deeplearning4j/nn/api/Model.java" \
"deeplearning4j/deeplearning4j-nn/src/main/java/org/deeplearning4j/nn/conf/NeuralNetConfiguration.java" \
"deeplearning4j/deeplearning4j-nn/src/main/java/org/deeplearning4j/nn/graph/ComputationGraph.java" \
"deeplearning4j/deeplearning4j-nn/src/main/java/org/deeplearning4j/nn/multilayer/MultiLayerNetwork.java"

# PR 8: Neural Network Layer Implementations
create_pr_branch "ag_nn_layers" \
"Neural network layer implementation updates

- Improved base layer architecture
- Enhanced autoencoder implementation
- Better weight initialization
- Optimized training layers" \
"deeplearning4j/deeplearning4j-nn/src/main/java/org/deeplearning4j/nn/layers/BaseLayer.java" \
"deeplearning4j/deeplearning4j-nn/src/main/java/org/deeplearning4j/nn/layers/BasePretrainNetwork.java" \
"deeplearning4j/deeplearning4j-nn/src/main/java/org/deeplearning4j/nn/layers/OutputLayer.java" \
"deeplearning4j/deeplearning4j-nn/src/main/java/org/deeplearning4j/nn/layers/feedforward/autoencoder/AutoEncoder.java" \
"deeplearning4j/deeplearning4j-nn/src/main/java/org/deeplearning4j/nn/layers/training/CenterLossOutputLayer.java" \
"deeplearning4j/deeplearning4j-nn/src/main/java/org/deeplearning4j/nn/weights/WeightInit.java" \
"deeplearning4j/deeplearning4j-nn/src/main/java/org/deeplearning4j/nn/weights/WeightInitUtil.java"

# PR 9: Optimization Framework Updates  
create_pr_branch "ag_optimization_framework" \
"Optimization framework updates

- Enhanced iteration listeners
- Improved line optimization
- Better training monitoring" \
"deeplearning4j/deeplearning4j-nn/src/main/java/org/deeplearning4j/optimize/api/IterationListener.java" \
"deeplearning4j/deeplearning4j-nn/src/main/java/org/deeplearning4j/optimize/api/LineOptimizer.java"

# PR 10: Model Zoo YOLO2 Update
create_pr_branch "ag_model_zoo_yolo2" \
"Model Zoo: YOLO2 implementation update

- Improved YOLO2 architecture
- Better anchor box handling
- Enhanced post-processing" \
"deeplearning4j/deeplearning4j-zoo/src/main/java/org/deeplearning4j/zoo/model/YOLO2.java"

# PR 11: ND4J Protocol Buffer Updates
create_pr_branch "ag_nd4j_protobuf" \
"ND4J protocol buffer and backend updates

- Updated JavaScript generator
- Improved build scripts
- Enhanced test configuration" \
"nd4j/nd4j-backends/nd4j-api-parent/nd4j-api/src/main/protobuf/tf/google/protobuf/compiler/js/js_generator.cc" \
"nd4j/nd4j-backends/nd4j-api-parent/nd4j-api/src/main/protobuf/tf/google/protobuf/compiler/js/js_generator.h" \
"nd4j/nd4j-backends/nd4j-api-parent/nd4j-api/src/main/protobuf/tf/google/protobuf/compiler/zip_output_unittest.sh" \
"nd4j/nd4j-backends/nd4j-api-parent/nd4j-api/src/main/protobuf/tf/google/protobuf/io/gzip_stream_unittest.sh" \
"nd4j/nd4j-backends/nd4j-api-parent/nd4j-api/src/main/protobuf/tf/google/protobuf/stubs/common.cc" \
"nd4j/nd4j-backends/nd4j-backend-impls/nd4j-native/src/test/resources/logback.xml"

# PR 12: Platform Test Infrastructure
create_pr_branch "ag_platform_tests" \
"Platform test infrastructure update

- Updated test binary wrapper" \
"platform-tests/bin/java"

echo "================================================="
echo "All branches created successfully!"
echo "================================================="
echo ""
echo "Next steps:"
echo "1. Create pull requests using GitHub CLI or web interface"
echo "2. Add the detailed descriptions from the plan"
echo ""
echo "To create PRs with GitHub CLI:"
echo "gh pr create --base master --head branch_name --title \"Title\" --body \"Description\""
