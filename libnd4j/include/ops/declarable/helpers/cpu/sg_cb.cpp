/* ******************************************************************************
*
*
* This program and the accompanying materials are made available under the
* terms of the Apache License, Version 2.0 which is available at
* https://www.apache.org/licenses/LICENSE-2.0.
*
*  See the NOTICE file distributed with this work for additional
*  information regarding copyright ownership.
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
* WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
* License for the specific language governing permissions and limitations
* under the License.
*
* SPDX-License-Identifier: Apache-2.0
******************************************************************************/

//
// @author raver119@gmail.com
//
#include <execution/Threads.h>
#include <ops/declarable/helpers/sg_cb.h>
#include <math/templatemath.h>
#define HS_MAX_EXP 6.0f
#include <algorithm>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <new>
#include <vector>

namespace sd {
namespace ops {
namespace helpers {
template <typename T>
void hSoftmax_(T *vsyn0, T *vsyn1, T *vexpTable, T *vneu1e, const double alpha, const int vectorLength, const int code,

               const int expLength, const bool isInference) {
  auto syn0 = reinterpret_cast<T *>(vsyn0);
  auto syn1 = reinterpret_cast<T *>(vsyn1);
  auto expTable = reinterpret_cast<T *>(vexpTable);
  auto neu1e = reinterpret_cast<T *>(vneu1e);

  T dot(0.0f);
  T g(0.0f);
  T f(0.0f);


  // dot
  PRAGMA_OMP_SIMD_ARGS(reduction(+:dot))
  for (int e = 0; e < vectorLength; e++) {
    dot += syn0[e] * syn1[e];

  }


  // gradient
  if (dot < (T)-HS_MAX_EXP || dot >= (T)HS_MAX_EXP) return;
  int idx = static_cast<int>((dot + HS_MAX_EXP) * ((float)expLength / HS_MAX_EXP / 2.0f));

  if (idx >= expLength || idx < 0) return;

  f = expTable[idx];
  g = (static_cast<T>(1.0f) - static_cast<T>(code) - f) * (T)alpha;

  if(!isInference) {
    PRAGMA_OMP_SIMD
    for (int x = 0; x < vectorLength; x++) {
      neu1e[x] += g * syn1[x];
      syn1[x] += g * syn0[x];

    }
  } else {
    PRAGMA_OMP_SIMD
    for (int e = 0; e < vectorLength; e++) {
      neu1e[e] = g * syn1[e] + neu1e[e];
    }


  }
}


template <typename T>
void nSampling_(void *vsyn0, void *vsyn1Neg, void *vexpTable, void *vneu1e, double alpha, int vectorLength, int code,
                int expLength, bool isInference) {
  auto syn0 = reinterpret_cast<T *>(vsyn0);
  auto syn1Neg = reinterpret_cast<T *>(vsyn1Neg);
  auto expTable = reinterpret_cast<T *>(vexpTable);
  auto neu1e = reinterpret_cast<T *>(vneu1e);
  T dot = (T)0.0f;
  T g = (T)0.0f;

  PRAGMA_OMP_SIMD_ARGS(reduction(+:dot))
  for (int e = 0; e < vectorLength; e++) {
    dot += syn0[e] * syn1Neg[e];
  }

  if (dot > HS_MAX_EXP)
    g = (code - 1) * alpha;
  else if (dot < (T)-HS_MAX_EXP)
    g = (code - 0) * alpha;
  else {
    int idx = (int)((dot + (T)HS_MAX_EXP) * ((T)expLength / HS_MAX_EXP / 2.0));
    if (idx >= expLength) return;

    if (idx < 0) return;

    g = ((T)code - expTable[idx]) * alpha;
  }


  // axpy2
  if (!isInference) {
    PRAGMA_OMP_SIMD
    for (int e = 0; e < vectorLength; e++) {
      neu1e[e] += g * syn1Neg[e];
      syn1Neg[e] += g * syn0[e];
    }


  } else {
    // axpy1
    PRAGMA_OMP_SIMD
    for (int e = 0; e < vectorLength; e++) {
      neu1e[e] += g * syn1Neg[e];
    }


  }

}

// One cbow round, `iterations` times over. The codes, indices, context and locked words are read element by element, so
// they can be arrays of any integer type. Every iteration averages the window into neu1 and collects the error in
// neu1e, both starting from zero; the error moves the words of the window - or the inference vector, the only thing
// that learns when there is one - and the learning rate decays after it:
//   alpha = (alpha - minLearningRate) / (iterations - iteration) + minLearningRate
template <typename T>
void cbow_(NDArray &vsyn0, NDArray &vsyn1, NDArray &vsyn1Neg, NDArray &vexpTable, NDArray &vnegTable, NDArray &vinfVector, int target,
           int ngStarter, NDArray &context, NDArray &lockedWords, NDArray &indices, NDArray &codes, double alpha,
           sd::LongType randomValue, const int contextWidth, const int hsRounds, const int nsRounds,
           const int vocabSize, const int vectorLength, const int expLength, const int negLength, const int numLabels,
           const bool trainWords,double minLearningRate,const int iterations) {
  auto syn0 = reinterpret_cast<T *>(vsyn0.bufferAsT<T>());
  auto syn1 = reinterpret_cast<T *>(vsyn1.bufferAsT<T>());
  auto syn1Neg = reinterpret_cast<T *>(vsyn1Neg.bufferAsT<T>());
  auto expTable = reinterpret_cast<T *>(vexpTable.bufferAsT<T>());
  auto negTable = reinterpret_cast<T *>(vnegTable.bufferAsT<T>());
  auto infVector = reinterpret_cast<T *>(vinfVector.bufferAsT<T>());

  // negative sampling cannot start from a word that is not a row of syn1Neg
  if (nsRounds > 0 && (ngStarter < 0 || ngStarter >= vocabSize)) {
    THROW_EXCEPTION("CBOW: the word to sample negatives for is not a row of syn1Neg");
  }

  if (hsRounds > indices.lengthOf()) {
    THROW_EXCEPTION("CBOW: there are fewer indices than codes");
  }

  // The window, its locked words and the rows of hierarchic softmax are the same in every iteration: they are read once.
  // The values that pad the window are negative.
  std::vector<int> window(contextWidth);
  int actualContext = 0;
  for (int c = 0; c < contextWidth; c++) {
    window[c] = context.e<int>(c);
    if (window[c] >= vocabSize) THROW_EXCEPTION("ContextID can't be >= vocab size");

    if (window[c] >= 0) actualContext++;
  }

  // for inference the inference vector is one more member of the window
  if (infVector != nullptr) actualContext++;

  const sd::LongType numLocked = lockedWords.lengthOf();
  std::vector<int> locked(contextWidth, 0);
  for (int c = 0; c < contextWidth && c < numLocked; c++) {
    locked[c] = lockedWords.e<int>(c);
  }

  std::vector<int> hsIndices(hsRounds);
  std::vector<int> hsCodes(hsRounds);
  for (int h = 0; h < hsRounds; h++) {
    hsIndices[h] = indices.e<int>(h);
    hsCodes[h] = codes.e<int>(h);

    // we're skipping padded values: the code of a padding is -1
    if (hsIndices[h] < 0 || hsCodes[h] < 0) continue;

    if (hsIndices[h] >= vocabSize) THROW_EXCEPTION("Index can't be > vocab size");
  }

  std::vector<T> neu1Buffer(vectorLength);
  std::vector<T> neu1eBuffer(vectorLength);
  T *neu1 = neu1Buffer.data();
  T *neu1e = neu1eBuffer.data();

  for (int iteration = 0; iteration < iterations; iteration++) {
    memset(neu1, 0, vectorLength * sizeof(T));
    memset(neu1e, 0, vectorLength * sizeof(T));

    // building neu1 for current window
    for (int c = 0; c < contextWidth; c++) {
      if (window[c] < 0) continue;

      T *syn0word = syn0 + static_cast<sd::LongType>(window[c]) * vectorLength;
      PRAGMA_OMP_SIMD
      for (int e = 0; e < vectorLength; e++) {
        neu1[e] += syn0word[e];
      }
    }

    if (infVector != nullptr) {
      PRAGMA_OMP_SIMD
      for (int e = 0; e < vectorLength; e++) {
        neu1[e] += infVector[e];
      }
    }

    if (actualContext > 1) {
      PRAGMA_OMP_SIMD
      for (int e = 0; e < vectorLength; e++) {
        neu1[e] /= actualContext;
      }
    }

    // softmax round
    for (int h = 0; h < hsRounds; h++) {
      // we're skipping padded values: the code of a padding is -1
      if (hsIndices[h] < 0 || hsCodes[h] < 0) continue;

      hSoftmax_<T>(neu1, syn1 + static_cast<sd::LongType>(hsIndices[h]) * vectorLength, expTable, neu1e, alpha,
                   vectorLength, hsCodes[h], expLength, infVector != nullptr);
    }


    auto nsStarter = ngStarter;
    auto irow = nsStarter;
    if (nsRounds > 0) {

      for (int r = 0; r < nsRounds + 1; r++) {
        if (r == 0) {
          // target is known in advance
        } else {
          randomValue = randomValue * (unsigned long long)25214903917 + 11;
          auto idx = sd::math::sd_abs<sd::LongType,sd::LongType>((randomValue >> 16) % negLength);
          irow = idx >= negLength ? -1 : static_cast<int>(negTable[idx]);
          if (irow < 0 || irow >= vocabSize) {
            irow = static_cast<int>(static_cast<unsigned long long>(randomValue) % (vocabSize - 1)) + 1;
          }
          if (irow == nsStarter) continue;
        }

        auto syn1NegRow = syn1Neg + static_cast<sd::LongType>(irow) * vectorLength;
        nSampling_<T>(neu1, syn1NegRow, expTable, neu1e, alpha, vectorLength, r == 0 ? 1 : 0,
                      expLength, infVector != nullptr);
      }
    }

    // if we don't train words - we skip start of idxSyn0
    int starter = trainWords == 1 ? 0 : contextWidth - numLabels;
    if (starter < 0) starter = 0;

    if (infVector == nullptr) {
      // propagate neu1e -> syn0
      for (int c = starter; c < contextWidth; c++) {
        // the words that are locked do not learn; the values that pad the window are negative
        if (locked[c] == 1 || window[c] < 0) continue;

        T *syn0word = syn0 + static_cast<sd::LongType>(window[c]) * vectorLength;
        PRAGMA_OMP_SIMD
        for (int e = 0; e < vectorLength; e++) {
          syn0word[e] += neu1e[e];
        }

      }
    } else {
      PRAGMA_OMP_SIMD
      for (int e = 0; e < vectorLength; e++) {
        infVector[e] += neu1e[e];
      }
    }

    alpha = ((alpha - static_cast<double>(minLearningRate)) / static_cast<double>((iterations - iteration))) + static_cast<double>(minLearningRate);
  }
}
BUILD_SINGLE_TEMPLATE( void cbow_,
                      (NDArray &syn0, NDArray &syn1,NDArray &syn1Neg, NDArray &expTable, NDArray &vnegTable, NDArray &vinfVector,
                          int target, int ngStarter, NDArray &context, NDArray &lockedWords, NDArray &indices, NDArray &codes,
                          double alpha, sd::LongType randomValue, const int contextWidth, const int hsRounds,
                          const int nsRounds, const int vocabSize, const int vectorLength, const int expLength,
                          const int negLength, const int numLabels, const bool trainWords,double minLearningRate,const int iterations),
                      SD_NATIVE_FLOAT_TYPES);

// One skipgram round, `iterations` times over. The indices and codes are read element by element, so they can be arrays
// of any integer type; a code of -1 is a padding. Every iteration trains the vector that learns - the inference vector
// when there is one, the row of the target in syn0 otherwise - against a neu1e that starts from zero, and the learning
// rate decays after it: alpha = (alpha - minLearningRate) / (iterations - iteration) + minLearningRate. With an
// inference vector syn1 and syn1Neg are only read.
template <typename T>
void skipgram_(void *vsyn0, void *vsyn1, void *vsyn1Neg, void *vexpTable, void *vnegTable, void *vinfVector, int target,
               int ngStarter, NDArray &indices, NDArray &codes, double alpha, sd::LongType randomValue, const int hsRounds,
               const int nsRounds, const int vocabSize, const int vectorLength, const int expLength,
               const int negLength,double minLearningRate,const int iterations) {
  auto syn0 = reinterpret_cast<T *>(vsyn0);
  auto syn1 = reinterpret_cast<T *>(vsyn1);
  auto syn1Neg = reinterpret_cast<T *>(vsyn1Neg);
  auto expTable = reinterpret_cast<T *>(vexpTable);
  auto negTable = reinterpret_cast<T *>(vnegTable);
  auto infVector = reinterpret_cast<T *>(vinfVector);

  if (infVector == nullptr && (target < 0 || target >= vocabSize)) {
    THROW_EXCEPTION("SkipGram: the target is not a row of syn0");
  }

  if (nsRounds > 0 && (ngStarter < 0 || ngStarter >= vocabSize)) {
    THROW_EXCEPTION("SkipGram: the word to sample negatives for is not a row of syn1Neg");
  }

  if (hsRounds > indices.lengthOf()) {
    THROW_EXCEPTION("SkipGram: there are fewer indices than codes");
  }

  // the rows of hierarchic softmax are the same in every iteration: they are read once
  std::vector<int> hsIndices(hsRounds);
  std::vector<int> hsCodes(hsRounds);
  for (int r = 0; r < hsRounds; r++) {
    hsIndices[r] = indices.e<int>(r);
    hsCodes[r] = codes.e<int>(r);
  }

  std::vector<T> neu1eBuffer(vectorLength);
  T *neu1e = neu1eBuffer.data();
  auto syn0row = infVector != nullptr ? infVector : syn0 + static_cast<sd::LongType>(target) * vectorLength;

  for(int iteration = 0; iteration < iterations; iteration++) {
    // neu1e is the error of one iteration: it starts from zero every time
    memset(neu1e, 0, vectorLength * sizeof(T));

    // hierarchic softmax goes first (if enabled)
    for (int r = 0; r < hsRounds; r++) {
      // the code of a padding is -1
      if (hsCodes[r] < 0 || hsIndices[r] < 0 || hsIndices[r] >= vocabSize) continue;

      hSoftmax_<T>(syn0row, syn1 + static_cast<sd::LongType>(hsIndices[r]) * vectorLength, expTable, neu1e, alpha,
                   vectorLength, hsCodes[r], expLength, infVector != nullptr);
    }

    // negative sampling goes second (if enabled)
    auto nsStarter = ngStarter;
    auto irow = nsStarter;
    if (nsRounds > 0) {

      for (int r = 0; r < nsRounds + 1; r++) {
        if (r == 0) {
          // target is known in advance
        } else {
          randomValue = randomValue * (unsigned long long)25214903917 + 11;
          auto idx = sd::math::sd_abs<sd::LongType,sd::LongType>((randomValue >> 16) % negLength);
          irow = idx >= negLength ? -1 : static_cast<int>(negTable[idx]);

          if (irow < 0 || irow >= vocabSize) {
            irow = static_cast<int>(static_cast<unsigned long long>(randomValue) % (vocabSize - 1)) + 1;
          }
          if (irow == nsStarter) continue;
        }
        nSampling_<T>(syn0row, syn1Neg + static_cast<sd::LongType>(irow) * vectorLength, expTable, neu1e, alpha,
                      vectorLength, r == 0 ? 1 : 0, expLength, infVector != nullptr);

      }
    }

    // syn0row is the row of the target, or the inference vector
    for (int e = 0; e < vectorLength; e++) {
      syn0row[e] += neu1e[e];
    }

    alpha = ((alpha - static_cast<double>(minLearningRate)) / static_cast<double>((iterations - iteration))) + static_cast<double>(minLearningRate);
  }
}

BUILD_SINGLE_TEMPLATE( void skipgram_,
                      (void *syn0, void *syn1, void *syn1Neg, void *expTable, void *vnegTable, void *vinfVector,
                          int target, int ngStarter,NDArray &indices, NDArray &codes, double alpha, sd::LongType randomValue,
                          const int hsRounds, const int nsRounds, const int vocabSize, const int vectorLength,
                          const int expLength, const int negLength,double minLearningRate,const int iterations),
                      SD_NATIVE_FLOAT_TYPES);

int binarySearch(const int *haystack, const int needle, const int totalElements) {
  int firstIndex = 0;
  int lastIndex = totalElements - 1;
  int halfIndex = sd::math::sd_floor<float, int>((lastIndex + firstIndex) / (float)2);

  while (haystack[halfIndex] != needle && firstIndex < lastIndex) {
    if (needle < haystack[halfIndex]) {
      lastIndex = halfIndex - 1;
    } else if (needle > haystack[halfIndex]) {
      firstIndex = halfIndex + 1;
    }
    halfIndex = sd::math::sd_floor<float, int>((lastIndex + firstIndex) / (float)2);
  }

  return (haystack[halfIndex] == needle) ? halfIndex : -1;
}


template <typename T>
void doSkipGramLoop_(NDArray &s0, NDArray &s1, NDArray &s1n, NDArray &vinfVector, NDArray&targets,
                     NDArray&negStarters, NDArray&indices, NDArray&codes, NDArray&lr,
                     NDArray&nextRandom, const int nsRounds, const int vocabSize, const int vectorLength,
                     const int expLength, const int negLength, T *const expTable, const T *negTable,
                     const LongType hsRounds, int t);

template <typename T>
void doSkipGramInferenceLoop_(NDArray &s1, NDArray &s1n, T *syn0row, NDArray&targets,
                              NDArray&negStarters, NDArray&indices, NDArray&codes,
                              const double lr, LongType &randomValue, const int nsRounds, const int vocabSize,
                              const int vectorLength, const int expLength, const int negLength, T *const expTable,
                              const T *negTable, const LongType hsRounds, int t, T *neu1e);

//used for lifecycle tracking in thread locals for error accumulation
template <typename T>
class BufferHolder {
 public:
  BufferHolder(const int vectorLength) {
    neu1e = new T[vectorLength];
  }
  T *neu1e;
  ~BufferHolder() {
    delete[] neu1e;
  }

};


#include <cstdlib>

template <typename T, std::size_t Alignment>
class AlignedAllocator
{
 public:
  typedef T value_type;
  typedef T* pointer;
  typedef const T* const_pointer;
  typedef T& reference;
  typedef const T& const_reference;
  typedef std::size_t size_type;
  typedef std::ptrdiff_t difference_type;

  template <typename U>
  struct rebind { typedef AlignedAllocator<U, Alignment> other; };

  AlignedAllocator() {}

  template <typename U>
  AlignedAllocator(const AlignedAllocator<U, Alignment>&) {}

  pointer address(reference x) const { return &x; }
  const_pointer address(const_reference x) const { return &x; }

  pointer allocate(size_type n, const void* = nullptr)
  {
#if defined(_MSC_VER) || defined(__MINGW32__) || defined(__CYGWIN__)
    void* ptr = this->_aligned_malloc(n * sizeof(T), Alignment);
#else
    void* ptr = nullptr;
    if(posix_memalign(&ptr, Alignment, n * sizeof(T)) != 0)
      ptr = nullptr;
#endif
    if (!ptr)
      THROW_EXCEPTION("Memory allocation failed (std::bad_alloc)");
    return static_cast<pointer>(ptr);
  }

  void deallocate(pointer p, size_type)
  {
#if defined(_MSC_VER)
    _aligned_free(p);
#else
    std::free(p);
#endif
  }

  size_type max_size() const
  {
    return static_cast<size_type>(-1) / sizeof(T);
  }

  void construct(pointer p, const value_type& x)
  {
    ::new(p) value_type(x);
  }

  void destroy(pointer p)
  {
    p->~value_type();
  }
};


// What can be wrong with the arguments of a batch is found before its targets run, on the thread that called: an
// exception that leaves one of the threads of the pool ends the process. With an inference vector the targets pull the
// vector, so no row of syn0 is read for them.
static void checkSkipgramBatch(NDArray &targets, NDArray &negStarters, NDArray &indices, NDArray &codes, NDArray &lr,
                               NDArray &nextRandom, const int nsRounds, const int vocabSize, const bool isInference,
                               const sd::LongType hsRounds) {
  const sd::LongType numTargets = targets.lengthOf();

  if (lr.lengthOf() < numTargets) {
    THROW_EXCEPTION("SkipGram: every target needs a learning rate");
  }

  if (nextRandom.lengthOf() < numTargets) {
    THROW_EXCEPTION("SkipGram: every target needs a random value");
  }

  if (hsRounds > 0 &&
      (indices.sizeAt(0) < numTargets || codes.sizeAt(0) < numTargets || indices.sizeAt(1) < hsRounds)) {
    THROW_EXCEPTION(
        "SkipGram: the indices and codes of a batch need a row for every target, and the indices as many columns as the "
        "codes");
  }

  if (!isInference) {
    for (sd::LongType t = 0; t < numTargets; t++) {
      const int target = targets.e<int>(t);
      if (target < 0 || target >= vocabSize) {
        THROW_EXCEPTION("SkipGram: the target is not a row of syn0");
      }
    }
  }

  if (nsRounds > 0) {
    if (negStarters.lengthOf() < numTargets) {
      THROW_EXCEPTION("SkipGram: negative sampling needs a word to sample for with every target");
    }

    for (sd::LongType t = 0; t < numTargets; t++) {
      const int ngStarter = negStarters.e<int>(t);
      if (ngStarter < 0 || ngStarter >= vocabSize) {
        THROW_EXCEPTION("SkipGram: the word to sample negatives for is not a row of syn1Neg");
      }
    }
  }
}

template <typename T>
void skipgramBatchExec_(NDArray &s0, NDArray &s1, NDArray &s1n, NDArray &vexpTable,NDArray &vnegTable, NDArray &vinfVector,
                        NDArray &targets, NDArray &negStarters, NDArray &indices, NDArray &codes, NDArray &lr,
                        NDArray &nextRandom, const int nsRounds, const int vocabSize, const int vectorLength,
                        const int expLength, const int negLength, const bool preciseMode, const int numThreads,const int iterations,double minLearningRate) {
  const auto expTable = reinterpret_cast<T *>(vexpTable.buffer());
  const auto negTable = reinterpret_cast<T *>(vnegTable.buffer());
  // hierarchic softmax needs its syn1 table: the points and codes of a configuration without it mean nothing
  const bool useHierarchicSoftmax = !codes.isEmpty() && !s1.isEmpty();
  if (useHierarchicSoftmax && (codes.rankOf() != 2 || indices.rankOf() != 2)) {
    THROW_EXCEPTION("SkipGram: the indices and codes of a batch are matrices");
  }

  const sd::LongType hsRounds = useHierarchicSoftmax ? codes.sizeAt(1) : 0;
  checkSkipgramBatch(targets, negStarters, indices, codes, lr, nextRandom, nsRounds, vocabSize, !vinfVector.isEmpty(),
                     hsRounds);

  //training
  if(vinfVector.isEmpty()) {
    const sd::LongType  targetsLen = targets.lengthOf();

    auto func = PRAGMA_THREADS_FOR {
      for (auto t = start; t < stop; t+= increment) {
        doSkipGramLoop_(s0, s1, s1n, vinfVector, targets, negStarters, indices, codes, lr, nextRandom, nsRounds,
                        vocabSize, vectorLength, expLength, negLength, expTable, negTable, hsRounds, t);
      }
    };


    int chunkSize = 1024;
    // numThreads comes from the Java-side workers configuration.  When 0, parallel_tad will
    // resolve it to maxMasterThreads() (all cores).  Passing it explicitly here allows callers
    // to request serial execution (numThreads=1) which eliminates Hogwild!-style data races on
    // shared Huffman-tree syn1 nodes and is essential for correct convergence on small vocabularies.
    if(targetsLen < chunkSize) {
      samediff::Threads::parallel_tad(func,0,targetsLen,1,numThreads);
    } else {
      // the last chunk is shorter than the others when the targets are not a multiple of the chunk size
      int chunks = (targetsLen + chunkSize - 1) / chunkSize;
      for(int i = 0; i < chunks; i++) {
        int start = i * chunkSize;
        int potentialEnd = start + chunkSize;
        int end = sd::math::sd_min(targetsLen,potentialEnd);
        samediff::Threads::parallel_tad(func,start,end,1,numThreads);
      }

    }



  } else { //inference
    // The targets pull the inference vector, every one of them against the same vector, and their errors add up once per
    // iteration; syn1 and syn1Neg are only read. The learning rate of every target decays after each iteration:
    // alpha = (alpha - minLearningRate) / (iterations - iteration) + minLearningRate
    const sd::LongType numTargets = targets.lengthOf();
    auto vec = reinterpret_cast<T *>(vinfVector.buffer());

    std::vector<T> neu1e(static_cast<size_t>(numTargets) * static_cast<size_t>(vectorLength));
    std::vector<double> lrs(numTargets);
    // the random values of a target go on from one iteration to the next, as they do in a round of a single target
    std::vector<sd::LongType> randoms(numTargets);
    for(sd::LongType t = 0; t < numTargets; t++) {
      lrs[t] = lr.e<double>(t);
      randoms[t] = nextRandom.e<sd::LongType>(t);
    }

    for(int curr = 0; curr < iterations; curr++) {
      // the targets read the vector and write their own error: they can run side by side
      auto func = PRAGMA_THREADS_FOR {
        for (auto t = start; t < stop; t+= increment) {
          T *currNeu1e = neu1e.data() + t * vectorLength;
          std::fill_n(currNeu1e, vectorLength, T(0));

          doSkipGramInferenceLoop_(s1, s1n, vec,
                                   targets, negStarters,
                                   indices,
                                   codes,
                                   lrs[t],
                                   randoms[t],
                                   nsRounds,
                                   vocabSize,
                                   vectorLength,
                                   expLength,
                                   negLength,
                                   expTable,
                                   negTable,
                                   hsRounds,
                                   t,
                                   currNeu1e);
        }
      };
      samediff::Threads::parallel_tad(func, 0, numTargets, 1, numThreads);

      for(sd::LongType t = 0; t < numTargets; t++) {
        const T *currNeu1e = neu1e.data() + t * vectorLength;
        for(int j = 0; j < vectorLength; j++) {
          vec[j] += currNeu1e[j];
        }
      }

      for(sd::LongType t = 0; t < numTargets; t++) {
        lrs[t] = ((lrs[t] - static_cast<double>(minLearningRate)) / static_cast<double>(iterations - curr)) + static_cast<double>(minLearningRate);
      }
    }

  }// end else
}




template <typename T>
void doSkipGramInferenceLoop_(NDArray &s1, NDArray &s1n, T *syn0row, NDArray&targets,
                              NDArray&negStarters, NDArray&indices, NDArray&codes,
                              const double alpha, LongType &randomValue, const int nsRounds, const int vocabSize,
                              const int vectorLength, const int expLength, const int negLength, T *const expTable,
                              const T *negTable, const LongType hsRounds, int t, T *neu1e) {

  if(t >= targets.lengthOf()) {
    std::string errorMessage;
    errorMessage += "Target index is greater than number of targets ";
    errorMessage += std::to_string(t);
    errorMessage += " >= ";
    errorMessage += std::to_string(targets.lengthOf());
    THROW_EXCEPTION(errorMessage.c_str())
  }

  // The inference vector is all that learns, so the order of the rounds does not matter. Every target works on its own
  // error vector, one round after the other: the rounds of a target share neu1e and must not run side by side.
  if(nsRounds > 0) {
    const int nsStarter = negStarters.e<int>(t);
    int irow = nsStarter;

    for (int r = 0; r < nsRounds + 1; r++) {
      if (r != 0) {
        randomValue = randomValue * (unsigned long long)25214903917 + 11;
        auto idx = math::sd_abs<LongType,LongType>((randomValue >> 16) % negLength);
        irow = idx >= negLength ? -1 : static_cast<int>(negTable[idx]);

        if (irow < 0 || irow >= vocabSize) {
          irow = static_cast<int>(static_cast<unsigned long long>(randomValue) % (vocabSize - 1)) + 1;
        }
        if (irow == nsStarter) continue;
      }

      nSampling_<T>(syn0row, s1n.bufferWithOffset(static_cast<LongType>(irow) * vectorLength), expTable, neu1e, alpha,
                    vectorLength, r == 0 ? 1 : 0, expLength, true);
    }
  }

  for (LongType e = 0; e < hsRounds; e++) {
    const int currRow = indices.e<int>(t,e);
    const int code = codes.e<int>(t,e);

    // the code of a padding is -1
    if(code < 0 || currRow < 0 || currRow >= vocabSize) {
      continue;
    }

    T *syn1row = (T *) s1.bufferWithOffset(static_cast<LongType>(currRow) * vectorLength);
    hSoftmax_<T>(syn0row,syn1row,expTable,neu1e,alpha,vectorLength,code,expLength,true);
  }



}


template <typename T>
void doSkipGramLoop_(NDArray &s0, NDArray &s1, NDArray &s1n, NDArray &vinfVector, NDArray&targets,
                     NDArray&negStarters, NDArray&indices, NDArray&codes, NDArray&lr,
                     NDArray&nextRandom, const int nsRounds, const int vocabSize, const int vectorLength,
                     const int expLength, const int negLength, T *const expTable, const T *negTable,
                     const LongType hsRounds, int t) {

  if(t >= lr.lengthOf()) {
    std::string errorMessage;
    errorMessage += "Target index is greater than number of learning rates ";
    errorMessage += std::to_string(t);
    errorMessage += " >= ";
    errorMessage += std::to_string(lr.lengthOf());
    THROW_EXCEPTION(errorMessage.c_str());

  }

  if(t >= targets.lengthOf()) {
    std::string errorMessage;
    errorMessage += "Target index is greater than number of targets ";
    errorMessage += std::to_string(t);
    errorMessage += " >= ";
    errorMessage += std::to_string(targets.lengthOf());
    THROW_EXCEPTION(errorMessage.c_str())
  }

  // the target and the word to sample negatives for were checked before the threads ran
  auto target = targets.e<int>(t);

  std::vector<T> neu1eBuffer(vectorLength);
  T *neu1e = neu1eBuffer.data();
  memset(neu1e, 0, vectorLength * sizeof(T));

  auto alpha = lr.e<double>(t);

  LongType randomValue = nextRandom.e<LongType>(t);
  auto syn0row = vinfVector.isEmpty() ?  reinterpret_cast<T *>(s0.bufferWithOffset(static_cast<LongType>(target) * vectorLength)) : reinterpret_cast<T *>(vinfVector.buffer());
  if(hsRounds > 0) {
    for (LongType e = 0; e < hsRounds; e++) {
      int currRow = indices.e<int>(t,e);
      int code = codes.e<int>(t,e);
      //codes are only 0 and 1, -1 are placeholders for invalid codes
      //the codes matrix is padded with extra values at time of allocation
      //this is due to the code rows effectively being a ragged matrix (rows have different shapes)
      if(code < 0 || currRow < 0 || currRow >= vocabSize)  {
        continue;
      }

      T *syn1row = (T *) s1.bufferWithOffset(static_cast<LongType>(currRow) * vectorLength);
      hSoftmax_<T>(syn0row,syn1row,expTable,neu1e,alpha,vectorLength,code,expLength,!vinfVector.isEmpty());

    }
  }

  if(nsRounds > 0) {
    int irow = negStarters.e<int>(t);
    int nsStarter = irow;
    for (int r = 0; r < nsRounds + 1; r++) {
      if (r == 0) {
        // target is known in advance
      } else {
        randomValue = randomValue * (unsigned long long)25214903917 + 11;
        auto idx = math::sd_abs<LongType,LongType>((randomValue >> 16) % negLength);
        irow = idx >= negLength ? -1 : static_cast<int>(negTable[idx]);

        if (irow < 0 || irow >= vocabSize) {
          irow = static_cast<int>(static_cast<unsigned long long>(randomValue) % (vocabSize - 1)) + 1;
        }

        if (irow == nsStarter) continue;
      }

      nSampling_<T>(syn0row, s1n.bufferWithOffset(static_cast<LongType>(irow) * vectorLength), expTable, neu1e, alpha,
                    vectorLength, r == 0 ? 1 : 0, expLength, !vinfVector.isEmpty());
    }
  }
      PRAGMA_OMP_SIMD
  for (int e = 0; e < vectorLength; e++) {
    syn0row[e] += neu1e[e];
  }
}

BUILD_SINGLE_TEMPLATE( void skipgramBatchExec_,
                      (NDArray & s0, NDArray &s1, NDArray &s1n, NDArray &vexpTable, NDArray &vnegTable, NDArray &vinfVector,
                          NDArray &targets, NDArray &negStarters, NDArray &indices, NDArray &codes, NDArray &lr,
                          NDArray &nextRandom, const int nsRounds, const int vocabSize, const int vectorLength,
                          const int expLength, const int negLength, const bool preciseMode, const int numThreads,const int iterations,double minLearningRate),
                      SD_NATIVE_FLOAT_TYPES);

template <typename T>
void doCbowLoop_(NDArray &s0, NDArray &s1, NDArray &s1n, NDArray&negStarters, NDArray&indices,
                 NDArray&codes, NDArray&lr, NDArray&nextRandom, NDArray&nLabels,
                 const int nsRounds, const int vocabSize, const int vectorLength, const int expLength,
                 const int negLength, const bool trainWords, T *const expTable, const T *negTable, T *infVector,
                 const int contextWidth, const int *bContext, const int *bLocker, const int *bStarters,
                 unsigned long long *bRandoms, const LongType numIndices, int t, const double alpha);

// What can be wrong with the arguments of a batch is found before its windows run, on the thread that called: an
// exception that leaves one of the threads of the pool ends the process. A window is a row of context; the words of the
// windows, the points of the hierarchic softmax and the words to sample negatives for have to be rows of the tables.
static void checkCbowBatch(NDArray &context, NDArray &lockedWords, NDArray &targets, NDArray &negStarters,
                           NDArray &indices, NDArray &codes, NDArray &lr, NDArray &nextRandom, NDArray &nLabels,
                           const int nsRounds, const int vocabSize, const sd::LongType numIndices) {
  const sd::LongType numWindows = targets.lengthOf();
  const sd::LongType contextWidth = context.sizeAt(1);

  if (numWindows > context.sizeAt(0)) {
    THROW_EXCEPTION("CBOW: every target needs a window");
  }

  if (lr.lengthOf() < numWindows) {
    THROW_EXCEPTION("CBOW: every window needs a learning rate");
  }

  if (!nLabels.isEmpty() && nLabels.lengthOf() < numWindows) {
    THROW_EXCEPTION("CBOW: every window needs a number of labels");
  }

  if (!lockedWords.isEmpty() && lockedWords.lengthOf() < numWindows * contextWidth) {
    THROW_EXCEPTION("CBOW: the locked words need a status for every word of the windows");
  }

  for (sd::LongType t = 0; t < numWindows; t++) {
    for (sd::LongType c = 0; c < contextWidth; c++) {
      // the values that pad a window are negative
      if (context.e<int>(t, c) >= vocabSize) THROW_EXCEPTION("ContextID can't be >= vocab size");
    }
  }

  if (numIndices > 0) {
    if (codes.rankOf() != 2 || indices.sizeAt(0) < numWindows || codes.sizeAt(0) < numWindows ||
        codes.sizeAt(1) < numIndices) {
      THROW_EXCEPTION(
          "CBOW: the indices and codes of a batch need a row for every window, and the codes as many columns as the "
          "indices");
    }

    for (sd::LongType t = 0; t < numWindows; t++) {
      for (sd::LongType i = 0; i < numIndices; i++) {
        // we're skipping padded values: the code of a padding is -1
        if (indices.e<int>(t, i) < 0 || codes.e<int>(t, i) < 0) continue;

        if (indices.e<int>(t, i) >= vocabSize) THROW_EXCEPTION("Index can't be > vocab size");
      }
    }
  }

  if (nsRounds > 0 && !negStarters.isEmpty()) {
    if (negStarters.lengthOf() < numWindows || nextRandom.lengthOf() < numWindows) {
      THROW_EXCEPTION("CBOW: negative sampling needs a word to sample for and a random value with every window");
    }

    for (sd::LongType t = 0; t < numWindows; t++) {
      const int ngStarter = negStarters.e<int>(t);
      if (ngStarter < 0 || ngStarter >= vocabSize) {
        THROW_EXCEPTION("CBOW: the word to sample negatives for is not a row of syn1Neg");
      }
    }
  }
}

template <typename T>
void cbowBatchExec_(NDArray &s0, NDArray &s1, NDArray &s1n, NDArray &vexpTable, NDArray &vnegTable, NDArray &vinfVector,
                    NDArray &context, NDArray &lockedWords, NDArray &targets, NDArray &negStarters, NDArray &indices,
                    NDArray &codes, NDArray &lr, NDArray &nextRandom, NDArray &nLabels, const int nsRounds,
                    const int vocabSize, const int vectorLength, const int expLength, const int negLength,
                    const bool trainWords, const int numThreads,double minLearningRate,int iterations) {

  const auto expTable = vexpTable.bufferAsT<T>();
  const auto negTable = vnegTable.bufferAsT<T>();
  const auto infVector = vinfVector.bufferAsT<T>();

  // a window is a row of context, and there is one for every target
  const sd::LongType numTargets = targets.lengthOf();
  const int contextWidth = context.sizeAt(1);

  // hierarchic softmax needs its syn1 table: the points and codes of a configuration without it mean nothing
  const sd::LongType numIndices = (indices.isEmpty() || codes.isEmpty() || s1.isEmpty()) ? 0 : indices.sizeAt(1);
  checkCbowBatch(context, lockedWords, targets, negStarters, indices, codes, lr, nextRandom, nLabels, nsRounds,
                 vocabSize, numIndices);

  // The windows, their locked words, the words to sample negatives for and their random values are read once, as the
  // integers they are: the windows of an inference vector are visited in every iteration.
  std::vector<int> windows(static_cast<size_t>(numTargets) * static_cast<size_t>(contextWidth));
  for (sd::LongType t = 0; t < numTargets; t++) {
    for (int c = 0; c < contextWidth; c++) {
      windows[t * contextWidth + c] = context.e<int>(t, c);
    }
  }

  std::vector<int> lockedBuffer;
  if (!lockedWords.isEmpty()) {
    lockedBuffer.resize(windows.size());
    for (sd::LongType t = 0; t < numTargets; t++) {
      for (int c = 0; c < contextWidth; c++) {
        lockedBuffer[t * contextWidth + c] = lockedWords.e<int>(t, c);
      }
    }
  }

  std::vector<int> startersBuffer;
  std::vector<unsigned long long> randomsBuffer;
  if (nsRounds > 0 && !negStarters.isEmpty()) {
    startersBuffer.resize(numTargets);
    randomsBuffer.resize(numTargets);
    for (sd::LongType t = 0; t < numTargets; t++) {
      startersBuffer[t] = negStarters.e<int>(t);
      randomsBuffer[t] = static_cast<unsigned long long>(nextRandom.e<sd::LongType>(t));
    }
  }

  // no word is locked when no locked words are given
  const int *bContext = windows.data();
  const int *bLocker = lockedBuffer.empty() ? nullptr : lockedBuffer.data();
  const int *bStarters = startersBuffer.empty() ? nullptr : startersBuffer.data();
  unsigned long long *bRandoms = randomsBuffer.empty() ? nullptr : randomsBuffer.data();

  std::vector<double> lrs(numTargets);
  for (sd::LongType t = 0; t < numTargets; t++) {
    lrs[t] = lr.e<double>(t);
  }

  if(vinfVector.isEmpty()) {
    auto func = PRAGMA_THREADS_FOR {
      for (auto t = start; t < stop; t+= increment) {
        doCbowLoop_(s0, s1, s1n, negStarters, indices, codes, lr, nextRandom, nLabels, nsRounds, vocabSize,
                    vectorLength, expLength, negLength, trainWords, expTable, negTable, infVector, contextWidth,
                    bContext, bLocker, bStarters, bRandoms, numIndices, t, lrs[t]);

      }
    };


    int targetsLen = numTargets;
    int chunkSize = 1024;
    // numThreads comes from the Java-side workers configuration, as it does for skipgram: 1 asks for serial execution,
    // 0 for all cores
    if(targetsLen < chunkSize) {
      samediff::Threads::parallel_tad(func,0,targetsLen,1,numThreads);
    } else {
      // the last chunk is shorter than the others when the targets are not a multiple of the chunk size
      int chunks = (targetsLen + chunkSize - 1) / chunkSize;
      for(int i = 0; i < chunks; i++) {
        int start = i * chunkSize;
        int potentialEnd = start + chunkSize;
        int end = sd::math::sd_min(targetsLen,potentialEnd);
        samediff::Threads::parallel_tad(func,start,end,1,numThreads);
      }
    }




  } else {
    // Inference: the targets are visited in order and every one of them moves the inference vector, which is all that
    // learns. The learning rate of every target decays after each iteration, and the random values of a target go on from
    // one iteration to the next:
    // alpha = (alpha - minLearningRate) / (iterations - iteration) + minLearningRate
    for(int iteration = 0; iteration < iterations; iteration++) {
      for (sd::LongType t = 0; t < numTargets; t++) {
        doCbowLoop_(s0, s1, s1n, negStarters, indices, codes, lr, nextRandom, nLabels, nsRounds, vocabSize,
                    vectorLength, expLength, negLength, trainWords, expTable, negTable, infVector, contextWidth,
                    bContext, bLocker, bStarters, bRandoms, numIndices, t, lrs[t]);
      }

      for(sd::LongType t = 0; t < numTargets; t++) {
        lrs[t] = ((lrs[t] - static_cast<double>(minLearningRate)) / static_cast<double>(iterations - iteration)) + static_cast<double>(minLearningRate);
      }
    }
  }


}
template <typename T>
void doCbowLoop_(NDArray &s0, NDArray &s1, NDArray &s1n, NDArray&negStarters, NDArray&indices,
                 NDArray&codes, NDArray&lr, NDArray&nextRandom, NDArray&nLabels,
                 const int nsRounds, const int vocabSize, const int vectorLength, const int expLength,
                 const int negLength, const bool trainWords, T *const expTable, const T *negTable, T *infVector,
                 const int contextWidth, const int *bContext, const int *bLocker, const int *bStarters,
                 unsigned long long *bRandoms, const LongType numIndices, int t, const double alpha) {
  std::vector<T> neu1Buffer(vectorLength);
  std::vector<T> neu1eBuffer(vectorLength);
  T *neu1 = neu1Buffer.data();
  T *neu1e = neu1eBuffer.data();

  // optionally we nullify temp arrays after successful (and on first) cycle
  memset(neu1, 0, sizeof(T) * vectorLength);
  memset(neu1e, 0, sizeof(T) * vectorLength);

  auto numLabels = nLabels.isEmpty() ? 0 : nLabels.e<int>(t);

  int actualContext = 0;

  // building neu1 for current window
  for (int c = 0; c < contextWidth; c++) {
    // getting next context word
    auto cContext = bContext[c + (t * contextWidth)];

    // skipping padded values
    if (cContext < 0) continue;

    T *syn0word = (T *) s0.bufferWithOffset(static_cast<LongType>(cContext) * vectorLength);

    for (int i = 0; i < vectorLength; i++) neu1[i] += syn0word[i];

    actualContext++;
  }

  // for inference the inference vector is one more member of the window
  if (infVector != nullptr) {
    for (int i = 0; i < vectorLength; i++) neu1[i] += infVector[i];

    actualContext++;
  }

  if (actualContext > 1) {
    for (int i = 0; i < vectorLength; i++) neu1[i] /= actualContext;
  }

  // hierarchic softmax step
  if (!indices.isEmpty()) {
    for (LongType i = 0; i < numIndices; i++) {
      const int cIndex = indices.e<int>(t,i);
      const int cCode = codes.e<int>(t,i);

      // we're skipping padded values: the code of a padding is -1
      if (cIndex < 0 || cCode < 0) continue;

      hSoftmax_<T>(neu1, s1.bufferasTWithOffset<T>(static_cast<LongType>(cIndex) * vectorLength), expTable, neu1e, alpha, vectorLength, cCode, expLength,
                   infVector != nullptr);
    }
  }

  // negative sampling step
  if (!negStarters.isEmpty() && nsRounds > 0) {
    int irow = bStarters[t];
    const int nsStarter = irow;
    unsigned long long &randomValue = bRandoms[t];

    for (int r = 0; r < nsRounds + 1; r++) {
      // we're skipping rng on 0 step
      if (r != 0) {
        randomValue = randomValue * (unsigned long long)25214903917 + 11;
        auto idx = math::sd_abs<LongType,LongType>((randomValue >> 16) % negLength);
        irow = idx >= negLength ? -1 : static_cast<int>(negTable[idx]);

        if (irow < 0 || irow >= vocabSize) irow = randomValue % (vocabSize - 1) + 1;
        if (irow == nsStarter) continue;

        nSampling_<T>(neu1, s1n.bufferWithOffset(static_cast<LongType>(irow) * vectorLength), expTable, neu1e, alpha, vectorLength,
                      r == 0 ? 1 : 0, expLength, infVector != nullptr);
      } else {
        nSampling_<T>(neu1, s1n.bufferWithOffset(static_cast<LongType>(irow) * vectorLength), expTable, neu1e, alpha, vectorLength,
                      r == 0 ? 1 : 0, expLength, infVector != nullptr);
      }

    }
  }

  if (infVector == nullptr) {
    // if we're skipping labels
    int starter = trainWords == 1 ? 0 : contextWidth - numLabels;
    if (starter < 0) starter = 0;

    // applying previously averaged results
    for (int c = starter; c < contextWidth; c++) {
      // getting context
      auto cContext = bContext[c + (t * contextWidth)];
      // no word is locked when no locked words are given
      const int cLock = bLocker != nullptr ? bLocker[c + (t * contextWidth)] : 0;

      // skipping padded values
      if (cContext < 0 || cLock == 1) continue;

      // one word from context
      T *syn0word = (T *) s0.bufferWithOffset(static_cast<LongType>(cContext) * vectorLength);
      PRAGMA_OMP_SIMD
      for (int i = 0; i < vectorLength; i++) syn0word[i] += neu1e[i];
    }
  } else {
    // inference: the inference vector is all that learns
    for (int i = 0; i < vectorLength; i++) infVector[i] += neu1e[i];
  }
}
BUILD_SINGLE_TEMPLATE( void cbowBatchExec_,
                      (NDArray & s0, NDArray &s1, NDArray &s1n, NDArray &vexpTable, NDArray &vnegTable, NDArray &vinfVector,
                          NDArray &context, NDArray &lockedWords, NDArray &targets, NDArray &negStarters, NDArray &indices,
                          NDArray &codes, NDArray &lr, NDArray &nextRandom, NDArray &nLabels, const int nsRounds,
                          const int vocabSize, const int vectorLength, const int expLength, const int negLength,
                          const bool trainWords, const int numThreads,double minLearningRate,const int iterations),
                      SD_NATIVE_FLOAT_TYPES);



void skipgramInference(NDArray &syn0, NDArray &syn1, NDArray &syn1Neg, NDArray &expTable, NDArray &negTable, int target,
                       int ngStarter, int nsRounds, NDArray &indices, NDArray &codes, double alpha, sd::LongType randomValue,
                       NDArray &inferenceVector, const bool preciseMode, const int numWorkers,double minLearningRate,const int iterations) {
  auto xType = syn0.dataType();
  // hierarchic softmax needs its syn1 table: the points and codes of a configuration without it mean nothing
  auto hsRounds = syn1.isEmpty() ? 0 : codes.lengthOf();
  BUILD_SINGLE_SELECTOR(
      xType, skipgram_,
      (syn0.buffer(), syn1.buffer(), syn1Neg.buffer(), expTable.buffer(), negTable.buffer(), inferenceVector.buffer(),
          target, ngStarter,
          indices, codes, alpha,
          randomValue, hsRounds, nsRounds, (int)syn0.sizeAt(0), (int)syn0.sizeAt(1),
          (int)expTable.lengthOf(), (int)negTable.lengthOf(),minLearningRate,iterations),
      SD_NATIVE_FLOAT_TYPES);
}


void cbowInference(NDArray &syn0, NDArray &syn1, NDArray &syn1Neg, NDArray &expTable, NDArray &negTable, int target,
                   int ngStarter, int nsRounds, NDArray &context, NDArray &lockedWords, NDArray &indices, NDArray &codes,
                   double alpha, sd::LongType randomValue, int numLabels, NDArray &inferenceVector, const bool trainWords,
                   int numWorkers,int iterations,double minLearningRate) {
  auto xType = syn0.dataType();
  // hierarchic softmax needs its syn1 table: the points and codes of a configuration without it mean nothing
  auto hsRounds = syn1.isEmpty() ? 0 : codes.lengthOf();
  BUILD_SINGLE_SELECTOR(
      xType, cbow_,
      (syn0, syn1, syn1Neg, expTable, negTable, inferenceVector,
          target, ngStarter,
          context, lockedWords,
          indices, codes, alpha,
          randomValue, (int)context.lengthOf(), hsRounds, nsRounds, (int)syn0.sizeAt(0),
          (int)syn0.sizeAt(1), (int)expTable.lengthOf(), (int)negTable.lengthOf(),
          numLabels, trainWords,minLearningRate,iterations),
      SD_NATIVE_FLOAT_TYPES);
}

void skipgram(NDArray &syn0, NDArray &syn1, NDArray &syn1Neg, NDArray &expTable, NDArray &negTable, NDArray &target,
              NDArray &ngStarter, int nsRounds, NDArray &indices, NDArray &codes, NDArray &alpha, NDArray &randomValue,
              NDArray &inferenceVector, const bool preciseMode, const int numWorkers,const int iterations,double minLearningRate) {
  auto xType = syn0.dataType();

  // single round case
  if ((ngStarter.isScalar() && !ngStarter.isEmpty()) || (target.isScalar() && !target.isEmpty())) {
    // hierarchic softmax needs its syn1 table: the points and codes of a configuration without it mean nothing
    auto hsRounds = syn1.isEmpty() ? 0 : codes.lengthOf();

    BUILD_SINGLE_SELECTOR(
        xType, skipgram_,
        (syn0.buffer(), syn1.buffer(), syn1Neg.buffer(), expTable.buffer(), negTable.buffer(), inferenceVector.buffer(),
            target.isEmpty() ? -1 : target.e<int>(0), ngStarter.isEmpty() ? -1 : ngStarter.e<int>(0),
            indices, codes, alpha.e<double>(0),
            randomValue.e<sd::LongType>(0), hsRounds, nsRounds, (int)syn0.sizeAt(0), (int)syn0.sizeAt(1),
            (int)expTable.lengthOf(), (int)negTable.lengthOf(),minLearningRate,iterations),
        SD_NATIVE_FLOAT_TYPES);
  } else if (ngStarter.isVector() || target.isVector()) {
    // batch mode
    BUILD_SINGLE_SELECTOR(xType, skipgramBatchExec_,
                          (syn0, syn1, syn1Neg, expTable, negTable, inferenceVector, target, ngStarter,
                              indices, codes, alpha, randomValue, nsRounds, syn0.sizeAt(0), syn0.sizeAt(1),
                              expTable.lengthOf(), negTable.lengthOf(), preciseMode, numWorkers,iterations,minLearningRate),
                          SD_NATIVE_FLOAT_TYPES);
  } else
    THROW_EXCEPTION("SkipGram: target must have rank 0 or 1");
}

void cbow(NDArray &syn0, NDArray &syn1, NDArray &syn1Neg, NDArray &expTable, NDArray &negTable, NDArray &target,
          NDArray &ngStarter, int nsRounds, NDArray &context, NDArray &lockedWords, NDArray &indices, NDArray &codes,
          NDArray &alpha, NDArray &randomValue, NDArray &numLabels, NDArray &inferenceVector, const bool trainWords,
          int numWorkers,double minLearningRate,const int iterations) {
  auto xType = syn0.dataType();

  if ((context.rankOf() == 0 || context.rankOf() == 1) && (indices.rankOf() == 1 || indices.rankOf() == 0)) {
    // hierarchic softmax needs its syn1 table: the points and codes of a configuration without it mean nothing
    auto hsRounds = syn1.isEmpty() ? 0 : codes.lengthOf();

    //convert every inline parameter below in to a variable
    BUILD_SINGLE_SELECTOR(
        xType, cbow_,
        (syn0,
         syn1,
         syn1Neg,
         expTable,
         negTable,
         inferenceVector,
            target.isEmpty() ? -1 : target.e<int>(0),
            ngStarter.isEmpty() ? -1 : ngStarter.e<int>(0),
            context,
            lockedWords,
            indices,
            codes,
            alpha.isEmpty() ? 0.025 : alpha.e<double>(0),
            randomValue.isEmpty() ? -1 : randomValue.e<sd::LongType>(0),
            (int)context.lengthOf(),
            hsRounds,
            nsRounds,
            (int)syn0.sizeAt(0),
            (int)syn0.sizeAt(1),
            expTable.isEmpty() ? 0 : (int)expTable.lengthOf(),
            negTable.isEmpty() ? 0 : (int)negTable.lengthOf(),
            numLabels.isEmpty() ? 0 : numLabels.e<int>(0),
            trainWords,minLearningRate,iterations),
        SD_NATIVE_FLOAT_TYPES);
  } else if (context.rankOf() == 2 && (indices.rankOf() == 2 || indices.isEmpty())) {
    // batch mode; a configuration without hierarchic softmax passes no indices and no codes
    BUILD_SINGLE_SELECTOR(
        xType, cbowBatchExec_,
        (syn0, syn1, syn1Neg, expTable, negTable, inferenceVector, context, lockedWords, target, ngStarter,
            indices, codes, alpha, randomValue, numLabels, nsRounds, syn0.sizeAt(0), syn0.sizeAt(1), expTable.lengthOf(),
            negTable.isEmpty() ? 0 : negTable.lengthOf(), trainWords, numWorkers,minLearningRate,iterations),
        SD_NATIVE_FLOAT_TYPES);
  } else
    THROW_EXCEPTION("CBOW: context must have rank 0/1 or 2");
}
}  // namespace helpers
}  // namespace ops
}  // namespace sd
