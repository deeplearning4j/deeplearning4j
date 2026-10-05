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
// random_shuffle for every backend: the permutation is built on the host with MergeShuffle
// ("MergeShuffle: A Very Fast, Parallel Random Permutation Algorithm", https://arxiv.org/abs/1508.03167), then the
// backend's gather applies it along dimension 0.
//
#include <system/op_boilerplate.h>
#include <execution/Threads.h>
#include <graph/RandomGenerator.h>
#include <ops/declarable/helpers/gather.h>
#include <ops/declarable/helpers/transforms.h>

#if NOT_EXCLUDED(OP_random_shuffle)
namespace sd {
namespace ops {
namespace helpers {

// Fisher-Yates over perm[0, len), drawing rng indices from ind
static void fisherYates(graph::RandomGenerator& rng, LongType* perm, const LongType len, LongType ind) {
  for (LongType i = len - 1; i > 0; --i) {
    const LongType j = rng.relativeLong(ind++) % (i + 1);
    if (i != j) math::sd_swap<LongType>(perm[i], perm[j]);
  }
}

// mutual shuffle of the adjacent shuffled ranges perm[0, len1) and perm[len1, totLen), drawing rng indices from ind
// (at most totLen + 1 draws)
static void mergeShuffle(graph::RandomGenerator& rng, LongType* perm, const LongType len1, const LongType totLen,
                         LongType ind) {
  LongType beg = 0;
  LongType mid = len1;

  while (true) {
    if (rng.relativeLong(ind++) % 2) {
      if (mid == totLen) break;
      math::sd_swap<LongType>(perm[beg], perm[mid++]);
    } else {
      if (beg == mid) break;
    }
    ++beg;
  }

  while (beg < totLen) {
    const LongType j = rng.relativeLong(ind++) % (beg + 1);
    if (beg != j) math::sd_swap<LongType>(perm[beg], perm[j]);
    ++beg;
  }
}

// The permutation of len positions: perm[i] is the position whose element moves to i. Chunks of at most 2^22 positions
// are shuffled with Fisher-Yates (the chunk at offset o draws from rng index o), then merged pairwise level by level
// (level l, offset o draws from (len + 1) * l + o, so no two steps share a draw). rng is rewound past every index drawn.
static void shufflePermutation(graph::RandomGenerator& rng, const LongType len, LongType* perm) {
  for (LongType i = 0; i < len; ++i) perm[i] = i;

  const LongType threshold = LongType(1) << 22;
  int power = 0;
  while ((len >> power) > threshold) ++power;
  const LongType numChunks = LongType(1) << power;

  auto funcFisherYates = PRAGMA_THREADS_FOR {
    for (auto c = start; c < stop; ++c) {
      const LongType offset = (len * c) >> power;
      const LongType chunkLen = ((len * (c + 1)) >> power) - offset;
      fisherYates(rng, perm + offset, chunkLen, offset);
    }
  };
  samediff::Threads::parallel_for(funcFisherYates, 0, numChunks);

  // level l merges pairs of adjacent ranges of 2^(l - 1) chunks each
  for (LongType half = 1, level = 1; half < numChunks; half += half, ++level) {
    auto funcMerge = PRAGMA_THREADS_FOR {
      for (auto c = start; c < stop; c += increment) {
        const LongType offset = (len * c) >> power;
        const LongType len1 = ((len * (c + half)) >> power) - offset;
        const LongType totLen = ((len * (c + 2 * half)) >> power) - offset;
        mergeShuffle(rng, perm + offset, len1, totLen, (len + 1) * level + offset);
      }
    };
    samediff::Threads::parallel_for(funcMerge, 0, numChunks, 2 * half);
  }

  // one past the largest index drawn: Fisher-Yates alone draws [0, len - 1)
  rng.rewindH(power == 0 ? len - 1 : (len + 1) * (power + 1));
}

void randomShuffle(LaunchContext* context, NDArray& input, NDArray& output, graph::RandomGenerator& rng,
                   const bool isInplace) {
  // TensorFlow shuffles along dimension 0: a vector's elements, a matrix's rows, ...
  const LongType firstDim = input.sizeAt(0);
  if (input.lengthOf() <= 1 || firstDim <= 1) {
    if (!isInplace) output.assign(&input);
    return;
  }

  std::vector<LongType> permShape = {firstDim};
  NDArray perm('c', permShape, DataType::INT64, context);
  NDArray::preparePrimaryUse({&perm}, {});
  shufflePermutation(rng, firstDim, perm.bufferAsT<LongType>());
  NDArray::registerPrimaryUse({&perm}, {});

  // in place, gather from a copy: output is the input
  NDArray* source = isInplace ? input.dup(input.ordering()) : &input;
  std::vector<LongType> axis = {0};
  gather(context, source, &perm, &output, axis);
  if (isInplace) delete source;
}

}  // namespace helpers
}  // namespace ops
}  // namespace sd
#endif
