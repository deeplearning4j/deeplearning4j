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

#ifndef LIBND4J_HELPER_RANDOM_H
#define LIBND4J_HELPER_RANDOM_H

#ifdef __CUDACC__
#include <curand.h>
#endif
#include <helpers/helper_generator.h>

#ifndef __CUDACC__
#include <mutex>

#endif

namespace sd {

namespace random {

template <typename T>
class RandomHelper {
 private:
  IGenerator *generator;
  RandomBuffer *buffer;

 public:
  SD_HOST_DEVICE RandomHelper(IGenerator *generator) {
    this->generator = generator;
    this->buffer = generator->getBuffer();
  }

  SD_HOST_DEVICE RandomHelper(RandomBuffer *buffer) { this->buffer = buffer; }

  /**
   * This method returns random int in range [0..SD_MAX_INT]
   * @return
   */
  SD_INLINE SD_DEVICE int nextInt() {
    int r = (int)nextUInt();
    return r < 0 ? -1 * r : r;
  };

  SD_INLINE SD_DEVICE uint64_t nextUInt() { return buffer->getNextElement(); }

  /**
   * This method returns random int in range [0..to]
   * @param to
   * @return
   */
  SD_INLINE SD_DEVICE int nextInt(int to) {
    int r = nextInt();
    int m = to - 1;
    if ((to & m) == 0)  // i.e., bound is a power of 2
      r = (int)((to * (long)r) >> 31);
    else {
      for (int u = r; u - (r = u % to) + m < 0; u = nextInt())
        ;
    }
    return r;
  };

  /**
   * This method returns random int in range [from..to]
   * @param from
   * @param to
   * @return
   */
  SD_INLINE SD_DEVICE int nextInt(int from, int to) {
    if (from == 0) return nextInt(to);

    return from + nextInt(to - from);
  };

  /**
   * This method returns random T in range of [0..SD_MAX_FLOAT]
   * @return
   */
  SD_INLINE SD_DEVICE T nextMaxT() {
    T rnd = (T)buffer->getNextElement();
    return rnd < 0 ? -1 * rnd : rnd;
  };

  /**
   * This method returns random T in range of [0..1]
   * @return
   */
  SD_INLINE SD_DEVICE T nextT() { return (T)nextUInt() / (T)DataTypeUtils::max<LongType>(); }

  /**
   * This method returns random T in range of [0..to]
   * @param to
   * @return
   */
  SD_INLINE SD_DEVICE T nextT(T to) {
    if (to == (T)1.0f) return nextT();

    return nextT((T)0.0f, to);
  };

  /**
   * This method returns random T in range [from..to]
   * @param from
   * @param to
   * @return
   */
  SD_INLINE SD_DEVICE T nextT(T from, T to) { return from + (nextT() * (to - from)); }

  SD_INLINE SD_DEVICE uint64_t relativeUInt(LongType index) { return buffer->getElement(index); }

  /**
   *  relative methods are made as workaround for lock-free concurrent execution
   */
  SD_INLINE SD_DEVICE int relativeInt(LongType index) {
    return (int)(relativeUInt(index) % (DataTypeUtils::max<uint32_t>() + 1));
  }

  /**
   * This method returns random int within [0..to]
   *
   * @param index
   * @param to
   * @return
   */
  SD_INLINE SD_DEVICE int relativeInt(LongType index, int to) {
    int rel = relativeInt(index);
    return rel % to;
  }

  /**
   * This method returns random int within [from..to]
   *
   * @param index
   * @param to
   * @param from
   * @return
   */
  inline int SD_DEVICE relativeInt(LongType index, int to, int from) {
    if (from == 0) return relativeInt(index, to);

    return from + relativeInt(index, to - from);
  }

  /**
   * This method returns random T within [0..1]
   *
   * @param index
   * @return
   */

  SD_INLINE SD_DEVICE T relativeT(LongType index) {
    if (sizeof(T) < 4) {
      // FIXME: this is fast hack for short types, like fp16. This should be improved.
      return (T)((float)relativeUInt(index) / (float)DataTypeUtils::max<uint32_t>());
    } else
      return (T)relativeUInt(index) / (T)DataTypeUtils::max<uint32_t>();
  }

  /**
   * This method returns random T within [0..to]
   *
   * @param index
   * @param to
   * @return
   */
  SD_INLINE SD_DEVICE T relativeT(LongType index, T to) {
    if (to == (T)1.0f) return relativeT(index);

    return relativeT(index, (T)0.0f, to);
  }

  /**
   * This method returns random T within [from..to]
   *
   * @param index
   * @param from
   * @param to
   * @return
   */
  SD_INLINE SD_DEVICE T relativeT(LongType index, T from, T to) { return from + (relativeT(index) * (to - from)); }

  /**
   * This method skips X elements from buffer
   *
   * @param numberOfElements number of elements to skip
   */
  SD_INLINE SD_DEVICE void rewind(LongType numberOfElements) { buffer->rewindH(numberOfElements); }
};
}  // namespace random
}  // namespace sd

#endif  // LIBND4J_HELPER_RANDOM_H
