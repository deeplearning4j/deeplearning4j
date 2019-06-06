/*******************************************************************************
 * Copyright (c) 2015-2018 Skymind, Inc.
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
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


namespace nd4j {

    namespace random {

        template<typename T>
        class RandomHelper {
        private:
            nd4j::random::IGenerator *generator;
            nd4j::random::RandomBuffer *buffer;


        public:

            _CUDA_HD RandomHelper(nd4j::random::IGenerator *generator) {
                this->generator = generator;
                this->buffer = generator->getBuffer();
            }

            _CUDA_HD RandomHelper(nd4j::random::RandomBuffer *buffer) {
                this->buffer = buffer;
            }


            /**
             * This method returns random int in range [0..MAX_INT]
             * @return
             */
            inline _CUDA_D int nextInt() {
                int r = (int) nextUInt();
                return r < 0 ? -1 * r : r;
            };

            inline _CUDA_D uint64_t nextUInt() {
                return buffer->getNextElement();
            }

            /**
             * This method returns random int in range [0..to]
             * @param to
             * @return
             */
            inline _CUDA_D int nextInt(int to) {
                int r = nextInt();
                int m = to - 1;
                if ((to & m) == 0)  // i.e., bound is a power of 2
                    r = (int) ((to * (long) r) >> 31);
                else {
                    for (int u = r;
                         u - (r = u % to) + m < 0;
                         u = nextInt());
                }
                return r;
            };

            /**
             * This method returns random int in range [from..to]
             * @param from
             * @param to
             * @return
             */
            inline _CUDA_D int nextInt(int from, int to) {
                if (from == 0)
                    return nextInt(to);

                return from + nextInt(to - from);
            };


            /**
             * This method returns random T in range of [0..MAX_FLOAT]
             * @return
             */
            inline _CUDA_D T nextMaxT() {
                T rnd = (T) buffer->getNextElement();
                return rnd < 0 ? -1 * rnd : rnd;
            };


            /**
             * This method returns random T in range of [0..1]
             * @return
             */
            inline _CUDA_D T nextT() {
                return (T) nextUInt() / (T) nd4j::DataTypeUtils::max<Nd4jULong>();
            }

            /**
             * This method returns random T in range of [0..to]
             * @param to
             * @return
             */
            inline _CUDA_D T nextT(T to) {
                if (to == (T) 1.0f)
                    return nextT();

                return nextT((T) 0.0f, to);
            };

            /**
             * This method returns random T in range [from..to]
             * @param from
             * @param to
             * @return
             */
            inline _CUDA_D T nextT(T from, T to) {
                return from + (nextT() * (to - from));
            }

            inline _CUDA_D uint64_t relativeUInt(Nd4jLong index) {
                return buffer->getElement(index);
            }

            /**
             *  relative methods are made as workaround for lock-free concurrent execution
             */
            inline _CUDA_D int relativeInt(Nd4jLong index) {
                return (int) (relativeUInt(index) % (nd4j::DataTypeUtils::max<uint32_t>() + 1));
            }

            /**
             * This method returns random int within [0..to]
             *
             * @param index
             * @param to
             * @return
             */
            inline _CUDA_D int relativeInt(Nd4jLong index, int to) {
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
            inline int _CUDA_D relativeInt(Nd4jLong index, int to, int from) {
                if (from == 0)
                    return relativeInt(index, to);

                return from + relativeInt(index, to - from);
            }

            /**
             * This method returns random T within [0..1]
             *
             * @param index
             * @return
             */

            inline _CUDA_D T relativeT(Nd4jLong index) {
                if (sizeof(T) < 4) {
                    // FIXME: this is fast hack for short types, like fp16. This should be improved.
                    return (T)((float) relativeUInt(index) / (float) nd4j::DataTypeUtils::max<uint32_t>());
                } else return (T) relativeUInt(index) / (T) nd4j::DataTypeUtils::max<uint32_t>();
            }

            /**
             * This method returns random T within [0..to]
             *
             * @param index
             * @param to
             * @return
             */
            inline _CUDA_D T relativeT(Nd4jLong index, T to) {
                if (to == (T) 1.0f)
                    return relativeT(index);

                return relativeT(index, (T) 0.0f, to);
            }

            /**
             * This method returns random T within [from..to]
             *
             * @param index
             * @param from
             * @param to
             * @return
             */
            inline _CUDA_D T relativeT(Nd4jLong index, T from, T to) {
                return from + (relativeT(index) * (to - from));
            }


            /**
             * This method skips X elements from buffer
             *
             * @param numberOfElements number of elements to skip
             */
            inline _CUDA_D void rewind(Nd4jLong numberOfElements) {
                buffer->rewindH(numberOfElements);
            }
        };
    }
}

#endif //LIBND4J_HELPER_RANDOM_H
