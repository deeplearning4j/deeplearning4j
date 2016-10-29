//
// @author raver119@gmail.com
//

#ifndef LIBND4J_HELPER_RANDOM_H
#define LIBND4J_HELPER_RANDOM_H

#ifdef __CUDACC__
#include <curand.h>
#endif

#define MAX_INT 2147483647
#define MAX_UINT 18446744073709551615LLU

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
            RandomHelper(nd4j::random::IGenerator *generator) {
                this->generator = generator;
                this->buffer = generator->getBuffer();
            }


            RandomHelper(nd4j::random::RandomBuffer *buffer) {
                this->buffer = buffer;
            }


            /**
             * This method returns random int in range [0..MAX_INT]
             * @return
             */
            int nextInt() {
                int r = (int) nextUInt();
                return r < 0 ? -1 * r : r;
            };

            uint64_t nextUInt() {
                return buffer->getNextElement();
            }

            /**
             * This method returns random int in range [0..to]
             * @param to
             * @return
             */
            int nextInt(int to) {
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
            int nextInt(int from, int to) {
                if (from == 0)
                    return nextInt(to);

                return from + nextInt(to - from);
            };


            /**
             * This method returns random T in range of [0..MAX_FLOAT]
             * @return
             */
            T nextMaxT() {
                T rnd = (T) buffer->getNextElement();
                return rnd < 0 ? -1 * rnd : rnd;
            };


            /**
             * This method returns random T in range of [0..1]
             * @return
             */
            T nextT() {
                return (T) nextUInt() / (T) MAX_UINT;
            }

            /**
             * This method returns random T in range of [0..to]
             * @param to
             * @return
             */
            T nextT(T to) {
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
            T nextT(T from, T to) {
                return from + (nextT() * (to - from));
            }

            uint64_t relativeUInt(int index) {
                return buffer->getElement(index);
            }

            /**
             *  relative methods are made as workaround for lock-free concurrent execution
             */

            int relativeInt(int index) {
                return (int) relativeInt(index);
            }

            /**
             * This method returns random int within [0..to]
             *
             * @param index
             * @param to
             * @return
             */
            int relativeInt(int index, int to) {
                // TODO: to be implemented
                return 0;
            }

            /**
             * This method returns random int within [from..to]
             *
             * @param index
             * @param to
             * @param from
             * @return
             */
            int relativeInt(int index, int to, int from) {
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
            T relativeT(int index) {
                return (T) relativeUInt(index) / (T) MAX_UINT;
            }

            /**
             * This method returns random T within [0..to]
             *
             * @param index
             * @param to
             * @return
             */
            T relativeT(int index, T to) {
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
            T relativeT(int index, T from, T to) {
                return from + (relativeT(index) * (to - from));
            }


            /**
             * This method skips X elements from buffer
             *
             * @param numberOfElements number of elements to skip
             */
            void rewind(long numberOfElements) {
                long newPos = buffer->getOffset() + numberOfElements;
                if (newPos > buffer->getSize())
                    newPos = numberOfElements - (buffer->getSize() - buffer->getOffset());
                else if (newPos == buffer->getSize())
                    newPos = 0;

                buffer->setOffset(newPos);
            }
        };
    }
}

#endif //LIBND4J_HELPER_RANDOM_H
