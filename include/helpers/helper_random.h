//
// @author raver119@gmail.com
//

#ifndef LIBND4J_HELPER_RANDOM_H
#define LIBND4J_HELPER_RANDOM_H

#ifdef __CUDACC__
#include <curand.h>
#endif

#define MAX_INT 2147483647;

#include <helpers/helper_generator.h>

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
                int r = (int) buffer->getNextElement();
                return r < 0 ? -1 * r : r;
            };

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
                return (T) nextInt() / (T) MAX_INT;
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

        };
    }
}

#endif //LIBND4J_HELPER_RANDOM_H
