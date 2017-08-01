//
// @author raver119@gmail.com
//

#ifndef PROJECT_OPS_H
#define PROJECT_OPS_H


#define no_special_ff static bool requiresSpecialFF = false; inline static void ffActivation(T *input, int *inputShapeInfo) {};
#define no_special_bp static bool requiresSpecialBP = false; inline static void bpActivation(T *input, T *epsilon, int *inputShapeInfo) {};

#define no_regular_ff static bool requiresSpecialFF = true; inline static T ffActivation(T value) {};
#define no_regular_bp static bool requiresSpecialBP = true; inline static T bpActivation(T value, T epsilon) {};

#include <layers/activations.h>

// this might be not the best idea, given limited math availability
#include <templatemath.h>

namespace nd4j {
    namespace activations {
        template<typename T>
        class ReLU  {

        public:
            no_special_ff
            no_special_bp

            /**
             * This method applies activation for FF pass
             *
             * @param input
             * @param inputShapeInfo
             */
            inline static T ffActivation(T value) {
                return nd4j::math::relu<T>(value);
            }

            /**
             * This method applies activation for BP pass
             *
             * @param input
             * @param inputShapeInfo
             */
            inline static T bpActivation(T value, T epsilon) {
                // FIXME: ultra-bad. should consider conigurable extra params here
                T extra[] = {(T) 0.0f};
                return simdOps::Step::op(value, extra) * epsilon;
            }
        };


        template<typename T>
        class Identity  {

        public:
            no_special_ff
            no_special_bp

            /**
             * This method applies activation for FF pass
             *
             * @param input
             * @param inputShapeInfo
             */
            inline static T ffActivation(T value) {
                return value;
            }

            /**
             * This method applies activation for BP pass
             *
             * @param input
             * @param inputShapeInfo
             */
            inline static T bpActivation(T value, T epsilon) {
                return epsilon;
            }
        };
    }
}

#endif //PROJECT_OPS_H
