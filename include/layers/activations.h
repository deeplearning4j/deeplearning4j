//
// @author raver119@gmail.com
//

#ifndef PROJECT_ACTIVATIONS_H
#define PROJECT_ACTIVATIONS_H

namespace nd4j {
    namespace activations {
        template <typename T>
        class IActivation {

        public:
            /**
             * This method applies activation for FF pass
             *
             * @param input
             * @param inputShapeInfo
             */
            inline static void ffActivation(T *input, int *inputShapeInfo);

            /**
             * This method applies activation for BP pass
             *
             * @param input
             * @param inputShapeInfo
             */
            inline static void bpActivation(T *input, T *epsilon, int *inputShapeInfo);
        };


        template <typename T>
        class ActivationsExecutioner {
        public:
            // add extraParams here probably?
            static inline void executeFF(int aNum, T *input, int *inputShapeInfo) {
                // we need to build activations executor here. some macros, anyone?
            }

            // add extraParams here probably?
            static inline void executeBP(int aNum, T *input, int *inputShapeInfo) {
                // we need to build activations executor here. some macros, again? :)

            }

            /**
             * This method should be backend-specific, and should be implemented accordingly
             *
             * @tparam Activation
             * @param input
             * @param inputShapeInfo
             */
            template<typename Activation>
            static virtual void executeFF(T * input, int *inputShapeInfo) = 0;

            template<typename Activation>
            static virtual void executeBP(T * input, int *inputShapeInfo) = 0;
        };
    }
}

#endif //PROJECT_ACTIVATIONS_H
