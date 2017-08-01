//
// @author raver119@gmail.com
//

#ifndef PROJECT_ACTIVATIONS_H
#define PROJECT_ACTIVATIONS_H

#define ACTIVATION_OPS \
        (0, nd4j::activations::Identity) ,\
        (1, nd4j::activations::ReLU)


        /*
         * We don't really need this Executioner class, and it will be removed in favor of backend-specific helper.
         * There's just nothing to override here.
         */
namespace nd4j {
    namespace activations {


        template <typename T>
        class IActivationsExecutioner {
        public:
            // DEPRECATED, WILL BE REMOVED
            // add extraParams here probably?
            static inline void executeFF(int aNum, T *input, T *output, int *inputShapeInfo) {
                // we need to build activations executor here. some macros, anyone?
            }

            // DEPRECATED, WILL BE REMOVED
            // add extraParams here probably?
            static inline void executeBP(int aNum, T *input, T *epsilon, T *output, int *inputShapeInfo) {
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
            static void executeFF(T *input, T *output, int *inputShapeInfo) {
                // platform-specific invocation loop here
            }

            template<typename Activation>
            static void executeBP(T *input, T *epsilon, T *output, int *inputShapeInfo) {
                // platform-specific invocation loop here
            };
        };
    }
}

#endif //PROJECT_ACTIVATIONS_H
