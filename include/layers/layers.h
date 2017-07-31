//
// @author raver119@gmail.com
//

#ifndef PROJECT_LAYERS_H
#define PROJECT_LAYERS_H


namespace nd4j {
    namespace layers {

        template<typename T>
        class INativeLayer {
        protected:
            T *bias;
            T *params;

            T *input;
            T *mask;
            T *output;

            Nd4jIndex allocated;
            Nd4jIndex length;
            void *workspace;

            /**
             * This method "allocates" memory chunk from workspace
             *
             * @param bytes
             * @return
             */
            T * allocate(long bytes) {
                // FIXME: i think this method should be backend-specific
                return nullptr;
            }

        public:
            /**
             * this method attaches layer to workspace memory
             * @param memory
             */
            void setWorkspace(void *memory, Nd4jIndex length) {
                this->length = length;
                this->workspace = memory;
            };

            /**
             * this method returns number of bytes used
             * @return
             */
            long getUsedMemory() {
                return allocated;
            }

            /**
             * This method allows to set parameters/biases for this layer
             * This input will be either activation from previous layer, or error coming from next layer
             *
             * @param params
             * @param paramsShapeInfo
             * @param bias
             * @param biasShapeInfo
             */
            void setParameters(T *params, int *paramsShapeInfo, T *bias, int *biasShapeInfo) {
                //
            }

            /**
             * This method allows to specify input data for layer
             * This output will be either activation of this layer, or error from next layer
             *
             * @param input
             * @param shapeInfo
             * @param mask
             * @param shapeInfo
             */
            virtual void setInput(T *input, int *inputShapeInfo, T *mask, int *maskShapeInfo) {
                //
            }

            /**
             * This method allows to specify output pointer for layer
             * @param output
             * @param shapeInfo
             */
            virtual void setOutput(T *output, int *shapeInfo) {
                //
            }


            /**
             * This method executes feed-forward pass on this layer
             */
            virtual void feedForward() = 0;

            /**
             * This method executes back-propagation pass on this layer
             */
            virtual void backPropagate() = 0;
        };
    }
}

#endif //PROJECT_LAYERS_H
