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
            T *biasShapeInfo;

            T *params;
            int *paramsShapeInfo;

            T *input;
            int *inputShapeInfo;

            T *mask;
            int *maskShapeInfo;

            T *output;
            T *outputShapeInfo;

            Nd4jIndex allocated;
            Nd4jIndex length;
            void *workspace;

            /**
             * This method "allocates" memory chunk from workspace
             *
             * @param bytes
             * @return
             */
            virtual T *allocate(long bytes) = 0;

            /**
             * This method should validate parameters & bias, and return TRUE if everything ok. False otherwise
             * @return
             */
            virtual bool validateParameters() = 0;

            /**
             * This method should validate input parameters, and return TRUE if everything ok. False otherwise
             * @return
             */
            virtual bool validateInput() = 0;

            /**
             * This method should valudate output parameters, and return TRUE if everything is ok, FALSE otherwise
             * @return
             */
            virtual bool validateOutput() = 0;

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
                return allocated; // usually just 0
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
            bool setParameters(T *params, int *paramsShapeInfo, T *bias, int *biasShapeInfo) {
                this->params = params;
                this->paramsShapeInfo = paramsShapeInfo;
                this->biasShapeInfo = biasShapeInfo;
                this->bias = bias;

                return validateParameters();
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
            bool setInput(T *input, int *inputShapeInfo, T *mask, int *maskShapeInfo) {
                this->input = input;
                this->inputShapeInfo = inputShapeInfo;
                this->mask = mask;
                this-maskShapeInfo = maskShapeInfo;

                return validateInput();
            }

            /**
             * This method allows to specify output pointer for layer
             * @param output
             * @param shapeInfo
             */
            bool setOutput(T *output, int *shapeInfo) {
                this->output = output;
                this->outputShapeInfo = shapeInfo;

                return validateOutput();
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
