//
// @author raver119@gmail.com
//

#ifndef LIBND4J_DECLARABLE_OPS_H
#define LIBND4J_DECLARABLE_OPS_H

#include <sstream>
#include <types/float16.h>
#include <pointercast.h>
#include <NDArray.h>
#include <Block.h>
#include "OpDescriptor.h"
#include <helpers/helper_hash.h>
#include <ShapeList.h>
#include <ArrayList.h>
//#include <ops/declarable/declarable_ops.h>

#include <chrono>
#include <ctime>

using namespace nd4j::graph;

namespace nd4j {
    namespace ops {

        Nd4jStatus conditionHelper(const char *file, int line, int condition, int argNumber, const char *format, ...);


        template<typename T>
        Nd4jStatus resultHelper(T status, const char *func, const char *file, int line) {
            if (status) {
                //  TODO: fill out error codes here
                fprintf(stderr, "Validation error at %s:%d code=%d(%s) \"%s\" \n", file, line,
                        static_cast<unsigned int>(status), "", func);

                return ND4J_STATUS_BAD_INPUT;
            }

            return ND4J_STATUS_OK;
        }

        template <typename T>
        class DeclarableOp {
        protected:
            Block<T> *_block;
            OpDescriptor *_descriptor;

            /**
             * This method executes this Op
             */
            virtual Nd4jStatus validateAndExecute(Block<T>& block) = 0;


            /**
             * This method ensures that target variable has enough space for op execution
             *
             * TODO: we want workspaces support right here
             */
            bool allocateResult(Block<T>& block, std::initializer_list<int>& shape, char order = 'c');
            bool allocateResult(Block<T>& block, int* shape);
            void storeResult(Block<T> &block, int outputNumber, NDArray<T>& array);
            nd4j::NDArray<T> *getZ(Block<T>& block, int inputId = 0);

            bool prepareOutputs(Block<T>& block);

            //std::vector<int>* calculateOutputShape(std::vector<int>* inputShape, nd4j::graph::Block<T>& block);
        public:
            DeclarableOp(int numInputs, int numOutputs, const char *opName, bool allowsInplace);
            DeclarableOp(int numInputs, int numOutputs, const char *opName, bool allowsInplace, bool divergent);
            DeclarableOp(int numInputs, int numOutputs, const char *opName, bool allowsInplace, int tArgs, int iArgs);

            ~DeclarableOp();

            OpDescriptor *getOpDescriptor();


            virtual ShapeList* calculateOutputShape(ShapeList* inputShape, nd4j::graph::Block<T>& block) = 0;

            /**
             * Returns opName
             *
             * @return
             */
            std::string *getOpName();

            Nd4jIndex getOpHash();

            /**
             * This method sets arguments for op
             */
            void setArguments();

            /**
             * This method returns pointer to results
             */
            void getResults();

            /**
             * This method executes everything
             * @param block
             * @return
             */
            Nd4jStatus execute(Block<T>* block);

            nd4j::ArrayList<T>* execute(std::initializer_list<NDArray<T>*> inputs, std::initializer_list<T> tArgs, std::initializer_list<int> iArgs);
            Nd4jStatus execute(std::initializer_list<NDArray<T>*> inputs, std::initializer_list<NDArray<T>*> outputs , std::initializer_list<T> tArgs, std::initializer_list<int> iArgs);

            // There methods provide various validation options
            Nd4jStatus validateNonEmptyInput(Block<T>& block);
            Nd4jStatus validateInputLengthMatch(Block<T>& block);
            Nd4jStatus validateInputDimensionsMatch(Block<T>& block);
            Nd4jStatus validateOrdersMatch(Block<T>& block);
            Nd4jStatus validateInput2D(Block<T>& block);
            Nd4jStatus validateInput3D(Block<T>& block);
            Nd4jStatus validateInput4D(Block<T>& block);
            Nd4jStatus validateInputDimensions(Block<T>& block, int rank);

            Nd4jStatus validateArguments(Block<T>& block);
        };
    }
}

#endif //LIBND4J_DECLARABLE_OPS_H
