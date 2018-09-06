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

#ifndef LIBND4J_DECLARABLE_OPS_H
#define LIBND4J_DECLARABLE_OPS_H

#include <sstream>
#include <types/float16.h>
#include <pointercast.h>
#include <NDArray.h>
#include <graph/Context.h>
#include "OpDescriptor.h"
#include <helpers/helper_hash.h>
#include <array/ShapeList.h>
#include <array/ResultSet.h>
#include <helpers/OpArgsHolder.h>
#include <dll.h>
//#include <ops/declarable/declarable_ops.h>

#include <chrono>
#include <ctime>

using namespace nd4j::graph;

namespace nd4j {
    namespace ops {

        Nd4jStatus ND4J_EXPORT conditionHelper(const char *file, int line, int condition, int argNumber, const char *format, ...);


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

        /**
         * This class is the basic building block of Graph Operations. Any CustomOp out there is built on top of this "abstract" class.
         *
         */
        class ND4J_EXPORT DeclarableOp {
        protected:
            OpDescriptor *_descriptor;

            /**
             * This method executes this Op, and defined for most of individual ops separately
             */
            virtual Nd4jStatus validateAndExecute(Context<T>& block) = 0;


            /**
             * This method ensures that target variable has enough space for op execution
             *
             * TODO: we want workspaces support right here
             */
            bool allocateResult(Context& block, std::initializer_list<Nd4jLong>& shape, char order = 'c');
            bool allocateResult(Context& block, Nd4jLong* shape);

            /**
             * This method overwrites existen NDArray or NDArrayList in VariableSpace
             *
             * PLEASE NOTE: This method is dangerous.
             *
             * @param block
             * @param numOutput
             * @param array
             */
            void overwriteResult(Context<T>& block, int outputIdx, NDArray<T>* array);
            void overwriteResult(Context<T>& block, int outputIdx, NDArrayList<T>* list);

            /*
            * This method attaches array to specific Variable, identified by node ID and outputNumber (which is output index for multi-output operations)
            */
            void storeResult(Context &block, int outputNumber, NDArray& array);
            void storeResult(Context &block, int outputNumber, NDArray* array);
            nd4j::NDArray* getZ(Context& block, int inputId = 0);

            /**
            *   This method pre-allocates NDArrays for Op output, in case they are not available at op execution time
            */
            int prepareOutputs(Context& block);

            //std::vector<int>* calculateOutputShape(std::vector<int>* inputShape, nd4j::graph::Block<T>& block);
        public:
            // for special cases, like BooleanOps
            DeclarableOp();
            DeclarableOp(const char *name, int numInputs, bool scalar);

            // regular constructors
            DeclarableOp(int numInputs, int numOutputs, const char *opName, bool allowsInplace);
            DeclarableOp(int numInputs, int numOutputs, const char *opName, bool allowsInplace, bool divergent);
            DeclarableOp(int numInputs, int numOutputs, const char *opName, bool allowsInplace, int tArgs, int iArgs);

            // for LogicalOps
            DeclarableOp(const char *name, bool isLogical);

            // default testructor
            ~DeclarableOp();

            // this method returns OpDescriptor, describing this Op instance
            OpDescriptor *getOpDescriptor();

            /**
            *   This method should be available in each implemented Op, and should return Op output shape(s), for a given input shape(s)
            */
            virtual ShapeList* calculateOutputShape(ShapeList* inputShape, nd4j::graph::Context& block) = 0;

            /**
             * Returns opName
             *
             * @return
             */
            std::string *getOpName();

            /**
             * Returns opHash
             */
            Nd4jLong getOpHash();

            /**
             * This method sets arguments for op
             */
//            void setArguments();

            /**
             * This method returns pointer to results
             */
//            void getResults();

            /**
             * This method executes given Op
             *
             * @param block
             * @return 0 if OK, error code otherwise
             */
            virtual Nd4jStatus execute(Context* block);

            nd4j::ResultSet* execute(std::initializer_list<NDArray*> inputs, std::initializer_list<double> tArgs, std::initializer_list<Nd4jLong> iArgs, bool isInplace = false);
            Nd4jStatus execute(std::initializer_list<NDArray*> inputs, std::initializer_list<NDArray*> outputs , std::initializer_list<double> tArgs, std::initializer_list<Nd4jLong> iArgs, bool isInplace = false);
            Nd4jStatus execute(nd4j::random::RandomBuffer *rng, std::initializer_list<NDArray*> inputs, std::initializer_list<NDArray*> outputs , std::initializer_list<double> tArgs, std::initializer_list<Nd4jLong> iArgs, bool isInplace = false);

            nd4j::ResultSet* execute(const std::vector<NDArray*>& inputs, const std::vector<double>& tArgs, const std::vector<Nd4jLong>& iArgs, bool isInplace = false);
            Nd4jStatus execute(std::vector<NDArray*>& inputs, std::vector<NDArray*>& outputs , std::vector<double>& tArgs, std::vector<Nd4jLong>& iArgs, bool isInplace = false);
            Nd4jStatus execute(nd4j::random::RandomBuffer *rng, std::vector<NDArray*>& inputs, std::vector<NDArray*>& outputs , std::vector<double>& tArgs, std::vector<Nd4jLong>& iArgs, bool isInplace = false);

            nd4j::ResultSet* execute(const nd4j::OpArgsHolder& holder, bool isInplace = false);

            // There methods provide various validation options
            Nd4jStatus validateNonEmptyInput(Context& block);

            // this method checks if all input arrays have equal lengths
            Nd4jStatus validateInputLengthMatch(Context& block);

            // this method checks if all input arrays have the same shapes (orders/strides are NOT checked)
            Nd4jStatus validateInputDimensionsMatch(Context& block);

            // this method check if all input arrays have the same orders
            Nd4jStatus validateOrdersMatch(Context& block);

            // this method checks if all input arrays are 2D
            Nd4jStatus validateInput2D(Context& block);

            // this method checks if all input arrays are 3D
            Nd4jStatus validateInput3D(Context& block);

            // this method checks if all input arrays are 4D
            Nd4jStatus validateInput4D(Context& block);

            // this method checks if all input arrays are ND
            Nd4jStatus validateInputDimensions(Context& block, int rank);

            // this method checks if number of available arguments matches op expectations
            Nd4jStatus validateArguments(Context& block);
        };
    }
}

#endif //LIBND4J_DECLARABLE_OPS_H
