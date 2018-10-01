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

#ifndef LIBND4J_OPDESCRIPTOR_H
#define LIBND4J_OPDESCRIPTOR_H

#include <string>
#include <vector>
#include <map>
#include <initializer_list>
#include <helpers/helper_hash.h>
#include <ops/InputType.h>
#include <graph/generated/node_generated.h>
#include <array/DataType.h>

namespace nd4j {
    namespace ops {

        /**
        *   This class is very basic info holder for ops. bean/pojo pretty much.
        *
        */
        class ND4J_EXPORT OpDescriptor {
        protected:
            // opNum for legacy XYZ ops
            int _opNum = 0;

            // opName for CustomOp
            std::string _opName;

            // hash is used for ops lookup in OpRegistrator
            Nd4jLong _hash = -1;

            // minimal required/expected number of inputs/outpus for this given op
            int _numInputs = 1;
            int _numOutputs = 1;

            // enum for ops. deprecated. will be removed
            nd4j::graph::OpClass _opClass;

            // special flag for divergent ops - ops that CAN and WILL modify graph behavior. Literally: IF, CASE.
            bool _divergent = false;

            // flag, if this given op allows in-place execution
            bool _allowsInplace = true;

            // minimal required number of T-type arguments.
            // -1 as value means: not limited, variable number of arguments
            int _tArgs = 0;

            // minimal required number of Integer-type arguments.
            // -1 as value means: not limited, variable number of arguments
            int _iArgs = 0;

            // field for BooleanOps
            bool _scalar = false;

            // field for LogicOps
            bool _logic = false;

            // default InputType is numeric
            InputType _inputType = InputType_NUMERIC;


            bool _sameMode = false;
            std::vector<nd4j::DataType> _allowedIns;
            std::vector<nd4j::DataType> _allowedOuts;

            // optional per-input configuration
            std::map<int, nd4j::DataType> _outputTypes;
            std::map<int, nd4j::DataType> _inputTypes;

        public:
            // default constructor
            OpDescriptor(int numInputs, int numOutputs, std::string opName, bool allowsInplace);

            // constructor for boolean ops
            OpDescriptor(int numInputs, std::string opName, bool isScalar);
            OpDescriptor(int numInputs, const char* opName, bool isScalar);

            // default constructor
            OpDescriptor(int numInputs, int numOutputs, const char *opName, bool allowsInplace);

            // constructor for configurable op
            OpDescriptor(int numInputs, int numOutputs, const char *opName, bool allowsInplace, int tArgs, int iArgs);

            // constructor for non-configurable divergent op
            OpDescriptor(int numInputs, int numOutputs, std::string opName, bool allowsInplace, bool divergent);

            // constructor for non-configurable divergent op
            OpDescriptor(int numInputs, int numOutputs, const char *opName, bool allowsInplace, bool divergent);

            // constructor for configurable divergent op
            OpDescriptor(int numInputs, int numOutputs, const char *opName, bool allowsInplace, bool divergent, int tArgs, int iArgs);

            // constructor for logical ops (while, scope, etc)
            OpDescriptor(const char * opName, bool isLogic);

            bool operator==(const OpDescriptor& other) const;

            // default destructor
            ~OpDescriptor();

            // this method returns minimal expected number of T arguments
            int getNumberOfTArgs();

            // this method returns minimal expected number of Integer arguments
            int getNumberOfIArgs();

            // this method returns minimal expected number of inputs
            int getNumberOfInputs();

            // this method returns hash code for this operation
            Nd4jLong getHash();

            // this method returns minimal expected number of outputs
            int getNumberOfOutputs();

            // this method returns opName (can be empty)
            std::string *getOpName();

            // returns TRUE if this op is divergent. FALSE otherwise
            bool isDivergent();

            // returns TRUE if this op allows in-place execution
            bool allowsInplace();

            // this method returns opNum (applicable for legacy XYZ ops only)
            int getOpNum();

            // this method allows to set specifc opNum
            void setOpNum(int opNum);

            void setHash(Nd4jLong hash);

            InputType inputType();

            OpDescriptor* setInputType(const InputType type);
            OpDescriptor* setAllowedInputTypes(const std::initializer_list<nd4j::DataType> &dtype);
            OpDescriptor* setAllowedOutputTypes(const std::initializer_list<nd4j::DataType> &dtype);
            OpDescriptor* setAllowedInputTypes(const nd4j::DataType dtype);
            OpDescriptor* setAllowedOutputTypes(const nd4j::DataType dtype);
            OpDescriptor* setSameMode(const bool reallySame);
            OpDescriptor* setInputType(const int idx, const nd4j::DataType dtype);
            OpDescriptor* setOutputType(const int idx, const nd4j::DataType dtype);

        };
    }
}

#endif //LIBND4J_OPDESCRIPTOR_H
