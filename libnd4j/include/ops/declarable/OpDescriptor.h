/* ******************************************************************************
 *
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
 *  See the NOTICE file distributed with this work for additional
 *  information regarding copyright ownership.
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
#include <array/DataType.h>
#include <graph/scheme/node_generated.h>
#include <helpers/helper_hash.h>
#include <ops/InputType.h>

#include <initializer_list>
#include <string>
#include <vector>

namespace sd {
namespace ops {
class SD_LIB_EXPORT OpExecTrace {
 public:
  std::vector<const LongType*> *inputShapeBuffers;
  std::vector<const LongType*> *outputShapeBuffers;
  const std::string *opName;
  std::vector<LongType> iArgs;
  std::vector<double> tArgs;
  std::vector<DataType> dArgs;
  std::vector<bool> bArgs;
  std::vector<std::string> sArguments;
  int opType = -1;


#ifndef __JAVACPP_HACK__
  OpExecTrace(std::vector<const LongType*> *inputShapeBuffers,
              std::vector<const LongType*> *outputShapeBuffers,
              const std::string *opName) {
    this->inputShapeBuffers = inputShapeBuffers;
    this->outputShapeBuffers = outputShapeBuffers;
    this->opName = opName;

  }

  OpExecTrace(std::vector<const LongType*> *inputShapeBuffers,
              std::vector<const LongType*> *outputShapeBuffers,
              const std::string *opName,
              std::vector<LongType> *iArgs,
              std::vector<double> *tArgs,
              std::vector<bool> *bArgs,
              std::vector<std::string> *sArgs,
              int opType) {
    this->inputShapeBuffers = inputShapeBuffers;
    this->outputShapeBuffers = outputShapeBuffers;
    this->opName = opName;
    this->opType = opType;
    for(size_t i = 0; i < tArgs->size(); i++) {
      this->tArgs.push_back(tArgs->at(i));
    }

    for(size_t i = 0; i < bArgs->size(); i++) {
      this->bArgs.push_back(bArgs->at(i));
    }

    for(size_t i = 0; i < iArgs->size(); i++) {
      this->iArgs.push_back(iArgs->at(i));
    }

    for(size_t i = 0; i < sArgs->size(); i++) {
      this->sArguments.push_back(sArgs->at(i));
    }

  }
#endif

  OpExecTrace() = default;

  ~OpExecTrace() = default;

  std::vector<const LongType*>* getInputShapeBuffers() const { return inputShapeBuffers; }
  void setInputShapeBuffers(std::vector<const LongType*>* inputShapeBuffersIn) {
    OpExecTrace::inputShapeBuffers = inputShapeBuffersIn;
  }
  std::vector<const LongType*>* getOutputShapeBuffers() const { return outputShapeBuffers; }
  void setOutputShapeBuffers(std::vector<const LongType*>* outputShapeBuffersIn) {
    OpExecTrace::outputShapeBuffers = outputShapeBuffersIn;
  }
  const std::string* getOpName() const { return opName; }
  void setOpName(const std::string* opNameIn) { OpExecTrace::opName = opNameIn; }
  const std::vector<LongType>& getIArgs() const { return iArgs; }
  void setIArgs(const std::vector<LongType>& iArgsIn) { OpExecTrace::iArgs = iArgsIn; }
  const std::vector<double>& getTArgs() const { return tArgs; }
  void setTArgs(const std::vector<double>& tArgsIn) { OpExecTrace::tArgs = tArgsIn; }
  const std::vector<DataType>& getDArgs() const { return dArgs; }
  void setDArgs(const std::vector<DataType>& dArgsIn) { OpExecTrace::dArgs = dArgsIn; }
  const std::vector<bool>& getBArgs() const { return bArgs; }
  void setBArgs(const std::vector<bool>& bArgsIn) { OpExecTrace::bArgs = bArgsIn; }
  const std::vector<std::string>& getSArguments() const { return sArguments; }
  void setSArguments(const std::vector<std::string>& sArgumentsIn) { OpExecTrace::sArguments = sArgumentsIn; }
  int getOpType() const { return opType; }
  void setOpType(int opTypeIn) { OpExecTrace::opType = opTypeIn; }
};

/**
 *   This class is very basic info holder for ops. bean/pojo pretty much.
 *
 */
class SD_LIB_EXPORT OpDescriptor {
 protected:
  // opType for legacy XYZ ops
  int _opNum = 0;

  // opName for CustomOp
  std::string _opName;

  // hash is used for ops lookup in OpRegistrator
  LongType _hash = -1;

  // minimal required/expected number of inputs/outpus for this given op
  int _numInputs = 1;
  int _numOutputs = 1;

  // enum for ops. deprecated. will be removed
  graph::OpClass _opClass;

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
  std::vector<DataType> _allowedIns;
  std::vector<DataType> _allowedOuts;

  // optional per-input configuration
  SD_MAP_IMPL<int, std::vector<DataType>> _outputTypes;
  SD_MAP_IMPL<int, std::vector<DataType>> _inputTypes;

  // field for ops that allow data type override at runtime
  bool _dtypeOverride = false;

  bool checkDataTypesMatch(DataType needle, std::vector<DataType>& haystack) const;

 public:
  // default constructor
  OpDescriptor(int numInputs, int numOutputs, std::string opName, bool allowsInplace);

  // constructor for boolean ops
  OpDescriptor(int numInputs, std::string opName, bool isScalar);
  OpDescriptor(int numInputs, const char* opName, bool isScalar);

  // default constructor
  OpDescriptor(int numInputs, int numOutputs, const char* opName, bool allowsInplace);

  // constructor for configurable op
  OpDescriptor(int numInputs, int numOutputs, const char* opName, bool allowsInplace, int tArgs, int iArgs);

  // constructor for non-configurable divergent op
  OpDescriptor(int numInputs, int numOutputs, std::string opName, bool allowsInplace, bool divergent);

  // constructor for non-configurable divergent op
  OpDescriptor(int numInputs, int numOutputs, const char* opName, bool allowsInplace, bool divergent);

  // constructor for configurable divergent op
  OpDescriptor(int numInputs, int numOutputs, const char* opName, bool allowsInplace, bool divergent, int tArgs,
               int iArgs);

  // constructor for logical ops (while, scope, etc)
  OpDescriptor(const char* opName, bool isLogic);

  bool operator==(const OpDescriptor& other) const;

  // default destructor
  ~OpDescriptor() = default;

  // this method returns minimal expected number of T arguments
  int getNumberOfTArgs();

  // this method returns minimal expected number of Integer arguments
  int getNumberOfIArgs();

  // this method returns minimal expected number of inputs
  int getNumberOfInputs();

  // this method returns hash code for this operation
  LongType getHash();

  // this method returns minimal expected number of outputs
  int getNumberOfOutputs();

  // this method returns opName (can be empty)
  std::string* getOpName();

  // returns TRUE if this op is divergent. FALSE otherwise
  bool isDivergent();

  // returns TRUE if this op allows in-place execution
  bool allowsInplace();

  // this method allows you to enable/disable inplace call for a given op
  void allowInplace(bool reallyAllow);

  // this method returns opType (applicable for legacy XYZ ops only)
  int getOpNum();

  // this method allows to set specific opNum
  void setOpNum(int opNum);

  void setHash(LongType hash);

  InputType inputType();

  OpDescriptor* setInputType(InputType type);
  OpDescriptor* setAllowedInputTypes(const std::initializer_list<DataType>& dtype);
  OpDescriptor* setAllowedOutputTypes(const std::initializer_list<DataType>& dtype);
  OpDescriptor* setAllowedInputTypes(int index, const std::vector<DataType>& dtype);
  OpDescriptor* setAllowedOutputTypes(int index, const std::vector<DataType>& dtype);
  OpDescriptor* setAllowedInputTypes(int index, DataType dtype);
  OpDescriptor* setAllowedOutputTypes(int index, DataType dtype);
  OpDescriptor* setAllowedInputTypes(DataType dtype);
  OpDescriptor* setAllowedOutputTypes(DataType dtype);
  OpDescriptor* allowOverride(bool reallyAllow);
  OpDescriptor* setSameMode(bool reallySame);
  OpDescriptor* setInputType(int idx, DataType dtype);
  OpDescriptor* setOutputType(int idx, DataType dtype);

  std::vector<DataType> getOutputTypesForOutput(int index);
  std::vector<DataType> getInputTypesForInput(int index);



  bool checkInputMatch(int index, DataType dataType);
  bool checkOutputMatch(int index, DataType dataType);
  bool isSameMode();

  bool isInherit(int index);
};
}  // namespace ops
}  // namespace sd

#endif  // LIBND4J_OPDESCRIPTOR_H
