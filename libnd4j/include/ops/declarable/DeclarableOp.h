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

#ifndef LIBND4J_DECLARABLE_OPS_H
#define LIBND4J_DECLARABLE_OPS_H
#include <system/common.h>
#include <array/NDArray.h>
#include <array/ResultSet.h>
#include <array/ShapeList.h>
#include <helpers/OpArgsHolder.h>
#include <helpers/helper_hash.h>
#include <ops/declarable/EmptyHandling.h>
#include <ops/declarable/OpDescriptor.h>
#include <types/float16.h>
#include <graph/Context.h>

#include <chrono>
#include <ctime>
#include <mutex>
#include <sstream>

using namespace sd::graph;

namespace sd {
namespace ops {

SD_LIB_EXPORT sd::Status conditionHelper(const char* file, int line, int condition, int argNumber, const char* format,
                                         ...);

template <typename T>
sd::Status resultHelper(T status, const char* func, const char* file, int line) {
  if (status != sd::Status::OK) {
    //  TODO: fill out error codes here
    fprintf(stderr, "Validation error at %s:%d code=%d(%s) \"%s\" \n", file, line, static_cast<unsigned int>(status),
            "", func);

    return sd::Status::BAD_INPUT;
  }

  return sd::Status::OK;
}

/**
 * This class is the basic building block of Graph Operations. Any CustomOp out there is built on top of this "abstract"
 * class.
 *
 */
class SD_LIB_EXPORT DeclarableOp {
 private:
  std::mutex _registrator;
  bool _registered = false;
  std::string _name;

 protected:
  OpDescriptor* _descriptor;
  NDArray* _scalar = nullptr;

  virtual void registerTypes();

  /**
   * This method executes this Op, and defined for most of individual ops separately
   */
  virtual sd::Status validateAndExecute(Context& block) = 0;

  /**
   * This method ensures that target variable has enough space for op execution
   *
   * TODO: we want workspaces support right here
   */
  bool allocateResult(Context& block, std::initializer_list<sd::LongType>& shape, char order = 'c');
  bool allocateResult(Context& block, sd::LongType* shape);

  /**
   * This method overwrites existen NDArray or NDArrayList in VariableSpace
   *
   * PLEASE NOTE: This method is dangerous.
   *
   * @param block
   * @param numOutput
   * @param array
   */
  void overwriteResult(Context& block, int outputIdx, NDArray* array);
  void overwriteResult(Context& block, int outputIdx, NDArrayList* list);

  /*
   * This method attaches array to specific Variable, identified by node ID and outputNumber (which is output index for
   * multi-output operations)
   */
  void storeResult(Context& block, int outputNumber, NDArray& array);
  void storeResult(Context& block, int outputNumber, NDArray* array);
  sd::NDArray* getZ(Context& block, int inputId = 0);
  sd::NDArray* getNullifiedZ(Context& block, int inputId = 0);

  /**
   *   This method pre-allocates NDArrays for Op output, in case they are not available at op execution time
   */
  int prepareOutputs(Context& block);

  virtual samediff::EmptyHandling emptyHandling();

 public:
  // for special cases, like BooleanOps
  DeclarableOp();
  DeclarableOp(const char* name, int numInputs, bool scalar);

  // regular constructors
  DeclarableOp(int numInputs, int numOutputs, const char* opName, bool allowsInplace);
  DeclarableOp(int numInputs, int numOutputs, const char* opName, bool allowsInplace, bool divergent);
  DeclarableOp(int numInputs, int numOutputs, const char* opName, bool allowsInplace, int tArgs, int iArgs);

  // for LogicalOps
  DeclarableOp(const char* name, bool isLogical);

  // default testructor
  virtual ~DeclarableOp();

  // this method returns OpDescriptor, describing this Op instance
  OpDescriptor* getOpDescriptor();

  virtual sd::Status validateDataTypes(Context& block);

  /**
   *   This method should be available in each implemented Op, and should return Op output shape(s), for a given input
   * shape(s)
   */
  virtual ShapeList* calculateOutputShape(ShapeList* inputShape, sd::graph::Context& block) = 0;

  /**
   * Returns opName
   *
   * @return
   */
  std::string* getOpName();

  /**
   * Returns opHash
   */
  sd::LongType getOpHash();

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
  virtual sd::Status execute(Context* block);

  sd::Status execute(const std::vector<NDArray*>& inputs, const std::vector<NDArray*>& outputs);

  template <class T, typename = std::enable_if<DataTypeUtils::scalarTypesForExecution<T>::value>>
  sd::Status execute(const std::vector<NDArray*>& inputs, const std::vector<NDArray*>& outputs,
                     std::initializer_list<T> tArgs);

  sd::Status execute(const std::vector<NDArray*>& inputs, const std::vector<NDArray*>& outputs,
                     const std::vector<double>& tArgs, const std::vector<sd::LongType>& iArgs,
                     const std::vector<bool>& bArgs = std::vector<bool>(),
                     const std::vector<sd::DataType>& dArgs = std::vector<sd::DataType>(), bool isInplace = false);

  sd::ResultSet evaluate(const std::vector<NDArray*>& inputs);

  template <class T, typename = std::enable_if<DataTypeUtils::scalarTypesForExecution<T>::value>>
  sd::ResultSet evaluate(const std::vector<NDArray*>& inputs, std::initializer_list<T> args);

  sd::ResultSet evaluate(const std::vector<NDArray*>& inputs, const std::vector<double>& tArgs,
                         const std::vector<sd::LongType>& iArgs, const std::vector<bool>& bArgs = std::vector<bool>(),
                         const std::vector<sd::DataType>& dArgs = std::vector<sd::DataType>(), bool isInplace = false);

  sd::Status execute(sd::graph::RandomGenerator& rng, const std::vector<NDArray*>& inputs,
                     const std::vector<NDArray*>& outputs, const std::vector<double>& tArgs,
                     const std::vector<sd::LongType>& iArgs, const std::vector<bool>& bArgs,
                     const std::vector<sd::DataType>& dArgs = std::vector<sd::DataType>(), bool isInplace = false,
                     sd::DataType type = sd::DataType::FLOAT32);

  sd::ResultSet execute(const sd::OpArgsHolder& holder, bool isInplace = false);

  // There methods provide various validation options
  sd::Status validateNonEmptyInput(Context& block);

  // this method checks if all input arrays have equal lengths
  sd::Status validateInputLengthMatch(Context& block);

  // this method checks if all input arrays have the same shapes (orders/strides are NOT checked)
  sd::Status validateInputDimensionsMatch(Context& block);

  // this method check if all input arrays have the same orders
  sd::Status validateOrdersMatch(Context& block);

  // this method checks if all input arrays are 2D
  sd::Status validateInput2D(Context& block);

  // this method checks if all input arrays are 3D
  sd::Status validateInput3D(Context& block);

  // this method checks if all input arrays are 4D
  sd::Status validateInput4D(Context& block);

  // this method checks if all input arrays are ND
  sd::Status validateInputDimensions(Context& block, int rank);

  // this method checks if number of available arguments matches op expectations
  sd::Status validateArguments(Context& block);
};
}  // namespace ops
}  // namespace sd

#endif  // LIBND4J_DECLARABLE_OPS_H
