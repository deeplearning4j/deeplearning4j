/* ******************************************************************************
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/
#ifndef LIBND4J_TPU_STABLEHLOKERNELEXPREMITTER_H
#define LIBND4J_TPU_STABLEHLOKERNELEXPREMITTER_H

#include <system/common.h>

#ifdef SD_TPU

#include <graph/kernelspec/KernelExpr.h>

#include <sstream>
#include <string>
#include <vector>

namespace sd {
namespace graph {

struct StableHloExprResult {
  bool success = false;
  std::string value;
  bool booleanValue = false;
  std::string error;
};

/**
 * StableHLO target sink for the shared KernelExpr semantic AST.
 *
 * This class contains no ND4J operation names or formulas. Normal operation
 * semantics live once in KernelSpec/KernelExpr; this sink only maps primitive
 * expression nodes to StableHLO MLIR syntax.
 */
class SD_LIB_EXPORT StableHloKernelExprEmitter {
 public:
  static StableHloExprResult emit(
      const kernelspec::ExprGraph& expression,
      const std::vector<std::string>& inputs,
      const std::vector<double>& scalarValues,
      const std::string& tensorType,
      const std::string& booleanTensorType,
      int& nextValueId,
      std::ostringstream& body);

 private:
  StableHloKernelExprEmitter() = delete;
};

}  // namespace graph
}  // namespace sd

#endif  // SD_TPU
#endif  // LIBND4J_TPU_STABLEHLOKERNELEXPREMITTER_H
