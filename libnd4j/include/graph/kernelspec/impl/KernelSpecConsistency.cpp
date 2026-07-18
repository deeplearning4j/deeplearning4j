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

#include <graph/gpu/OpCategoryTable.h>
#include <graph/kernelspec/KernelSpec.h>
#include <graph/kernelspec/KernelSpecConsistency.h>

#include <sstream>

namespace sd {
namespace kernelspec {

// getOpCategoryTable() is compiled whenever its guard holds; config.h always
// defines HAVE_MLIR (0 or 1), so `defined(HAVE_MLIR)` in that guard is true on
// every build that includes <config.h>. Mirror the guard here so this file
// degrades to a stub rather than failing on an exotic configuration.
#if defined(SD_CUDA) || defined(HAVE_MLIR) || HAVE_TRITON || HAVE_MLX

namespace {

const char* expectedCategoryName(KernelCategory c) {
  // TritonOpCategory names for the authorable subset (1:1 by design).
  switch (c) {
    case KernelCategory::UNARY_ELEMENTWISE: return "UNARY_ELEMENTWISE";
    case KernelCategory::BINARY_ELEMENTWISE: return "BINARY_ELEMENTWISE";
    case KernelCategory::TERNARY_ELEMENTWISE: return "TERNARY";
    case KernelCategory::COMPARISON: return "COMPARISON";
    case KernelCategory::LOGICAL: return "LOGICAL";
    case KernelCategory::REDUCTION: return "REDUCTION";
    case KernelCategory::IDENTITY: return "IDENTITY";
  }
  return "?";
}

graph::TritonOpCategory expectedCategory(KernelCategory c) {
  switch (c) {
    case KernelCategory::UNARY_ELEMENTWISE: return graph::TritonOpCategory::UNARY_ELEMENTWISE;
    case KernelCategory::BINARY_ELEMENTWISE: return graph::TritonOpCategory::BINARY_ELEMENTWISE;
    case KernelCategory::TERNARY_ELEMENTWISE: return graph::TritonOpCategory::TERNARY;
    case KernelCategory::COMPARISON: return graph::TritonOpCategory::COMPARISON;
    case KernelCategory::LOGICAL: return graph::TritonOpCategory::LOGICAL;
    case KernelCategory::REDUCTION: return graph::TritonOpCategory::REDUCTION;
    case KernelCategory::IDENTITY: return graph::TritonOpCategory::IDENTITY;
  }
  return graph::TritonOpCategory::UNSUPPORTED;
}

const char* tritonCategoryName(graph::TritonOpCategory c) {
  switch (c) {
    case graph::TritonOpCategory::BINARY_ELEMENTWISE: return "BINARY_ELEMENTWISE";
    case graph::TritonOpCategory::UNARY_ELEMENTWISE: return "UNARY_ELEMENTWISE";
    case graph::TritonOpCategory::COMPARISON: return "COMPARISON";
    case graph::TritonOpCategory::LOGICAL: return "LOGICAL";
    case graph::TritonOpCategory::TERNARY: return "TERNARY";
    case graph::TritonOpCategory::IDENTITY: return "IDENTITY";
    case graph::TritonOpCategory::MATMUL: return "MATMUL";
    case graph::TritonOpCategory::REDUCTION: return "REDUCTION";
    case graph::TritonOpCategory::NORMALIZATION: return "NORMALIZATION";
    case graph::TritonOpCategory::CAST: return "CAST";
    case graph::TritonOpCategory::FUSED_ATTENTION: return "FUSED_ATTENTION";
    case graph::TritonOpCategory::SHAPE_MANIPULATION: return "SHAPE_MANIPULATION";
    case graph::TritonOpCategory::DATA_MOVEMENT: return "DATA_MOVEMENT";
    case graph::TritonOpCategory::CONSTANT_GENERATION: return "CONSTANT_GENERATION";
    case graph::TritonOpCategory::CONVOLUTION: return "CONVOLUTION";
    case graph::TritonOpCategory::ROPE: return "ROPE";
    case graph::TritonOpCategory::FUSED_LLM: return "FUSED_LLM";
    case graph::TritonOpCategory::UNSUPPORTED: return "UNSUPPORTED";
  }
  return "?";
}

}  // namespace

std::vector<KernelSpecCatalogIssue> crossCheckKernelSpecsWithOpCategoryTable() {
  std::vector<KernelSpecCatalogIssue> issues;
  const auto& table = graph::getOpCategoryTable();

  for (const KernelSpec* spec : KernelSpecRegistry::getInstance().all()) {
    std::vector<std::string> names;
    names.push_back(spec->name);
    for (const auto& a : spec->aliases) names.push_back(a);

    const auto expected = expectedCategory(spec->category);
    for (const auto& lookup : names) {
      auto it = table.find(lookup);
      if (it == table.end()) {
        KernelSpecCatalogIssue issue;
        issue.specName = spec->name;
        issue.lookupName = lookup;
        issue.kind = "MISSING_IN_OP_CATEGORY_TABLE";
        issue.detail = std::string("spec category ") + expectedCategoryName(spec->category) +
                       " has no OpCategoryTable entry for this name";
        issues.push_back(issue);
        continue;
      }
      if (it->second != expected) {
        KernelSpecCatalogIssue issue;
        issue.specName = spec->name;
        issue.lookupName = lookup;
        issue.kind = "CATEGORY_MISMATCH";
        issue.detail = std::string("spec says ") + expectedCategoryName(spec->category) +
                       " but OpCategoryTable says " + tritonCategoryName(it->second);
        issues.push_back(issue);
      }
    }
  }
  return issues;
}

#else

std::vector<KernelSpecCatalogIssue> crossCheckKernelSpecsWithOpCategoryTable() { return {}; }

#endif  // guard mirroring OpCategoryTable.h

std::string formatKernelSpecCatalogReport(const std::vector<KernelSpecCatalogIssue>& issues) {
  std::ostringstream out;
  out << "KernelSpec catalog cross-check: " << issues.size() << " issue(s), "
      << KernelSpecRegistry::getInstance().size() << " spec(s) checked\n";
  for (const auto& issue : issues) {
    out << "  [" << issue.kind << "] " << issue.specName;
    if (issue.lookupName != issue.specName) out << " (as '" << issue.lookupName << "')";
    out << ": " << issue.detail << "\n";
  }
  return out.str();
}

}  // namespace kernelspec
}  // namespace sd
