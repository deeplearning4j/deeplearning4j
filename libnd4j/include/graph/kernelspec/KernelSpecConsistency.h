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

#ifndef LIBND4J_KERNELSPEC_KERNELSPECCONSISTENCY_H
#define LIBND4J_KERNELSPEC_KERNELSPECCONSISTENCY_H

#include <string>
#include <vector>

// Drift detection between KernelSpecs and the hand-maintained per-op catalogs
// (ADR-0116 phase 0). Today it cross-checks the shared OpCategoryTable; the
// Triton buildOpTable and Vulkan catalog checks land when those tables expose
// enumeration. Nothing calls this yet — it is the seed of the future
// startup/CI consistency gate.

namespace sd {
namespace kernelspec {

struct KernelSpecCatalogIssue {
  std::string specName;    // primary spec name
  std::string lookupName;  // the name/alias that was checked
  std::string kind;        // MISSING_IN_OP_CATEGORY_TABLE | CATEGORY_MISMATCH
  std::string detail;
};

// Registered pilot specs must be present before calling (see
// registerPilotKernelSpecs()); checks every registered spec's name and
// aliases against getOpCategoryTable().
std::vector<KernelSpecCatalogIssue> crossCheckKernelSpecsWithOpCategoryTable();

std::string formatKernelSpecCatalogReport(const std::vector<KernelSpecCatalogIssue>& issues);

// Registers the pilot specs (idempotent). Defined in KernelSpecPilots.cpp.
void registerPilotKernelSpecs();

}  // namespace kernelspec
}  // namespace sd

#endif  // LIBND4J_KERNELSPEC_KERNELSPECCONSISTENCY_H
