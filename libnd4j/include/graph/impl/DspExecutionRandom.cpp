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

#include <graph/DspExecutionRandom.h>

namespace sd {
namespace graph {

static thread_local RandomGenerator* tl_dspExecutionRandom = nullptr;

RandomGenerator* dspExecutionRandom() { return tl_dspExecutionRandom; }

DspExecutionRandomScope::DspExecutionRandomScope(RandomGenerator* generator) : previous_(tl_dspExecutionRandom) {
  tl_dspExecutionRandom = generator;
}

DspExecutionRandomScope::~DspExecutionRandomScope() { tl_dspExecutionRandom = previous_; }

SlotRandomStateScope::SlotRandomStateScope(bool drawsRandomState, Context& context)
    : execution_(drawsRandomState ? tl_dspExecutionRandom : nullptr), context_(context) {
  if (execution_ != nullptr) context_.setRng(*execution_);
}

SlotRandomStateScope::~SlotRandomStateScope() {
  if (execution_ != nullptr) *execution_ = context_.getRng();
}

}  // namespace graph
}  // namespace sd
