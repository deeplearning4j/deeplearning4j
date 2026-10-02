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

#ifndef LIBND4J_DSP_EXECUTION_RANDOM_H
#define LIBND4J_DSP_EXECUTION_RANDOM_H

#include <graph/Context.h>
#include <graph/RandomGenerator.h>

namespace sd {
namespace graph {

/**
 * The random generator of the DSP plan execution running on this thread: the plan entry
 * context's generator, which its caller (DynamicShapePlanExecutor) seeds from Nd4j.getRandom()
 * and reads back after the execution. Null outside a plan execution.
 */
SD_LIB_HIDDEN RandomGenerator* dspExecutionRandom();

/** Makes a plan execution's generator current on this thread for its scope. */
class SD_LIB_HIDDEN DspExecutionRandomScope {
 public:
  explicit DspExecutionRandomScope(RandomGenerator* generator);
  ~DspExecutionRandomScope();
  DspExecutionRandomScope(const DspExecutionRandomScope&) = delete;
  DspExecutionRandomScope& operator=(const DspExecutionRandomScope&) = delete;

 private:
  RandomGenerator* previous_;
};

/**
 * Executes a slot that draws random state (NativeSlot::drawsRandomState) with the plan
 * execution's generator: its step context takes the execution's state, and the execution takes
 * back the state the op advanced, so successive executions draw anew and the caller can reseed.
 * A slot that draws nothing, or one run outside a plan execution, keeps its context as it is.
 */
class SD_LIB_HIDDEN SlotRandomStateScope {
 public:
  SlotRandomStateScope(bool drawsRandomState, Context& context);
  ~SlotRandomStateScope();
  SlotRandomStateScope(const SlotRandomStateScope&) = delete;
  SlotRandomStateScope& operator=(const SlotRandomStateScope&) = delete;

 private:
  RandomGenerator* execution_;
  Context& context_;
};

}  // namespace graph
}  // namespace sd

#endif  // LIBND4J_DSP_EXECUTION_RANDOM_H
