/* ******************************************************************************
*
*
* This program and the accompanying materials are made available under the
* terms of the Apache License, Version 2.0 which is available at
* https://www.apache.org/licenses/LICENSE-2.0.
*
*  See the NOTICE file distributed with this work for additional
*  information regarding copyright ownership.
* Unless required by applicable law or agreed to in writing,
* software distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See
* the License for the specific language governing permissions and limitations
* under the License.
*
* SPDX-License-Identifier: Apache-2.0
******************************************************************************/

//
// Created by agibsonccc on 11/6/24.
//
#include <legacy/NativeOpExecutioner.h>
#include <system/selective_rendering.h>
void NativeOpExecutioner::execSort(sd::NDArray *x, bool descending) {
  auto xType = x->dataType();
#if SD_IS_SINGLE_TYPE_COMPILED(xType)
  BUILD_SINGLE_SELECTOR(xType, sd::SpecialMethods, ::sortGeneric(x, descending), SD_NUMERIC_TYPES);
#endif
}

 void NativeOpExecutioner::execSort(sd::NDArray *x, sd::LongType *dimension,  sd::LongType dimensionLength,
                     bool descending) {
  auto xType = x->dataType();
#if SD_IS_SINGLE_TYPE_COMPILED(xType)
  BUILD_SINGLE_SELECTOR(
      xType, sd::SpecialMethods,
      ::sortTadGeneric(x, dimension, dimensionLength, descending),
      SD_NUMERIC_TYPES);
#endif
}

