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

#include <array/ArrayOptions.h>
#include <array/ArrayOptions.hXX>

// This compilation unit ensures that all ArrayOptions functions 
// defined in ArrayOptions.hXX are compiled into object code.
// This is necessary because the functions are declared in ArrayOptions.h
// but implemented in ArrayOptions.hXX, and without this compilation unit,
// the linker would not find the symbol definitions when using clang.

namespace sd {
// No additional implementation needed - the inclusion of ArrayOptions.hXX
// provides all the function definitions that were missing at link time.
}