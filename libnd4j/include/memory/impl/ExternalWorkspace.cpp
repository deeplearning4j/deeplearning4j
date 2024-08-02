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
//  @author raver119@gmail.com
//
#include <memory/ExternalWorkspace.h>

namespace sd {
namespace memory {
ExternalWorkspace::ExternalWorkspace(Pointer ptrH, LongType sizeH, Pointer ptrD, LongType sizeD) {
  _ptrH = ptrH;
  _sizeH = sizeH;

  _ptrD = ptrD;
  _sizeD = sizeD;
};

void* ExternalWorkspace::pointerHost() { return _ptrH; }

void* ExternalWorkspace::pointerDevice() { return _ptrD; }

LongType ExternalWorkspace::sizeHost() { return _sizeH; }

LongType ExternalWorkspace::sizeDevice() { return _sizeD; }
}  // namespace memory
}  // namespace sd
