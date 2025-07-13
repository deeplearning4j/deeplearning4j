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


#include <array/ArrayOptions.h>
#include <array/ArrayOptions.hXX>

// This compilation unit ensures that all ArrayOptions functions
// defined in ArrayOptions.hXX are compiled into object code.
// This is necessary because the functions are declared in ArrayOptions.h
// but implemented in ArrayOptions.hXX, and without this compilation unit,
// the linker would not find the symbol definitions when using clang.

namespace sd {

// Force explicit instantiation of the inline functions that are causing linker errors
// This ensures they are available as symbols for the linker, especially with Clang

// Create a dummy function that references all the problematic symbols
// This forces Clang to generate actual symbols for these inline functions
static void force_arrayoptions_symbol_generation() {
 // This function is never called, but its presence forces symbol generation
 LongType dummyShape[] = {2, 3, 4, 1, 1, 0, 1, 99};
 const LongType* constDummyShape = dummyShape;
 LongType dummyShape2[] = {2, 3, 4, 1, 1, 0, 1, 99};

 // Reference each problematic function to force symbol generation
 (void)ArrayOptions::extraIndex(dummyShape);
 (void)ArrayOptions::extra(constDummyShape);
 (void)ArrayOptions::arrayType(dummyShape);
 (void)ArrayOptions::dataType(constDummyShape);
 (void)ArrayOptions::enumerateSetFlags(constDummyShape);
 ArrayOptions::copyDataType(dummyShape2, constDummyShape);
 (void)ArrayOptions::flagForDataType(sd::DataType::FLOAT32);
 (void)ArrayOptions::hasPropertyBitSet(constDummyShape,0);
 (void)ArrayOptions::validateSingleDataType(0);
 (void)ArrayOptions::setDataType(dummyShape,sd::DataType::FLOAT32);
 (void)ArrayOptions::setExtra(dummyShape,0);
 (void)ArrayOptions::arrayNeedsCopy(dummyShape);
 (void)ArrayOptions::togglePropertyBit(dummyShape,0);
 (void)ArrayOptions::toggleIsEmpty(constDummyShape);
 (void)ArrayOptions::setPropertyBit(dummyShape,0);
 (void)ArrayOptions::propertyWithoutDataTypeValue(0);
 (void)ArrayOptions::setPropertyBits(dummyShape2,{0});
 (void)ArrayOptions::setDataTypeValue(0,sd::DataType::FLOAT32);
 (void)ArrayOptions::flagForDataType(sd::DataType::FLOAT32);
 (void)ArrayOptions::copyDataType(dummyShape,dummyShape);
 (void)ArrayOptions::enumerateSetFlags(dummyShape);

}

// Ensure the dummy function itself isn't optimized away by taking its address
static void* dummy_ref = (void*)&force_arrayoptions_symbol_generation;

// No additional implementation needed beyond the inclusion of ArrayOptions.hXX
// The forced instantiation above ensures symbols are available for linking
}