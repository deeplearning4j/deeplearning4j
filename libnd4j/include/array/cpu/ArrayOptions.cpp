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

// Force explicit template instantiations for all the problematic functions
// This approach directly tells the compiler to generate symbols

// Explicit instantiation of function templates
template sd::LongType ArrayOptions::extraIndex<sd::LongType*>(sd::LongType*);
template sd::LongType ArrayOptions::extraIndex<const sd::LongType*>(const sd::LongType*);

// Force non-inline versions of all the problematic functions by creating wrapper functions
// that call the inline versions. This ensures symbols are generated.

extern "C" {
// Create C-linkage wrapper functions that are guaranteed not to be inlined
sd::LongType __arrayoptions_extraindex_nonconst(sd::LongType* shapeInfo) {
 return ArrayOptions::extraIndex(shapeInfo);
}

sd::LongType __arrayoptions_extraindex_const(const sd::LongType* shapeInfo) {
 return ArrayOptions::extraIndex(shapeInfo);
}

sd::LongType __arrayoptions_extra(const sd::LongType* shapeInfo) {
 return ArrayOptions::extra(shapeInfo);
}

sd::DataType __arrayoptions_datatype(const sd::LongType* shapeInfo) {
 return ArrayOptions::dataType(shapeInfo);
}

bool __arrayoptions_haspropertybits(const sd::LongType* shapeInfo, sd::LongType property) {
 return ArrayOptions::hasPropertyBitSet(shapeInfo, property);
}

void __arrayoptions_validatesingle(sd::LongType property) {
 ArrayOptions::validateSingleDataType(property);
}

void __arrayoptions_setdatatype(sd::LongType* shapeInfo, sd::DataType dataType) {
 ArrayOptions::setDataType(shapeInfo, dataType);
}

void __arrayoptions_setextra(sd::LongType* shapeInfo, sd::LongType value) {
 ArrayOptions::setExtra(shapeInfo, value);
}

sd::ArrayType __arrayoptions_arraytype_nonconst(sd::LongType* shapeInfo) {
 return ArrayOptions::arrayType(shapeInfo);
}

sd::ArrayType __arrayoptions_arraytype_const(const sd::LongType* shapeInfo) {
 return ArrayOptions::arrayType(shapeInfo);
}

const char* __arrayoptions_enumeratesetflags(const sd::LongType* shapeInfo) {
 return ArrayOptions::enumerateSetFlags(shapeInfo);
}

void __arrayoptions_copydatatype(sd::LongType* to, const sd::LongType* from) {
 ArrayOptions::copyDataType(to, from);
}

sd::LongType __arrayoptions_flagfordatatype(sd::DataType dataType) {
 return ArrayOptions::flagForDataType(dataType);
}

bool __arrayoptions_arrayneedscopy(sd::LongType* shapeInfo) {
 return ArrayOptions::arrayNeedsCopy(shapeInfo);
}

bool __arrayoptions_togglepropertybits(sd::LongType* shapeInfo, sd::LongType property) {
 return ArrayOptions::togglePropertyBit(shapeInfo, property);
}

void __arrayoptions_toggleisempty(sd::LongType* shapeInfo) {
 ArrayOptions::toggleIsEmpty(shapeInfo);
}

void __arrayoptions_setpropertybits(sd::LongType* shapeInfo, sd::LongType property) {
 ArrayOptions::setPropertyBit(shapeInfo, property);
}

sd::LongType __arrayoptions_propertywithoutdatatype(sd::LongType extra) {
 return ArrayOptions::propertyWithoutDataTypeValue(extra);
}

sd::LongType __arrayoptions_setdatatypevalue(sd::LongType extra, sd::DataType dataType) {
 return ArrayOptions::setDataTypeValue(extra, dataType);
}
}

// Create a function table that references all the wrapper functions
// This absolutely ensures they won't be optimized away
struct FunctionTable {
 void* extraIndex1;
 void* extraIndex2;
 void* extra;
 void* dataType;
 void* hasPropertyBitSet;
 void* validateSingleDataType;
 void* setDataType;
 void* setExtra;
 void* arrayType1;
 void* arrayType2;
 void* enumerateSetFlags;
 void* copyDataType;
 void* flagForDataType;
 void* arrayNeedsCopy;
 void* togglePropertyBit;
 void* toggleIsEmpty;
 void* setPropertyBit;
 void* propertyWithoutDataTypeValue;
 void* setDataTypeValue;
};

// Initialize the function table with all the wrapper functions
static FunctionTable g_function_table = {
   (void*)__arrayoptions_extraindex_nonconst,
   (void*)__arrayoptions_extraindex_const,
   (void*)__arrayoptions_extra,
   (void*)__arrayoptions_datatype,
   (void*)__arrayoptions_haspropertybits,
   (void*)__arrayoptions_validatesingle,
   (void*)__arrayoptions_setdatatype,
   (void*)__arrayoptions_setextra,
   (void*)__arrayoptions_arraytype_nonconst,
   (void*)__arrayoptions_arraytype_const,
   (void*)__arrayoptions_enumeratesetflags,
   (void*)__arrayoptions_copydatatype,
   (void*)__arrayoptions_flagfordatatype,
   (void*)__arrayoptions_arrayneedscopy,
   (void*)__arrayoptions_togglepropertybits,
   (void*)__arrayoptions_toggleisempty,
   (void*)__arrayoptions_setpropertybits,
   (void*)__arrayoptions_propertywithoutdatatype,
   (void*)__arrayoptions_setdatatypevalue
};

// Export a function that uses the table to prevent optimization
extern "C" void* get_arrayoptions_function_table() {
 return &g_function_table;
}

}  // namespace sd