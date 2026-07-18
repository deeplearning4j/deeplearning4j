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
// Split from NativeOpsHelpers_DataBuffers.cpp to reduce object file size
// Contains: NPZ/numpy interop functions
//

#include <legacy/NativeOps.h>
#include <helpers/logger.h>
#include <cstring>
#include <string>


/*
 * TypeDef:
 *     void convertTypes(Pointer *extras, DataType srcType, Pointer hX, long N, DataType dstType, Pointer hZ);
 */
void* mapFromNpzFile(std::string path) {
  cnpy::npz_t* mapPtr = new cnpy::npz_t();
  cnpy::npz_t map = cnpy::npzLoad(path);
  mapPtr->insert(map.begin(), map.end());
  return reinterpret_cast<void*>(mapPtr);
}

int getNumNpyArraysInMap(void* map) {
  cnpy::npz_t* arrays = reinterpret_cast<cnpy::npz_t*>(map);
  int n = arrays->size();
  return n;
}

const char* getNpyArrayNameFromMap(void* map, int index, char* nameBuffer) {
  cnpy::npz_t* arrays = reinterpret_cast<cnpy::npz_t*>(map);
  cnpy::npz_t::iterator it = arrays->begin();
  cnpy::npz_t::iterator end = arrays->end();
  int cnt = 0;
  for (; it != end; ++it, ++cnt) {
    if (cnt == index) {
      size_t len_of_str = strlen(it->first.c_str());
      memcpy(nameBuffer, it->first.c_str(), len_of_str);
    }
  }
  return "";
}

void* getNpyArrayFromMap(void* map, int index) {
  cnpy::npz_t* arrays = reinterpret_cast<cnpy::npz_t*>(map);
  cnpy::npz_t::iterator it = arrays->begin();
  cnpy::npz_t::iterator end = arrays->end();
  cnpy::NpyArray* arr = new cnpy::NpyArray();
  int cnt = 0;
  for (; it != end; ++it, ++cnt) {
    if (cnt == index) {
      *arr = it->second;
      return arr;
    }
  }

 return nullptr;
}


void* getNpyArrayData(void* npArray) {
  cnpy::NpyArray* npyArray2 = reinterpret_cast<cnpy::NpyArray*>(npArray);
  return reinterpret_cast<void*>(npyArray2->data);
}

int getNpyArrayRank(void* npArray) {
  cnpy::NpyArray* arr = reinterpret_cast<cnpy::NpyArray*>(npArray);
  int rank = arr->shape.size();
  return rank;
}

sd::LongType* getNpyArrayShape(void* npArray) {
  cnpy::NpyArray* arr = reinterpret_cast<cnpy::NpyArray*>(npArray);
  int ndim = arr->shape.size();
  sd::LongType* shape = new sd::LongType[ndim];
  for (int i = 0; i < ndim; i++) {
    shape[i] = arr->shape.at(i);
  }
  return shape;
}

char getNpyArrayOrder(void* npArray) {
  cnpy::NpyArray* arr = reinterpret_cast<cnpy::NpyArray*>(npArray);
  return (arr->fortranOrder) ? 'f' : 'c';
}

int getNpyArrayElemSize(void* npArray) {
  cnpy::NpyArray* arr = reinterpret_cast<cnpy::NpyArray*>(npArray);
  return arr->wordSize;
}

void deleteNPArrayStruct(void* npArray) {
  cnpy::NpyArray* arr = reinterpret_cast<cnpy::NpyArray*>(npArray);
  delete arr;
}

void deleteNPArrayMap(void* map) {
  cnpy::npz_t* arrays = reinterpret_cast<cnpy::npz_t*>(map);
  delete arrays;
}

/**
 * Get the element size for a numpy array
 * @param npyArray  the numpy array's address
 * to get the length for
 * @return
 */
int elementSizeForNpyArray(sd::Pointer npyArray) {
  cnpy::NpyArray arr = cnpy::loadNpyFromPointer(reinterpret_cast<char*>(npyArray));
  cnpy::NpyArray* arrPointer = &arr;
  int size = arrPointer->wordSize;
  return size;
}

/**
 * Get the element size for a numpy array
 * @param npyArray  the numpy array's address
 * to get the length for
 * @return
 */
int elementSizeForNpyArrayHeader(sd::Pointer npyArray) {
  cnpy::NpyArray arr = cnpy::loadNpyFromHeader(reinterpret_cast<char*>(npyArray));
  cnpy::NpyArray* arrPointer = &arr;
  int size = arrPointer->wordSize;
  return size;
}

void releaseNumpy(sd::Pointer npyArray) { free(reinterpret_cast<void*>(npyArray)); }


