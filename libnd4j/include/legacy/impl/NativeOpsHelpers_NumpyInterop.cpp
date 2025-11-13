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

#include <graph/GraphExecutioner.h>
#include <graph/GraphHolder.h>
#include <helpers/ConstantTadHelper.h>
#include <legacy/NativeOps.h>
#include <ops/declarable/OpRegistrator.h>

#include "execution/Threads.h"
#include "helpers/OpTracker.h"

#include <exceptions/allocation_exception.h>
#include <fcntl.h>
#include <graph/GraphExecutioner.h>

#include <helpers/BlasHelper.h>
#include <helpers/helper_ptrmap.h>
#include <helpers/logger.h>
#include <legacy/NativeOpExecutioner.h>
#include <legacy/NativeOps.h>
#include <loops/type_conversions.h>
#include <math/templatemath.h>
#include <ops/declarable/helpers/transforms.h>
#include <stdio.h>
#include <stdlib.h>
#include <types/float8.h>
#include <types/types.h>
#ifndef _WIN32
#include <sys/mman.h>
#include <unistd.h>

#else
#include <helpers/mman.h>
#include <io.h>
#endif
#include <errno.h>
#include <ops/declarable/CustomOperations.h>
#include <sys/types.h>


extern bool experimentalSupport; // Defined in NativeOpsHelpers_Arrays.cpp

// OpaqueNDArray allocation tracking
static std::atomic<size_t> g_opaqueArrayCount{0};
static std::atomic<size_t> g_opaqueArrayBytes{0};
static std::mutex g_opaqueArrayMutex;

// InteropDataBuffer/OpaqueDataBuffer allocation tracking
static std::atomic<size_t> g_dataBufferCount{0};
static std::atomic<size_t> g_dataBufferBytes{0};
static std::mutex g_dataBufferMutex;

#include <execution/Threads.h>
#include <graph/Context.h>
#include <graph/ResultWrapper.h>
#include <helpers/ConstantTadHelper.h>
#include <helpers/DebugHelper.h>

#include <ops/declarable/OpRegistrator.h>
#include <ops/specials.h>
#include <system/Environment.h>
#ifdef CPU_FEATURES
#include <cpuinfo_x86.h>
#endif
#include <array/DataType.h>
#include <array/DataTypeUtils.h>




/*
 * TypeDef:
 *     void convertTypes(Pointer *extras, DataType srcType, Pointer hX, long N, DataType dstType, Pointer hZ);
 */
static long lengthInBytes(OpaqueDataBuffer *buffer) {
  return buffer->dataBuffer()->getLenInBytes();
}

template <typename T>
static sd::Pointer _numpyHeaderForNd4j(sd::Pointer data, const sd::Pointer shapeBuffer, sd::LongType wordSize,
                                       sd::LongType* headerSize) {
  sd::LongType const* shapeBufferCast = reinterpret_cast<const sd::LongType*>(shapeBuffer);
  int rank = shape::rank(shapeBufferCast);
  const sd::LongType* shape = shape::shapeOf(shapeBufferCast);
  unsigned int* npShape = new unsigned int[rank];
  for (int i = 0; i < rank; i++) {
    npShape[i] = shape[i];
  }

  sd::LongType length = shape::prodLong(shape, rank);
  auto npHeader = cnpy::createNpyHeader<T>(npShape, rank, wordSize);
  
  // FIX: Clean up npShape after use
  delete[] npShape;
  
  char* ret = new char[npHeader.size() + 1];
  int count = 0;
  for (size_t i = 0; i < npHeader.size(); i++) {
    ret[count] = npHeader[i];
    count++;
  }

  ret[count] = '\0';
  count++;

  *headerSize = count;
  return reinterpret_cast<sd::Pointer>(ret);
}


sd::Pointer numpyHeaderForNd4j(sd::Pointer data, sd::Pointer shapeBuffer, sd::LongType wordSize,
                               sd::LongType* headerSize) {
  auto shapeBufferCast = reinterpret_cast<sd::LongType*>(shapeBuffer);
  auto type = sd::ArrayOptions::dataType(shapeBufferCast);
  BUILD_SINGLE_SELECTOR(type, return _numpyHeaderForNd4j, (data, shapeBuffer, wordSize, headerSize), SD_COMMON_TYPES);
  return nullptr;
}

/**
 * Load numpy from a header
 * based on the cnpy parse from header method.
 * @param data the header data to parse
 * @return a pointer to a numpy cnpy:NpyArray struct
 */
sd::Pointer loadNpyFromHeader(sd::Pointer data) {
  char* header = reinterpret_cast<char*>(data);

  cnpy::NpyArray arr = cnpy::loadNpyFromHeader(header);
  cnpy::NpyArray* ret = new cnpy::NpyArray();

  ret->data = arr.data;
  ret->wordSize = arr.wordSize;
  ret->shape = arr.shape;
  return reinterpret_cast<sd::Pointer>(ret);
}


/**
 * Create a numpy array from an nd4j
 * array
 * @param data a pointer to the data
 * @param shapeBuffer  the shapebuffer for the nd4j array
 * @param wordSize  the word size (4 for float, 8 for doubles)
 * @return a pointer to a numpy array
 */

template <typename T>
sd::Pointer _numpyFromNd4j(sd::Pointer data, sd::Pointer shapeBuffer, sd::LongType wordSize) {
  sd::LongType* shapeBufferCast = reinterpret_cast<sd::LongType*>(shapeBuffer);
  int rank = shape::rank(shapeBufferCast);
  sd::LongType* shape = shape::shapeOf(shapeBufferCast);
  unsigned int* npShape = new unsigned int[rank];
  for (int i = 0; i < rank; i++) {
    npShape[i] = shape[i];
  }

  sd::LongType length = shape::prodLong(shape, rank);
  auto npHeader = cnpy::createNpyHeader<T>( npShape, rank, wordSize);
  
  // FIX: Clean up npShape after use
  delete[] npShape;
  
  char* dataChar = reinterpret_cast<char*>(data);
  char* npHeaderData = npHeader.data();
  char* ret = new char[(wordSize * length) + npHeader.size()];
  char* cursorStart = ret + npHeader.size();
  std::memcpy(ret, npHeaderData,
              npHeader.size());
  std::memcpy(cursorStart, dataChar,length  * wordSize);
  sd::Pointer rettPointer = reinterpret_cast<sd::Pointer>(ret);
  return rettPointer;
}
template<typename T>
long _numpyHeaderLength(OpaqueDataBuffer *opaqueDataBuffer,sd::Pointer shapeBuffer) {
  sd::LongType wordSize = opaqueDataBuffer->dataBuffer()->getLenInBytes() / opaqueDataBuffer->dataBuffer()->getNumElements();
  sd::LongType* shapeBufferCast = reinterpret_cast<sd::LongType*>(shapeBuffer);
  int rank = shape::rank(shapeBufferCast);
  sd::LongType* shape = shape::shapeOf(shapeBufferCast);
  unsigned int* npShape = new unsigned int[rank];
  for (int i = 0; i < rank; i++) {
    npShape[i] = shape[i];
  }

  sd::LongType length = shape::prodLong(shape, rank);
  auto npHeader = cnpy::createNpyHeader<T>(npShape, rank, wordSize);
  long ret = npHeader.size();
  
  // FIX: Clean up npShape after use
  delete[] npShape;
  
  return ret;
}

template<typename  T>
long _numpyHeaderLengthWordSize(sd::Pointer shapeBuffer,long wordSize) {
  sd::LongType* shapeBufferCast = reinterpret_cast<sd::LongType*>(shapeBuffer);
  int rank = shape::rank(shapeBufferCast);
  sd::LongType* shape = shape::shapeOf(shapeBufferCast);
  unsigned int* npShape = new unsigned int[rank];
  for (int i = 0; i < rank; i++) {
    npShape[i] = shape[i];
  }

  sd::LongType length = shape::prodLong(shape, rank);
  auto npHeader = cnpy::createNpyHeader<T>(npShape, rank, wordSize);
  long ret = npHeader.size();
  
  // FIX: Clean up npShape after use
  delete[] npShape;
  
  return ret;
}



long numpyHeaderLengthWordSize(sd::Pointer shapeBuffer,long wordSize) {
  auto shapeBufferCast = reinterpret_cast<sd::LongType*>(shapeBuffer);
  auto type = sd::ArrayOptions::dataType(shapeBufferCast);
  BUILD_SINGLE_SELECTOR(type, return _numpyHeaderLengthWordSize, (shapeBuffer, wordSize), SD_COMMON_TYPES);
  return 0;

}

long numpyHeaderLength(OpaqueDataBuffer *opaqueDataBuffer,sd::Pointer shapeBuffer) {
  auto shapeBufferCast = reinterpret_cast<sd::LongType*>(shapeBuffer);
  auto type = sd::ArrayOptions::dataType(shapeBufferCast);

  BUILD_SINGLE_SELECTOR(type, return _numpyHeaderLength, (opaqueDataBuffer, shapeBuffer), SD_COMMON_TYPES);
  return 0;

}



sd::Pointer numpyFromNd4j(sd::Pointer data, sd::Pointer shapeBuffer, sd::LongType wordSize) {
  auto shapeBufferCast = reinterpret_cast<sd::LongType*>(shapeBuffer);
  auto type = sd::ArrayOptions::dataType(shapeBufferCast);

  BUILD_SINGLE_SELECTOR(type, return _numpyFromNd4j, (data, shapeBuffer, wordSize), SD_COMMON_TYPES);
  return nullptr;
}


sd::Pointer shapeBufferForNumpy(sd::Pointer npyArray) {
  try {
    cnpy::NpyArray arr = cnpy::loadNpyFromPointer(reinterpret_cast<char *>(npyArray));
    unsigned int shapeSize = arr.shape.size();
    std::vector<sd::LongType> shape(shapeSize);
    bool _empty = false;
    for (unsigned int i = 0; i < shapeSize; i++) {
      shape[i] = arr.shape[i];

      if (arr.shape[i] == 0) _empty = true;
    }

    auto dtype = cnpy::dataTypeFromHeader(reinterpret_cast<char *>(npyArray));

    sd::LongType *shapeBuffer;
    if (shape.size() == 1 && shape[0] == 0) {
      // scalar case
      shapeBuffer = sd::ShapeBuilders::createScalarShapeInfo(dtype);
    } else if (_empty) {
      if (shapeSize > 0)
        shapeBuffer = sd::ShapeBuilders::emptyShapeInfo(dtype, arr.fortranOrder ? 'f' : 'c', shape);
      else
        shapeBuffer = sd::ShapeBuilders::emptyShapeInfo(dtype);
    } else {
      shapeBuffer = sd::ShapeBuilders::createShapeInfo(dtype, arr.fortranOrder ? 'f' : 'c', shape);
    }
    return (sd::Pointer)(sd::ConstantShapeHelper::getInstance().createFromExisting(
        shapeBuffer));  // TO DO: this can lead to unpleasant crash sometimes
  } catch (std::exception &e) {
    sd::LaunchContext::defaultContext()->errorReference()->setErrorCode(1);
    sd::LaunchContext::defaultContext()->errorReference()->setErrorMessage(e.what());
    return nullptr;
  }
}

