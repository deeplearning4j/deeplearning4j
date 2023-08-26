#include <array/ArrayOptions.h>

namespace sd {
SD_HOST_DEVICE sd::LongType ArrayOptions::extraIndex(const sd::LongType *shapeInfo) {
  return ArrayOptions::extraIndex(const_cast<sd::LongType *>(shapeInfo));
}


SD_HOST_DEVICE sd::LongType ArrayOptions::extraIndex(sd::LongType *shapeInfo) {
  sd::LongType  rank = shapeInfo[0];
  sd::LongType  idx = 0;
  //rank takes up 1 element + usual elements
  if(rank == 0)
    idx = 3;
  else
    // FIXME magic numbers
    idx = rank + rank + 1;
  return idx;
}


SD_HOST_DEVICE void ArrayOptions::setExtra(sd::LongType *shapeInfo, sd::LongType value) {
  sd::LongType  idx = ArrayOptions::extraIndex(shapeInfo);
  shapeInfo[idx] = value;
}

SD_HOST_DEVICE LongType ArrayOptions::extra(sd::LongType *shapeInfo) {
  sd::LongType  rank = shapeInfo[0];
  sd::LongType  idx = ArrayOptions::extraIndex(shapeInfo);
  return shapeInfo[idx];
}

SD_HOST_DEVICE bool ArrayOptions::isNewFormat(const sd::LongType *shapeInfo) {
  return (extra(const_cast<sd::LongType *>(shapeInfo)) != 0);
}

SD_HOST_DEVICE bool ArrayOptions::isSparseArray(sd::LongType *shapeInfo) {
  return hasPropertyBitSet(shapeInfo, ARRAY_SPARSE);
}

SD_HOST_DEVICE bool ArrayOptions::hasExtraProperties(sd::LongType *shapeInfo) {
  return hasPropertyBitSet(shapeInfo, ARRAY_EXTRAS);
}

SD_HOST_DEVICE bool ArrayOptions:: hasPropertyBitSet(const sd::LongType *shapeInfo, int property) {
  if (!isNewFormat(shapeInfo)) return false;

  return ((extra(const_cast<sd::LongType *>(shapeInfo)) & property) == property);
}


SD_HOST_DEVICE bool hasPropertyBitSetForFlags(const sd::LongType& flagStorage, int property) {
  return (flagStorage & property) == property;
}

SD_HOST_DEVICE void unsetPropertyBitForFlags(sd::LongType& flagStorage, int property) {
  flagStorage &= ~property;
}

SD_HOST_DEVICE int ArrayOptions::enumerateSetFlagsForFlags(const sd::LongType& flagStorage, const char* setFlagsOutput[], int maxFlags) {
  int setFlagCount = 0;
  int flagsArray[] = {
      ARRAY_SPARSE,
      ARRAY_COMPRESSED,
      ARRAY_EMPTY,
      ARRAY_RAGGED,
      ARRAY_CSR,
      ARRAY_CSC,
      ARRAY_COO,
      ARRAY_COMPLEX,
      ARRAY_QUANTIZED,
      ARRAY_HALF,
      ARRAY_BHALF,
      ARRAY_FLOAT,
      ARRAY_DOUBLE,
      ARRAY_CHAR,
      ARRAY_SHORT,
      ARRAY_INT,
      ARRAY_LONG,
      ARRAY_BOOL,
      ARRAY_UTF8,
      ARRAY_UTF16,
      ARRAY_UTF32,
      ARRAY_EXTRAS,
      ARRAY_UNSIGNED,
      ARRAY_HAS_PADDED_BUFFER
  };

  const char* flagsStrings[] = {
      "ARRAY_SPARSE",
      "ARRAY_COMPRESSED",
      "ARRAY_EMPTY",
      "ARRAY_RAGGED",
      "ARRAY_CSR",
      "ARRAY_CSC",
      "ARRAY_COO",
      "ARRAY_COMPLEX",
      "ARRAY_QUANTIZED",
      "ARRAY_HALF",
      "ARRAY_BHALF",
      "ARRAY_FLOAT",
      "ARRAY_DOUBLE",
      "ARRAY_CHAR",
      "ARRAY_SHORT",
      "ARRAY_INT",
      "ARRAY_LONG",
      "ARRAY_BOOL",
      "ARRAY_UTF8",
      "ARRAY_UTF16",
      "ARRAY_UTF32",
      "ARRAY_EXTRAS",
      "ARRAY_UNSIGNED",
      "ARRAY_HAS_PADDED_BUFFER"
  };
  for (int i = 0; i <  setFlagCount < maxFlags; i++) {
    if (hasPropertyBitSetForFlags(flagStorage, flagsArray[i])) {
      setFlagsOutput[setFlagCount++] = flagsStrings[i];
    }
  }

  return setFlagCount;  // Returns the number of set flags found
}

SD_HOST_DEVICE  void ArrayOptions::unsetAllFlags(sd::LongType& flagStorage) {

  int flagsArray[] = {
      ARRAY_SPARSE,
      ARRAY_COMPRESSED,
      ARRAY_EMPTY,
      ARRAY_RAGGED,
      ARRAY_CSR,
      ARRAY_CSC,
      ARRAY_COO,
      ARRAY_COMPLEX,
      ARRAY_QUANTIZED,
      ARRAY_HALF,
      ARRAY_BHALF,
      ARRAY_FLOAT,
      ARRAY_DOUBLE,
      ARRAY_CHAR,
      ARRAY_SHORT,
      ARRAY_INT,
      ARRAY_LONG,
      ARRAY_BOOL,
      ARRAY_UTF8,
      ARRAY_UTF16,
      ARRAY_UTF32,
      ARRAY_EXTRAS,
      ARRAY_UNSIGNED,
      ARRAY_HAS_PADDED_BUFFER
  };

  for (int i = 0; i < sizeof(flagsArray)/sizeof(int); i++) {
    unsetPropertyBitForFlags(flagStorage, flagsArray[i]);
  }
}

SD_HOST_DEVICE int ArrayOptions::enumerateSetFlags(const sd::LongType *shapeInfo, const char* setFlagsOutput[], int maxFlags) {
  return enumerateSetFlagsForFlags(shapeInfo[ArrayOptions::extraIndex(shapeInfo)], setFlagsOutput, maxFlags);
}

SD_HOST_DEVICE  void ArrayOptions::unsetAllFlags(sd::LongType *shapeInfo) {
  ArrayOptions::unsetAllFlags(shapeInfo[ArrayOptions::extraIndex(shapeInfo)]);
}


SD_HOST_DEVICE bool ArrayOptions::isUnsigned(sd::LongType *shapeInfo) {
  if (!isNewFormat(shapeInfo)) return false;

  return hasPropertyBitSet(shapeInfo, ARRAY_UNSIGNED);
}

SD_HOST_DEVICE sd::DataType ArrayOptions::dataType(const sd::LongType *shapeInfo) {
  if (hasPropertyBitSet(shapeInfo, ARRAY_FLOAT))
    return sd::DataType::FLOAT32;
  else if (hasPropertyBitSet(shapeInfo, ARRAY_DOUBLE))
    return sd::DataType::DOUBLE;
  else if (hasPropertyBitSet(shapeInfo, ARRAY_HALF))
    return sd::DataType::HALF;
  else if (hasPropertyBitSet(shapeInfo, ARRAY_BHALF))
    return sd::DataType::BFLOAT16;
  else if (hasPropertyBitSet(shapeInfo, ARRAY_BOOL))
    return sd::DataType ::BOOL;
  else if (hasPropertyBitSet(shapeInfo, ARRAY_UNSIGNED)) {
    if (hasPropertyBitSet(shapeInfo, ARRAY_CHAR))
      return sd::DataType ::UINT8;
    else if (hasPropertyBitSet(shapeInfo, ARRAY_SHORT))
      return sd::DataType ::UINT16;
    else if (hasPropertyBitSet(shapeInfo, ARRAY_INT))
      return sd::DataType ::UINT32;
    else if (hasPropertyBitSet(shapeInfo, ARRAY_LONG))
      return sd::DataType ::UINT64;
    else if (hasPropertyBitSet(shapeInfo, ARRAY_UTF8))
      return sd::DataType ::UTF8;
    else if (hasPropertyBitSet(shapeInfo, ARRAY_UTF16))
      return sd::DataType ::UTF16;
    else if (hasPropertyBitSet(shapeInfo, ARRAY_UTF32))
      return sd::DataType ::UTF32;
    else {

#ifndef __CUDA_ARCH__
      THROW_EXCEPTION("Bad datatype A");
#endif
    }
  } else if (hasPropertyBitSet(shapeInfo, ARRAY_CHAR))
    return sd::DataType::INT8;
  else if (hasPropertyBitSet(shapeInfo, ARRAY_SHORT))
    return sd::DataType::INT16;
  else if (hasPropertyBitSet(shapeInfo, ARRAY_INT))
    return sd::DataType::INT32;
  else if (hasPropertyBitSet(shapeInfo, ARRAY_LONG))
    return sd::DataType::INT64;
  else if (hasPropertyBitSet(shapeInfo, ARRAY_UTF8))
    return sd::DataType::UTF8;
  else if (hasPropertyBitSet(shapeInfo, ARRAY_UTF16))
    return sd::DataType::UTF16;
  else if (hasPropertyBitSet(shapeInfo, ARRAY_UTF32))
    return sd::DataType::UTF32;
  else {
#ifndef __CUDA_ARCH__
    THROW_EXCEPTION("Bad datatype B");
#endif
  }
}



SD_HOST_DEVICE  SpaceType ArrayOptions::spaceTypeForFlags(const sd::LongType& flagStorage) {
  if (hasPropertyBitSetForFlags(flagStorage, ARRAY_QUANTIZED)) return SpaceType::QUANTIZED;
  if (hasPropertyBitSetForFlags(flagStorage, ARRAY_COMPLEX)) return SpaceType::COMPLEX;
  return SpaceType::CONTINUOUS;  // by default we return continuous type here
}

SD_HOST_DEVICE  ArrayType ArrayOptions::arrayTypeForFlags(const sd::LongType& flagStorage) {
  if (hasPropertyBitSetForFlags(flagStorage, ARRAY_SPARSE)) return ArrayType::SPARSE;
  if (hasPropertyBitSetForFlags(flagStorage, ARRAY_COMPRESSED)) return ArrayType::COMPRESSED;
  if (hasPropertyBitSetForFlags(flagStorage, ARRAY_EMPTY)) return ArrayType::EMPTY;
  if (hasPropertyBitSetForFlags(flagStorage, ARRAY_RAGGED)) return ArrayType::RAGGED;
  return ArrayType::DENSE;  // by default we return DENSE type here
}

SD_HOST_DEVICE  bool ArrayOptions::togglePropertyBitForFlags(sd::LongType& flagStorage, int property) {
  flagStorage ^= property;
  return hasPropertyBitSetForFlags(flagStorage, property);
}

SD_HOST_DEVICE  void ArrayOptions::unsetPropertyBitForFlags(sd::LongType& flagStorage, int property) {
  flagStorage &= ~property;
}

SD_HOST_DEVICE SparseType ArrayOptions::sparseTypeForFlags(const sd::LongType& flagStorage) {
  if (hasPropertyBitSetForFlags(flagStorage, ARRAY_CSC)) return SparseType::CSC;
  if (hasPropertyBitSetForFlags(flagStorage, ARRAY_CSR)) return SparseType::CSR;
  if (hasPropertyBitSetForFlags(flagStorage, ARRAY_COO)) return SparseType::COO;
  return SparseType::LIL;
}

// Existing function that works with shapeInfo:
SD_HOST_DEVICE SpaceType ArrayOptions::spaceType(const sd::LongType *shapeInfo) {
  return spaceTypeForFlags(shapeInfo[ArrayOptions::extraIndex(shapeInfo)]);
}

SD_HOST_DEVICE ArrayType ArrayOptions::arrayType(const sd::LongType *shapeInfo) {
  return arrayTypeForFlags(shapeInfo[ArrayOptions::extraIndex(shapeInfo)]);
}

SD_HOST_DEVICE ArrayType ArrayOptions::arrayType(sd::LongType *shapeInfo) {
  return arrayTypeForFlags(shapeInfo[ArrayOptions::extraIndex(shapeInfo)]);
}



SD_HOST_DEVICE bool ArrayOptions::togglePropertyBit(sd::LongType *shapeInfo, int property) {
  return togglePropertyBitForFlags(shapeInfo[ArrayOptions::extraIndex(shapeInfo)], property);
}

SD_HOST_DEVICE void ArrayOptions::setPropertyBit(sd::LongType *shapeInfo, int property) {
  setPropertyBitForFlagsValue(shapeInfo[ArrayOptions::extraIndex(shapeInfo)], property);
}

SD_HOST_DEVICE void ArrayOptions::unsetPropertyBit(sd::LongType *shapeInfo, int property) {
  unsetPropertyBitForFlags(shapeInfo[ArrayOptions::extraIndex(shapeInfo)], property);
}

SD_HOST_DEVICE SparseType ArrayOptions::sparseType(const sd::LongType *shapeInfo) {
  return sparseTypeForFlags(shapeInfo[ArrayOptions::extraIndex(shapeInfo)]);
}
SD_HOST_DEVICE void ArrayOptions::setPropertyBits(sd::LongType *shapeInfo,
                                                  std::initializer_list<int> properties) {
  for (auto v : properties) {
    if (!hasPropertyBitSet(shapeInfo, v)) setPropertyBit(shapeInfo, v);
  }
}

SD_HOST_DEVICE void ArrayOptions::flagAsPaddedBuffer(sd::LongType *shapeInfo) {
  if (!isNewFormat(shapeInfo)) return;

  return setPropertyBit(shapeInfo, ARRAY_HAS_PADDED_BUFFER);
}

SD_HOST_DEVICE bool ArrayOptions::hasPaddedBuffer(const sd::LongType *shapeInfo) {
  if (!isNewFormat(shapeInfo)) return false;

  return hasPropertyBitSet(shapeInfo, ARRAY_HAS_PADDED_BUFFER);
}

SD_HOST_DEVICE sd::LongType ArrayOptions::propertyWithoutDataType(const sd::LongType *shapeInfo) {
  printf("About to access property without data type\n");
  auto newCast = const_cast<sd::LongType *>(shapeInfo);
  printf("Casting to newCast\n");
  sd::LongType property = extra(newCast);
  printf("Property is %lld\n", property);
  property = property & (~ARRAY_BOOL);
  property = property & (~ARRAY_HALF);
  property = property & (~ARRAY_BHALF);
  property = property & (~ARRAY_FLOAT);
  property = property & (~ARRAY_DOUBLE);
  property = property & (~ARRAY_INT);
  property = property & (~ARRAY_LONG);
  property = property & (~ARRAY_CHAR);
  property = property & (~ARRAY_SHORT);
  property = property & (~ARRAY_UNSIGNED);
  return property;
}

SD_HOST_DEVICE void ArrayOptions::resetDataType(sd::LongType *shapeInfo) {
  setExtra(shapeInfo, propertyWithoutDataType(shapeInfo));
}

SD_HOST_DEVICE void ArrayOptions::setDataType(sd::LongType *shapeInfo, const sd::DataType dataType) {
  if (dataType == sd::DataType::UINT8 || dataType == sd::DataType::UINT16 || dataType == sd::DataType::UINT32 ||
      dataType == sd::DataType::UINT64) {
    printf("setPropertyBit ARRAY_UNSIGNED\n");
    setPropertyBit(shapeInfo, ARRAY_UNSIGNED);
  }

  switch (dataType) {
    case sd::DataType::BOOL:
      setPropertyBit(shapeInfo, ARRAY_BOOL);
      break;
    case sd::DataType::HALF:
      setPropertyBit(shapeInfo, ARRAY_HALF);
      break;
    case sd::DataType::BFLOAT16:
      setPropertyBit(shapeInfo, ARRAY_BHALF);
      break;
    case sd::DataType::FLOAT32:
      printf("setPropertyBit ARRAY_FLOAT\n");
      setPropertyBit(shapeInfo, ARRAY_FLOAT);
      break;
    case sd::DataType::DOUBLE:
      printf("setPropertyBit ARRAY_DOUBLE\n");
      setPropertyBit(shapeInfo, ARRAY_DOUBLE);
      break;
    case sd::DataType::INT8:
      setPropertyBit(shapeInfo, ARRAY_CHAR);
      break;
    case sd::DataType::INT16:
      setPropertyBit(shapeInfo, ARRAY_SHORT);
      break;
    case sd::DataType::INT32:
      setPropertyBit(shapeInfo, ARRAY_INT);
      break;
    case sd::DataType::INT64:
      setPropertyBit(shapeInfo, ARRAY_LONG);
      break;
    case sd::DataType::UINT8:
      setPropertyBit(shapeInfo, ARRAY_CHAR);
      break;
    case sd::DataType::UINT16:
      setPropertyBit(shapeInfo, ARRAY_SHORT);
      break;
    case sd::DataType::UINT32:
      setPropertyBit(shapeInfo, ARRAY_INT);
      break;
    case sd::DataType::UINT64:
      setPropertyBit(shapeInfo, ARRAY_LONG);
      break;
    case sd::DataType::UTF8:
      setPropertyBit(shapeInfo, ARRAY_UTF8);
      break;
    case sd::DataType::UTF16:
      setPropertyBit(shapeInfo, ARRAY_UTF16);
      break;
    case sd::DataType::UTF32:
      setPropertyBit(shapeInfo, ARRAY_UTF32);
      break;
    default:
#ifndef __CUDA_ARCH__
      THROW_EXCEPTION("Can't set unknown data type");
#else
      printf("Can't set unknown data type");
#endif
  }
}

////////////////////////////////////////////////////////////////////////////////
void ArrayOptions::copyDataType(sd::LongType *to, const sd::LongType *from) {
  setDataType(to, dataType(from));
}
void ArrayOptions::setPropertyBitForFlagsValue(LongType &extraStorage, int property) {
  extraStorage |= property;
}
}