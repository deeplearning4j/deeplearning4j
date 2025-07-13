#ifndef ND4J_ARRAY_OPTIONS_HXX
#define ND4J_ARRAY_OPTIONS_HXX
#include <array/ArrayOptions.h>
#include <system/op_boilerplate.h>
#include <helpers/shape.h>
#include <array/DataTypeUtils.h>
#pragma  once

namespace sd {



SD_HOST SD_INLINE sd::LongType ArrayOptions::extraIndex(const sd::LongType *shapeInfo) {
  return ArrayOptions::extraIndex(const_cast<sd::LongType *>(shapeInfo));
}


SD_HOST  SD_INLINE sd::LongType ArrayOptions::extraIndex(sd::LongType *shapeInfo) {
  if(shapeInfo == nullptr)
    THROW_EXCEPTION("Shape info was null!");
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


SD_HOST SD_INLINE void ArrayOptions::setExtra(sd::LongType *shapeInfo, sd::LongType value) {
  sd::LongType  idx = ArrayOptions::extraIndex(shapeInfo);
  shapeInfo[idx] = value;
}

SD_HOST SD_INLINE LongType ArrayOptions::extra(const LongType *shapeInfo) {
  sd::LongType  idx = ArrayOptions::extraIndex(shapeInfo);
  if(idx >  shape::shapeInfoLength(shape::rank(shapeInfo)))
    THROW_EXCEPTION("Extra index is out of bounds!");
  return shapeInfo[idx];
}

SD_HOST SD_INLINE bool ArrayOptions::isNewFormat(const sd::LongType *shapeInfo) {
  return (extra(const_cast<sd::LongType *>(shapeInfo)) != 0);
}

SD_HOST SD_INLINE bool ArrayOptions::isSparseArray(sd::LongType *shapeInfo) {
  return hasPropertyBitSet(shapeInfo, ARRAY_SPARSE);
}

SD_HOST_DEVICE SD_INLINE bool ArrayOptions::hasExtraProperties(sd::LongType *shapeInfo) {
  return hasPropertyBitSet(shapeInfo, ARRAY_EXTRAS);
}

SD_HOST  SD_INLINE bool ArrayOptions::hasPropertyBitSet(const sd::LongType extra, LongType property) {
  return ((static_cast<int>(extra) & static_cast<int>(property)) == static_cast<int>(property));
}

SD_HOST SD_INLINE bool ArrayOptions:: hasPropertyBitSet(const sd::LongType *shapeInfo, LongType property) {
  if (!isNewFormat(shapeInfo)) return false;

  return ((static_cast<int>(extra(const_cast<sd::LongType *>(shapeInfo)) & property)) == static_cast<int>(property));
}


SD_HOST_DEVICE SD_INLINE bool hasPropertyBitSetForFlags(const sd::LongType& flagStorage, LongType property) {
  return static_cast<sd::LongType>(flagStorage & (property)) == (property);
}

SD_HOST  SD_INLINE void unsetPropertyBitForFlags(sd::LongType& flagStorage, LongType property) {
  flagStorage &= ~property;
}

SD_HOST SD_INLINE const char *ArrayOptions::enumerateSetFlagsForFlags(const LongType flagStorage) {
  int maxFlags = 24;
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
      ARRAY_HAS_PADDED_BUFFER,
      ARRAY_IS_VIEW
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
      "ARRAY_HAS_PADDED_BUFFER",
      "ARRAY_IS_VIEW"
  };

  std::string flags;
  for (int i = 0; i <  maxFlags; i++) {
    if (hasPropertyBitSetForFlags(flagStorage, flagsArray[i])) {
      flags += flagsStrings[i];
      flags += " ";
    }

  }

  auto ret =  new std::string(flags);  // Returns the number of set flags found
  return ret->c_str();
}

SD_HOST SD_INLINE void ArrayOptions::unsetAllFlags(sd::LongType& flagStorage) {

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

  for (int i = 0; i < 24; i++) {
    unsetPropertyBitForFlags(flagStorage, flagsArray[i]);
  }
}

SD_HOST SD_INLINE const char *ArrayOptions::enumerateSetFlags(const sd::LongType *shapeInfo) {
  return enumerateSetFlagsForFlags(shapeInfo[ArrayOptions::extraIndex(shapeInfo)]);
}

SD_HOST  SD_INLINE void ArrayOptions::unsetAllFlags(sd::LongType *shapeInfo) {
  ArrayOptions::unsetAllFlags(shapeInfo[ArrayOptions::extraIndex(shapeInfo)]);
}


SD_HOST  SD_INLINE bool ArrayOptions::isUnsigned(sd::LongType *shapeInfo) {
  if (!isNewFormat(shapeInfo)) return false;

  return hasPropertyBitSet(shapeInfo, ARRAY_UNSIGNED);
}



#define DATA_TYPE_FLAGS { \
    ARRAY_FLOAT, \
    ARRAY_DOUBLE, \
    ARRAY_HALF, \
    ARRAY_BHALF, \
    ARRAY_BOOL, \
    ARRAY_CHAR, \
    ARRAY_SHORT, \
    ARRAY_INT, \
    ARRAY_LONG, \
    ARRAY_UTF8, \
    ARRAY_UTF16, \
    ARRAY_UTF32 \
}

#define DATA_TYPES { \
    sd::DataType::FLOAT32, \
    sd::DataType::DOUBLE, \
    sd::DataType::HALF, \
    sd::DataType::BFLOAT16, \
    sd::DataType::BOOL, \
    sd::DataType::INT8, \
    sd::DataType::INT16, \
    sd::DataType::INT32, \
    sd::DataType::INT64, \
    sd::DataType::UTF8, \
    sd::DataType::UTF16, \
    sd::DataType::UTF32 \
}

#define ARRAY_UNSIGNED_TYPES { \
    ARRAY_CHAR, \
    ARRAY_SHORT, \
    ARRAY_INT, \
    ARRAY_LONG, \
    ARRAY_UTF8, \
    ARRAY_UTF16, \
    ARRAY_UTF32 \
}

#define UNSIGNED_DATA_TYPES { \
    sd::DataType::UINT8, \
    sd::DataType::UINT16, \
    sd::DataType::UINT32, \
    sd::DataType::UINT64, \
    sd::DataType::UTF8, \
    sd::DataType::UTF16, \
    sd::DataType::UTF32 \
}


SD_HOST SD_INLINE sd::DataType ArrayOptions::dataTypeValue(sd::LongType property) {
  const sd::LongType dataTypeFlags[] = DATA_TYPE_FLAGS;
  const sd::DataType dataTypes[] = DATA_TYPES;
  const size_t numTypes = sizeof(dataTypeFlags) / sizeof(sd::LongType);


  if (hasPropertyBitSetForFlags(property, ARRAY_UNSIGNED)) {
    const sd::LongType unsignedTypeFlags[] = ARRAY_UNSIGNED_TYPES;
    const sd::DataType unsignedDataTypes[] = UNSIGNED_DATA_TYPES;
    const size_t numUnsignedTypes = sizeof(unsignedTypeFlags) / sizeof(sd::LongType);

    for (size_t i = 0; i < numUnsignedTypes; ++i) {
      if (hasPropertyBitSetForFlags(property, unsignedTypeFlags[i])) {
        return unsignedDataTypes[i];
      }
    }
  } else {
    for (size_t i = 0; i < numTypes; ++i) {
      if (hasPropertyBitSetForFlags(property, dataTypeFlags[i])) {
        return dataTypes[i];
      }
    }
  }

  return sd::DataType::UNKNOWN;
}



SD_HOST  SD_INLINE void validateFlags(sd::LongType property, const sd::LongType flags[], size_t numFlags) {
  LongType flagIndices[numFlags];
  int numFlagsSet = 0;
  for (size_t i = 0; i < numFlags; ++i) {
    if (hasPropertyBitSetForFlags(property, flags[i])) {
      flagIndices[i] = 1;
      numFlagsSet++;
    } else {
      flagIndices[i] = 0;
    }
  }

  if (numFlagsSet > 1) {
    std::ostringstream errorMsg;
    errorMsg << "Multiple data types are set for the given property: ";
    for (size_t i = 0; i < numFlags; i++) {
      if(flagIndices[i] == 1) {
        errorMsg << "Flag index " << i << " (flag value: " << flags[i] << "), ";
      }
    }
    errorMsg << "Total: " << numFlagsSet << " data types set.";
    THROW_EXCEPTION(errorMsg.str().c_str());
  }
}

SD_HOST SD_INLINE void ArrayOptions::validateSingleDataType(sd::LongType property) {
  const sd::LongType dataTypeFlags[] = DATA_TYPE_FLAGS;
  const size_t numDataTypeFlags = sizeof(dataTypeFlags) / sizeof(sd::LongType);
  validateFlags(property, dataTypeFlags, numDataTypeFlags);

  if (hasPropertyBitSetForFlags(property, ARRAY_UNSIGNED)) {
    const sd::LongType unsignedTypeFlags[] = ARRAY_UNSIGNED_TYPES;
    const size_t numUnsignedTypeFlags = sizeof(unsignedTypeFlags) / sizeof(sd::LongType);
    validateFlags(property, unsignedTypeFlags, numUnsignedTypeFlags);
  }
}


SD_HOST SD_INLINE sd::LongType ArrayOptions::numDataTypesSet(sd::LongType property) {
  const sd::LongType dataTypeFlags[] = DATA_TYPE_FLAGS;
  const size_t numDataTypeFlags = sizeof(dataTypeFlags) / sizeof(sd::LongType);

  sd::LongType numFlagsSet = 0;
  for (size_t i = 0; i < numDataTypeFlags; ++i) {
    if (hasPropertyBitSetForFlags(property, dataTypeFlags[i])) {
      numFlagsSet++;
    }
  }

  return numFlagsSet;
}

SD_HOST SD_INLINE sd::DataType ArrayOptions::dataType(const sd::LongType *shapeInfo) {
  if(shapeInfo == nullptr)
    THROW_EXCEPTION("ArrayOptions::dataType(..) shapeInfo can not be null!");

  auto extra = ArrayOptions::extra(shapeInfo);
  return ArrayOptions::dataTypeValue(extra);
}



SD_HOST SD_INLINE SpaceType ArrayOptions::spaceTypeForFlags(const sd::LongType& flagStorage) {
  if (hasPropertyBitSetForFlags(flagStorage, ARRAY_QUANTIZED)) return SpaceType::QUANTIZED;
  if (hasPropertyBitSetForFlags(flagStorage, ARRAY_COMPLEX)) return SpaceType::COMPLEX;
  return SpaceType::CONTINUOUS;  // by default we return continuous type here
}

SD_HOST SD_INLINE ArrayType ArrayOptions::arrayTypeForFlags(const sd::LongType& flagStorage) {
  if (hasPropertyBitSetForFlags(flagStorage, ARRAY_SPARSE)) return ArrayType::SPARSE;
  if (hasPropertyBitSetForFlags(flagStorage, ARRAY_COMPRESSED)) return ArrayType::COMPRESSED;
  if (hasPropertyBitSetForFlags(flagStorage, ARRAY_EMPTY)) return ArrayType::EMPTY;
  if (hasPropertyBitSetForFlags(flagStorage, ARRAY_RAGGED)) return ArrayType::RAGGED;
  return ArrayType::DENSE;  // by default we return DENSE type here
}

SD_HOST SD_INLINE  bool ArrayOptions::togglePropertyBitForFlags(sd::LongType& flagStorage, LongType property) {
  flagStorage ^= property;
  return hasPropertyBitSetForFlags(flagStorage, property);
}

SD_HOST SD_INLINE sd::LongType ArrayOptions::unsetPropertyBitForFlags(sd::LongType& flagStorage, LongType property) {
  return flagStorage & ~property;
}

SD_HOST  SD_INLINE SparseType ArrayOptions::sparseTypeForFlags(const sd::LongType& flagStorage) {
  if (hasPropertyBitSetForFlags(flagStorage, ARRAY_CSC)) return SparseType::CSC;
  if (hasPropertyBitSetForFlags(flagStorage, ARRAY_CSR)) return SparseType::CSR;
  if (hasPropertyBitSetForFlags(flagStorage, ARRAY_COO)) return SparseType::COO;
  return SparseType::LIL;
}

// Existing function that works with shapeInfo:
SD_HOST_DEVICE  SD_INLINE SpaceType ArrayOptions::spaceType(const sd::LongType *shapeInfo) {
  return spaceTypeForFlags(shapeInfo[ArrayOptions::extraIndex(shapeInfo)]);
}

SD_HOST  SD_INLINE ArrayType ArrayOptions::arrayType(const sd::LongType *shapeInfo) {
  return arrayTypeForFlags(shapeInfo[ArrayOptions::extraIndex(shapeInfo)]);
}

SD_HOST SD_INLINE ArrayType ArrayOptions::arrayType(sd::LongType *shapeInfo) {
  return arrayTypeForFlags(shapeInfo[ArrayOptions::extraIndex(shapeInfo)]);
}

SD_HOST SD_INLINE bool ArrayOptions::isEmpty(sd::LongType *shapeInfo) {
  return hasPropertyBitSet(shapeInfo, EMPTY);
}


 SD_HOST SD_INLINE bool ArrayOptions::arrayNeedsCopy(LongType *shapeInfo) {
  return hasPropertyBitSetForFlags(shapeInfo[ArrayOptions::extraIndex(shapeInfo)], ARRAY_NEEDS_COPY);
}

 SD_HOST SD_INLINE void ArrayOptions::toggleArrayNeedsCopy(LongType *shapeInfo) {
    togglePropertyBit(shapeInfo, ARRAY_NEEDS_COPY);
}

SD_HOST SD_INLINE void ArrayOptions::toggleIsEmpty(sd::LongType *shapeInfo) {
  togglePropertyBit(shapeInfo, EMPTY);
}

SD_HOST SD_INLINE bool ArrayOptions::isView(sd::LongType *shapeInfo) {
  return hasPropertyBitSet(shapeInfo, ARRAY_IS_VIEW);
}

SD_HOST SD_INLINE void ArrayOptions::toggleIsView(sd::LongType *shapeInfo) {
  togglePropertyBit(shapeInfo, ARRAY_IS_VIEW);
}

SD_HOST SD_INLINE bool ArrayOptions::togglePropertyBit(sd::LongType *shapeInfo, LongType property) {
  return togglePropertyBitForFlags(shapeInfo[ArrayOptions::extraIndex(shapeInfo)], property);
}

SD_HOST SD_INLINE void ArrayOptions::setPropertyBit(sd::LongType *shapeInfo, LongType property) {
  shapeInfo[ArrayOptions::extraIndex(shapeInfo)] = setPropertyBitForFlagsValue(shapeInfo[ArrayOptions::extraIndex(shapeInfo)], property);
}

SD_HOST SD_INLINE void ArrayOptions::unsetPropertyBit(sd::LongType *shapeInfo, LongType property) {
  shapeInfo[ArrayOptions::extraIndex(shapeInfo)] = unsetPropertyBitForFlags(shapeInfo[ArrayOptions::extraIndex(shapeInfo)], property);
}

SD_HOST SD_INLINE SparseType ArrayOptions::sparseType(const sd::LongType *shapeInfo) {
  return sparseTypeForFlags(shapeInfo[ArrayOptions::extraIndex(shapeInfo)]);
}
SD_HOST SD_INLINE void ArrayOptions::setPropertyBits(sd::LongType *shapeInfo, std::initializer_list<LongType> properties) {
  for (auto v : properties) {
    if (!hasPropertyBitSet(shapeInfo, v)) setPropertyBit(shapeInfo, v);
  }
}

SD_HOST SD_INLINE void ArrayOptions::flagAsPaddedBuffer(sd::LongType *shapeInfo) {
  if (!isNewFormat(shapeInfo)) return;

  return setPropertyBit(shapeInfo, ARRAY_HAS_PADDED_BUFFER);
}

SD_HOST SD_INLINE bool ArrayOptions::hasPaddedBuffer(const sd::LongType *shapeInfo) {
  if (!isNewFormat(shapeInfo)) return false;

  return hasPropertyBitSet(shapeInfo, ARRAY_HAS_PADDED_BUFFER);
}

SD_HOST SD_INLINE sd::LongType ArrayOptions::propertyWithoutDataTypeValue(sd::LongType extra) {
  sd::LongType property = extra;
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


SD_HOST SD_INLINE sd::LongType ArrayOptions::propertyWithoutDataType(const sd::LongType *shapeInfo) {
  auto newCast = const_cast<sd::LongType *>(shapeInfo);
  sd::LongType property = extra(newCast);
  return propertyWithoutDataTypeValue(property);
}

SD_HOST SD_INLINE void ArrayOptions::resetDataType(sd::LongType *shapeInfo) {
  setExtra(shapeInfo, propertyWithoutDataType(shapeInfo));
}

SD_HOST SD_INLINE LongType ArrayOptions::flagForDataType(const sd::DataType dataType) {
  switch (dataType) {
    case sd::DataType::BOOL:
      return ARRAY_BOOL;
    case sd::DataType::HALF:
      return ARRAY_HALF;
    case sd::DataType::BFLOAT16:
      return ARRAY_BHALF;
    case sd::DataType::FLOAT32:
      return ARRAY_FLOAT;
    case sd::DataType::DOUBLE:
      return ARRAY_DOUBLE;
    case sd::DataType::INT8:
      return ARRAY_CHAR;
    case sd::DataType::INT16:
      return ARRAY_SHORT;
    case sd::DataType::INT32:
      return ARRAY_INT;
    case sd::DataType::INT64:
      return ARRAY_LONG;
    case sd::DataType::UINT8:
      return ARRAY_CHAR | ARRAY_UNSIGNED;
    case sd::DataType::UINT16:
      return ARRAY_SHORT | ARRAY_UNSIGNED;
    case sd::DataType::UINT32  :
      return ARRAY_INT | ARRAY_UNSIGNED;
    case sd::DataType::UINT64 :
      return ARRAY_LONG | ARRAY_UNSIGNED;
    case sd::DataType::UTF8:
      return ARRAY_UTF8;
    case sd::DataType::UTF16:
      return ARRAY_UTF16;
    case sd::DataType::UTF32:
      return ARRAY_UTF32;
    default:
#ifndef __CUDA_ARCH__
      std::string errorMessage;
      errorMessage += "Can't set unknown data type ArrayOptions: ";
      errorMessage += sd::DataTypeUtils::asString(dataType);
      THROW_EXCEPTION(errorMessage.c_str());

#else
      printf("Can't set unknown data type");
#endif
  }
}

SD_HOST SD_INLINE void ArrayOptions::setDataType(sd::LongType *shapeInfo, const sd::DataType dataType) {
  switch (dataType) {
    case sd::DataType::BOOL:
      ArrayOptions::resetDataType(shapeInfo);
      setPropertyBit(shapeInfo, ARRAY_BOOL);
      break;
    case sd::DataType::HALF:
      ArrayOptions::resetDataType(shapeInfo);
      setPropertyBit(shapeInfo, ARRAY_HALF);
      break;
    case sd::DataType::BFLOAT16:
      ArrayOptions::resetDataType(shapeInfo);
      setPropertyBit(shapeInfo, ARRAY_BHALF);
      break;
    case sd::DataType::FLOAT32:
      ArrayOptions::resetDataType(shapeInfo);
      setPropertyBit(shapeInfo, ARRAY_FLOAT);
      break;
    case sd::DataType::DOUBLE:
      ArrayOptions::resetDataType(shapeInfo);
      setPropertyBit(shapeInfo, ARRAY_DOUBLE);
      break;
    case sd::DataType::INT8:
      ArrayOptions::resetDataType(shapeInfo);
      setPropertyBit(shapeInfo, ARRAY_CHAR);
      break;
    case sd::DataType::INT16:
      ArrayOptions::resetDataType(shapeInfo);
      setPropertyBit(shapeInfo, ARRAY_SHORT);
      break;
    case sd::DataType::INT32:
      ArrayOptions::resetDataType(shapeInfo);
      setPropertyBit(shapeInfo, ARRAY_INT);
      break;
    case sd::DataType::INT64:
      ArrayOptions::resetDataType(shapeInfo);
      setPropertyBit(shapeInfo, ARRAY_LONG);
      break;
    case sd::DataType::UINT8:
      ArrayOptions::resetDataType(shapeInfo);
      setPropertyBit(shapeInfo, ARRAY_CHAR | ARRAY_UNSIGNED);
      break;
    case sd::DataType::UINT16:
      ArrayOptions::resetDataType(shapeInfo);
      setPropertyBit(shapeInfo, ARRAY_SHORT | ARRAY_UNSIGNED);
      break;
    case sd::DataType::UINT32:
      ArrayOptions::resetDataType(shapeInfo);
      setPropertyBit(shapeInfo, ARRAY_INT | ARRAY_UNSIGNED);
      break;
    case sd::DataType::UINT64:
      ArrayOptions::resetDataType(shapeInfo);
      setPropertyBit(shapeInfo, ARRAY_LONG | ARRAY_UNSIGNED);
      break;
    case sd::DataType::UTF8:
      ArrayOptions::resetDataType(shapeInfo);
      setPropertyBit(shapeInfo, ARRAY_UTF8);
      break;
    case sd::DataType::UTF16:
      ArrayOptions::resetDataType(shapeInfo);
      setPropertyBit(shapeInfo, ARRAY_UTF16);
      break;
    case sd::DataType::UTF32:
      ArrayOptions::resetDataType(shapeInfo);
      setPropertyBit(shapeInfo, ARRAY_UTF32);
      break;
    default:
#ifndef __CUDA_ARCH__
      std::string errorMessage;
      errorMessage += "Can't set unknown data type: ";
      errorMessage += DataTypeUtils::asString(dataType);
      THROW_EXCEPTION(errorMessage.c_str());
#else
      printf("Can't set unknown data type");
#endif

      validateSingleDataType(shapeInfo[ArrayOptions::extraIndex(shapeInfo)]);

  }

#ifndef __CUDA_ARCH__
  if(ArrayOptions::dataType(shapeInfo) != dataType) {
    std::string errorMessage;
    errorMessage += "setDataType:  Data type set was not correct one. Expected ";
    errorMessage += DataTypeUtils::asString(dataType);
    errorMessage += " but got ";
    errorMessage += sd::DataTypeUtils::asString(dataType);
    THROW_EXCEPTION(errorMessage.c_str());
  }

#else
  printf("setDataType: Data type set was incorrect.");
#endif

}

SD_HOST SD_INLINE sd::LongType ArrayOptions::setDataTypeValue(sd::LongType extraStorage, const sd::DataType dataType) {
  sd::LongType  ret = extraStorage;
  if (dataType == sd::DataType::UINT8
      || dataType == sd::DataType::UINT16
      || dataType == sd::DataType::UINT32 ||
      dataType == sd::DataType::UINT64) {
    ret = setPropertyBitForFlagsValue(extraStorage, ARRAY_UNSIGNED);
    extraStorage = ret;
    return extraStorage;
  }


  switch (dataType) {
    case sd::DataType::BOOL:
      ret = setPropertyBitForFlagsValue(extraStorage, ARRAY_BOOL);
      break;
    case sd::DataType::HALF:
      ret = setPropertyBitForFlagsValue(extraStorage, ARRAY_HALF);
      break;
    case sd::DataType::BFLOAT16:
      ret = setPropertyBitForFlagsValue(extraStorage, ARRAY_BHALF);
      break;
    case sd::DataType::FLOAT32:
      ret = setPropertyBitForFlagsValue(extraStorage, ARRAY_FLOAT);
      break;
    case sd::DataType::DOUBLE:
      ret = setPropertyBitForFlagsValue(extraStorage, ARRAY_DOUBLE);
      break;
    case sd::DataType::INT8:
      ret = setPropertyBitForFlagsValue(extraStorage, ARRAY_CHAR);
      break;
    case sd::DataType::INT16:
      ret =  setPropertyBitForFlagsValue(extraStorage, ARRAY_SHORT);
      break;
    case sd::DataType::INT32:
      ret = setPropertyBitForFlagsValue(extraStorage, ARRAY_INT);
      break;
    case sd::DataType::INT64:
      ret =  setPropertyBitForFlagsValue(extraStorage, ARRAY_LONG);
      break;
    case sd::DataType::UINT8:
      ret =  setPropertyBitForFlagsValue(extraStorage, ARRAY_CHAR);
      break;
    case sd::DataType::UINT16:
      ret =  setPropertyBitForFlagsValue(extraStorage, ARRAY_SHORT);
      break;
    case sd::DataType::UINT32:
      ret = setPropertyBitForFlagsValue(extraStorage, ARRAY_INT);
      break;
    case sd::DataType::UINT64:
      ret = setPropertyBitForFlagsValue(extraStorage, ARRAY_LONG);
      break;
    case sd::DataType::UTF8:
      ret =  setPropertyBitForFlagsValue(extraStorage, ARRAY_UTF8);
      break;
    case sd::DataType::UTF16:
      ret =  setPropertyBitForFlagsValue(extraStorage, ARRAY_UTF16);
      break;
    case sd::DataType::UTF32:
      ret = setPropertyBitForFlagsValue(extraStorage, ARRAY_UTF32);
      break;
    default:
#ifndef __CUDA_ARCH__
      THROW_EXCEPTION("Can't set unknown data type");
#else
      printf("Can't set unknown data type");
#endif
  }



  return ret;

}

SD_HOST sd::LongType ArrayOptions::defaultFlag() {
  return DEFAULT_FLAG;
}

SD_HOST void ArrayOptions::resetFlags(sd::LongType *to) {
  to[ArrayOptions::extraIndex(to)] = 1;
}

////////////////////////////////////////////////////////////////////////////////
SD_HOST SD_INLINE  void ArrayOptions::copyDataType(sd::LongType *to, const sd::LongType *from) {
  if(to == nullptr)
    THROW_EXCEPTION("To Shape info was null!");
  if(from == nullptr)
    THROW_EXCEPTION("From Shape info was null!");
  DataType dt = dataType(from);
  sd::LongType flagForDt = flagForDataType(dt);
  if(hasPropertyBitSet(to, flagForDt))
    return;
  setDataType(to, dt);
}
SD_HOST sd::LongType ArrayOptions::setPropertyBitForFlagsValue(LongType extraStorage, LongType property) {
  if(hasPropertyBitSet(extraStorage, property))
    return extraStorage;
  return extraStorage | property;
}
}
#endif