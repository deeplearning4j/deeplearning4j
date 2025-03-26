/**
* Streamlined implementation of NDArray operators
 */
#include <array/NDArray.h>
#include <array/NDArrayFactory.h>
#include <legacy/NativeOpExecutioner.h>
#include <system/Environment.h>

namespace sd {

/////////////////////////////////////////////////////
// Scalar-Array Operators - Helper Functions
/////////////////////////////////////////////////////
SD_LIB_EXPORT NDArray operator+(NDArray& arr1, NDArray& arr2);
SD_LIB_EXPORT NDArray operator+(NDArray&& arr1, NDArray& arr2);
SD_LIB_EXPORT NDArray operator+(NDArray& arr1, NDArray&& arr2);
SD_LIB_EXPORT NDArray operator+(NDArray&& arr1, NDArray&& arr2);

// Subtraction: NDArray - NDArray
SD_LIB_EXPORT NDArray operator-(NDArray& arr1, NDArray& arr2);
SD_LIB_EXPORT NDArray operator-(NDArray&& arr1, NDArray& arr2);
SD_LIB_EXPORT NDArray operator-(NDArray& arr1, NDArray&& arr2);
SD_LIB_EXPORT NDArray operator-(NDArray&& arr1, NDArray&& arr2);

// Multiplication: NDArray * NDArray
SD_LIB_EXPORT NDArray operator*(NDArray& arr1, NDArray& arr2);
SD_LIB_EXPORT NDArray operator*(NDArray&& arr1, NDArray& arr2);
SD_LIB_EXPORT NDArray operator*(NDArray& arr1, NDArray&& arr2);
SD_LIB_EXPORT NDArray operator*(NDArray&& arr1, NDArray&& arr2);

// Division: NDArray / NDArray
SD_LIB_EXPORT NDArray operator/(NDArray& arr1, NDArray& arr2);
SD_LIB_EXPORT NDArray operator/(NDArray&& arr1, NDArray& arr2);
SD_LIB_EXPORT NDArray operator/(NDArray& arr1, NDArray&& arr2);
SD_LIB_EXPORT NDArray operator/(NDArray&& arr1, NDArray&& arr2);


template <typename T>
NDArray scalarArrayOpHelper(NDArray* arr, T scalar, sd::scalar::Ops op, bool inPlace) {
  if (arr->isS()) {
    std::string errorMessage = "scalarArrayOpHelper: you can't use this method on String array!";
    THROW_EXCEPTION(errorMessage.c_str());
  }

  auto tmp = NDArrayFactory::create(arr->dataType(), scalar, arr->getContext());

  if (inPlace && !arr->isView()) {
    NDArray::prepareSpecialUse({arr}, {arr, &tmp});
    NativeOpExecutioner::execScalar(arr->getContext(), op,
                                    arr->buffer(), arr->shapeInfo(), arr->specialBuffer(), arr->specialShapeInfo(),
                                    arr->buffer(), arr->shapeInfo(), arr->specialBuffer(), arr->specialShapeInfo(),
                                    tmp.buffer(), tmp.shapeInfo(), tmp.specialBuffer(), tmp.specialShapeInfo(), nullptr);
    NDArray::registerSpecialUse({arr}, {arr, &tmp});
    return std::move(*arr);
  } else {
    NDArray result(arr->shapeInfo(), DataTypeUtils::pickPairwiseResultType(arr->dataType(), DataTypeUtils::fromT<T>()),
                   false, arr->getContext());
    NDArray::prepareSpecialUse({&result}, {arr, &tmp});
    NativeOpExecutioner::execScalar(arr->getContext(), op,
                                    arr->buffer(), arr->shapeInfo(), arr->specialBuffer(), arr->specialShapeInfo(),
                                    result.buffer(), result.shapeInfo(), result.specialBuffer(), result.specialShapeInfo(),
                                    tmp.buffer(), tmp.shapeInfo(), tmp.specialBuffer(), tmp.specialShapeInfo(), nullptr);
    NDArray::registerSpecialUse({&result}, {arr, &tmp});
    return result;
  }
}

/////////////////////////////////////////////////////
// Array-Array Operators - Helper Function
/////////////////////////////////////////////////////

NDArray arrayArrayOpHelper(NDArray* arr1, NDArray* arr2, sd::pairwise::Ops op, bool inPlace) {
  if (arr1->isS() || arr2->isS()) {
    std::string errorMessage = "arrayArrayOpHelper: you can't use this method on String arrays!";
    THROW_EXCEPTION(errorMessage.c_str());
  }

  if (!Environment::getInstance().isExperimentalBuild() && arr1->dataType() != arr2->dataType() &&
      (arr1->dataType() != DataType::BOOL || arr2->dataType() != BOOL)) {
    std::string errorMessage;
    errorMessage += "arrayArrayOpHelper: Cannot operate on arrays of different types: ";
    errorMessage += DataTypeUtils::asString(arr1->dataType());
    errorMessage += " and ";
    errorMessage += DataTypeUtils::asString(arr2->dataType());
    THROW_EXCEPTION(errorMessage.c_str());
  }
  if (arr1->lengthOf() == arr2->lengthOf() && arr1->rankOf() == arr2->rankOf()) {
    NDArray* result = nullptr;
    const bool canUseArr1 = inPlace && !arr1->isView();
    const bool canUseArr2 = inPlace && !arr2->isView();

    if (canUseArr1)
      result = arr1;
    else if (canUseArr2)
      result = arr2;
    else
      result = new NDArray(arr1->shapeInfo(),
                           DataTypeUtils::pickPairwiseResultType(arr1->shapeInfo(), arr2->shapeInfo()),
                           false, arr1->getContext());

    NDArray::prepareSpecialUse({result}, {arr1, arr2});
    NativeOpExecutioner::execPairwiseTransform(
        arr1->getContext(), op,
        arr1->buffer(), arr1->shapeInfo(), arr1->specialBuffer(), arr1->specialShapeInfo(),
        arr2->buffer(), arr2->shapeInfo(), arr2->specialBuffer(), arr2->specialShapeInfo(),
        result->buffer(), result->shapeInfo(), result->specialBuffer(), result->specialShapeInfo(), nullptr);
    NDArray::registerSpecialUse({result}, {arr1, arr2});

    if (!canUseArr1 && !canUseArr2 && result != arr1 && result != arr2) {
      NDArray res = std::move(*result);
      delete result;
      return res;
    }

    return std::move(*result);
  }

  // Broadcast case
  sd::BroadcastOpsTuple broadcastOp;
  switch (op) {
    case sd::pairwise::Add: broadcastOp = sd::BroadcastOpsTuple::Add(); break;
    case sd::pairwise::Subtract: broadcastOp = sd::BroadcastOpsTuple::Subtract(); break;
    case sd::pairwise::Multiply: broadcastOp = sd::BroadcastOpsTuple::Multiply(); break;
    case sd::pairwise::Divide: broadcastOp = sd::BroadcastOpsTuple::Divide(); break;
    default:
      std::string errorMessage = "arrayArrayOpHelper: Unsupported pairwise operation for broadcasting";
      THROW_EXCEPTION(errorMessage.c_str());
  }

  return *arr1->applyTrueBroadcast(broadcastOp, arr2);
}

/////////////////////////////////////////////////////
// Addition Operators (NDArray + scalar)
/////////////////////////////////////////////////////

template <typename T, typename>
NDArray operator+(NDArray& arr, T scalar) {
  return scalarArrayOpHelper(&arr, scalar, sd::scalar::Add, false);
}

template <typename T, typename>
NDArray operator+(NDArray&& arr, T scalar) {
  return scalarArrayOpHelper(&arr, scalar, sd::scalar::Add, true);
}

template <typename T, typename>
NDArray operator+(T scalar, NDArray& arr) {
  return scalarArrayOpHelper(&arr, scalar, sd::scalar::Add, false);
}

template <typename T, typename>
NDArray operator+(T scalar, NDArray&& arr) {
  return scalarArrayOpHelper(&arr, scalar, sd::scalar::Add, true);
}

#define INSTANTIATE_SCALAR_OP(OP, T) \
 template SD_LIB_EXPORT NDArray operator OP<GET_SECOND(T), typename std::enable_if<DataTypeUtils::scalarTypesForNDarray<GET_SECOND(T)>::value>::type>(NDArray& arr, GET_SECOND(T) scalar); \
 template SD_LIB_EXPORT NDArray operator OP<GET_SECOND(T), typename std::enable_if<DataTypeUtils::scalarTypesForNDarray<GET_SECOND(T)>::value>::type>(NDArray&& arr, GET_SECOND(T) scalar); \
 template SD_LIB_EXPORT NDArray operator OP<GET_SECOND(T), typename std::enable_if<DataTypeUtils::scalarTypesForNDarray<GET_SECOND(T)>::value>::type>(GET_SECOND(T) scalar, NDArray& arr); \
 template SD_LIB_EXPORT NDArray operator OP<GET_SECOND(T), typename std::enable_if<DataTypeUtils::scalarTypesForNDarray<GET_SECOND(T)>::value>::type>(GET_SECOND(T) scalar, NDArray&& arr);

#define INSTANTIATE_ALL_SCALAR_OPS(T) \
 INSTANTIATE_SCALAR_OP(+, T) \
 INSTANTIATE_SCALAR_OP(-, T) \
 INSTANTIATE_SCALAR_OP(*, T) \
 INSTANTIATE_SCALAR_OP(/, T)

ITERATE_LIST((SD_NUMERIC_TYPES), INSTANTIATE_ALL_SCALAR_OPS)

/////////////////////////////////////////////////////
// Subtraction Operators (NDArray - scalar)
/////////////////////////////////////////////////////

template <typename T, typename>
NDArray operator-(NDArray& arr, T scalar) {
  return scalarArrayOpHelper(&arr, scalar, sd::scalar::Subtract, false);
}

template <typename T, typename>
NDArray operator-(NDArray&& arr, T scalar) {
  return scalarArrayOpHelper(&arr, scalar, sd::scalar::Subtract, true);
}

template <typename T, typename>
NDArray operator-(T scalar, NDArray& arr) {
  return scalarArrayOpHelper(&arr, scalar, sd::scalar::ReverseSubtract, false);
}

template <typename T, typename>
NDArray operator-(T scalar, NDArray&& arr) {
  return scalarArrayOpHelper(&arr, scalar, sd::scalar::ReverseSubtract, true);
}

/////////////////////////////////////////////////////
// Multiplication Operators (NDArray * scalar)
/////////////////////////////////////////////////////

template <typename T, typename>
NDArray operator*(NDArray& arr, T scalar) {
  return scalarArrayOpHelper(&arr, scalar, sd::scalar::Multiply, false);
}

template <typename T, typename>
NDArray operator*(NDArray&& arr, T scalar) {
  return scalarArrayOpHelper(&arr, scalar, sd::scalar::Multiply, true);
}

template <typename T, typename>
NDArray operator*(T scalar, NDArray& arr) {
  return scalarArrayOpHelper(&arr, scalar, sd::scalar::Multiply, false);
}

template <typename T, typename>
NDArray operator*(T scalar, NDArray&& arr) {
  return scalarArrayOpHelper(&arr, scalar, sd::scalar::Multiply, true);
}

/////////////////////////////////////////////////////
// Division Operators (NDArray / scalar)
/////////////////////////////////////////////////////

template <typename T, typename>
NDArray operator/(NDArray& arr, T scalar) {
  return scalarArrayOpHelper(&arr, scalar, sd::scalar::Divide, false);
}

template <typename T, typename>
NDArray operator/(NDArray&& arr, T scalar) {
  return scalarArrayOpHelper(&arr, scalar, sd::scalar::Divide, true);
}

template <typename T, typename>
NDArray operator/(T scalar, NDArray& arr) {
  return scalarArrayOpHelper(&arr, scalar, sd::scalar::ReverseDivide, false);
}

template <typename T, typename>
NDArray operator/(T scalar, NDArray&& arr) {
  return scalarArrayOpHelper(&arr, scalar, sd::scalar::ReverseDivide, true);
}

/////////////////////////////////////////////////////
// Array-Array Operators
/////////////////////////////////////////////////////

// Addition: NDArray + NDArray
NDArray operator+(NDArray& arr1, NDArray& arr2) {
  return arrayArrayOpHelper(&arr1, &arr2, sd::pairwise::Add, false);
}

NDArray operator+(NDArray&& arr1, NDArray& arr2) {
  return arrayArrayOpHelper(&arr1, &arr2, sd::pairwise::Add, true);
}

NDArray operator+(NDArray& arr1, NDArray&& arr2) {
  return arrayArrayOpHelper(&arr1, &arr2, sd::pairwise::Add, true);
}

NDArray operator+(NDArray&& arr1, NDArray&& arr2) {
  return arrayArrayOpHelper(&arr1, &arr2, sd::pairwise::Add, true);
}

// Subtraction: NDArray - NDArray
NDArray operator-(NDArray& arr1, NDArray& arr2) {
  return arrayArrayOpHelper(&arr1, &arr2, sd::pairwise::Subtract, false);
}

NDArray operator-(NDArray&& arr1, NDArray& arr2) {
  return arrayArrayOpHelper(&arr1, &arr2, sd::pairwise::Subtract, true);
}

NDArray operator-(NDArray& arr1, NDArray&& arr2) {
  return arrayArrayOpHelper(&arr1, &arr2, sd::pairwise::Subtract, true);
}

NDArray operator-(NDArray&& arr1, NDArray&& arr2) {
  return arrayArrayOpHelper(&arr1, &arr2, sd::pairwise::Subtract, true);
}

// Multiplication: NDArray * NDArray
NDArray operator*(NDArray& arr1, NDArray& arr2) {
  return arrayArrayOpHelper(&arr1, &arr2, sd::pairwise::Multiply, false);
}

NDArray operator*(NDArray&& arr1, NDArray& arr2) {
  return arrayArrayOpHelper(&arr1, &arr2, sd::pairwise::Multiply, true);
}

NDArray operator*(NDArray& arr1, NDArray&& arr2) {
  return arrayArrayOpHelper(&arr1, &arr2, sd::pairwise::Multiply, true);
}

NDArray operator*(NDArray&& arr1, NDArray&& arr2) {
  return arrayArrayOpHelper(&arr1, &arr2, sd::pairwise::Multiply, true);
}

// Division: NDArray / NDArray
NDArray operator/(NDArray& arr1, NDArray& arr2) {
  return arrayArrayOpHelper(&arr1, &arr2, sd::pairwise::Divide, false);
}

NDArray operator/(NDArray&& arr1, NDArray& arr2) {
  return arrayArrayOpHelper(&arr1, &arr2, sd::pairwise::Divide, true);
}

NDArray operator/(NDArray& arr1, NDArray&& arr2) {
  return arrayArrayOpHelper(&arr1, &arr2, sd::pairwise::Divide, true);
}

NDArray operator/(NDArray&& arr1, NDArray&& arr2) {
  return arrayArrayOpHelper(&arr1, &arr2, sd::pairwise::Divide, true);
}

/////////////////////////////////////////////////////
// Compound Assignment Operators
/////////////////////////////////////////////////////

// Direct implementation for compound assignment with scalar values
template <typename T, typename>
void NDArray::operator+=(const T scalar) {
  if (isS()) {
    std::string errorMessage = "NDArray::operator+=: you can't use this method on String array!";
    THROW_EXCEPTION(errorMessage.c_str());
  }

  auto tmp = NDArrayFactory::create(dataType(), scalar, getContext());

  NDArray::prepareSpecialUse({this}, {this, &tmp});
  NativeOpExecutioner::execScalar(getContext(), sd::scalar::Add,
                                  buffer(), shapeInfo(), specialBuffer(), specialShapeInfo(),
                                  buffer(), shapeInfo(), specialBuffer(), specialShapeInfo(),
                                  tmp.buffer(), tmp.shapeInfo(), tmp.specialBuffer(), tmp.specialShapeInfo(), nullptr);
  NDArray::registerSpecialUse({this}, {this, &tmp});
}

template <typename T, typename>
void NDArray::operator-=(const T scalar) {
  if (isS()) {
    std::string errorMessage = "NDArray::operator-=: you can't use this method on String array!";
    THROW_EXCEPTION(errorMessage.c_str());
  }

  auto tmp = NDArrayFactory::create(dataType(), scalar, getContext());

  NDArray::prepareSpecialUse({this}, {this, &tmp});
  NativeOpExecutioner::execScalar(getContext(), sd::scalar::Subtract,
                                  buffer(), shapeInfo(), specialBuffer(), specialShapeInfo(),
                                  buffer(), shapeInfo(), specialBuffer(), specialShapeInfo(),
                                  tmp.buffer(), tmp.shapeInfo(), tmp.specialBuffer(), tmp.specialShapeInfo(), nullptr);
  NDArray::registerSpecialUse({this}, {this, &tmp});
}

template <typename T, typename>
void NDArray::operator*=(const T scalar) {
  if (isS()) {
    std::string errorMessage = "NDArray::operator*=: you can't use this method on String array!";
    THROW_EXCEPTION(errorMessage.c_str());
  }

  auto tmp = NDArrayFactory::create(dataType(), scalar, getContext());

  NDArray::prepareSpecialUse({this}, {this, &tmp});
  NativeOpExecutioner::execScalar(getContext(), sd::scalar::Multiply,
                                  buffer(), shapeInfo(), specialBuffer(), specialShapeInfo(),
                                  buffer(), shapeInfo(), specialBuffer(), specialShapeInfo(),
                                  tmp.buffer(), tmp.shapeInfo(), tmp.specialBuffer(), tmp.specialShapeInfo(), nullptr);
  NDArray::registerSpecialUse({this}, {this, &tmp});
}

template <typename T, typename>
void NDArray::operator/=(const T scalar) {
  if (isS()) {
    std::string errorMessage = "NDArray::operator/=: you can't use this method on String array!";
    THROW_EXCEPTION(errorMessage.c_str());
  }

  auto tmp = NDArrayFactory::create(dataType(), scalar, getContext());

  NDArray::prepareSpecialUse({this}, {this, &tmp});
  NativeOpExecutioner::execScalar(getContext(), sd::scalar::Divide,
                                  buffer(), shapeInfo(), specialBuffer(), specialShapeInfo(),
                                  buffer(), shapeInfo(), specialBuffer(), specialShapeInfo(),
                                  tmp.buffer(), tmp.shapeInfo(), tmp.specialBuffer(), tmp.specialShapeInfo(), nullptr);
  NDArray::registerSpecialUse({this}, {this, &tmp});
}

// Define instantiations matching the header file declarations exactly
#define INSTANTIATE_COMPOUND_ASSIGN_OP(OP, T) \
 template SD_LIB_EXPORT void NDArray::operator OP<GET_SECOND(T), typename std::enable_if<DataTypeUtils::scalarTypesForNDarray<GET_SECOND(T)>::value>::type>(const GET_SECOND(T) scalar);

#define INSTANTIATE_ALL_COMPOUND_ASSIGN_OPS(T) \
 INSTANTIATE_COMPOUND_ASSIGN_OP(+=, T) \
 INSTANTIATE_COMPOUND_ASSIGN_OP(-=, T) \
 INSTANTIATE_COMPOUND_ASSIGN_OP(*=, T) \
 INSTANTIATE_COMPOUND_ASSIGN_OP(/=, T)

ITERATE_LIST((SD_NUMERIC_TYPES), INSTANTIATE_ALL_COMPOUND_ASSIGN_OPS)

// Helper for compound assignment operators with arrays
void compoundAssignArray(NDArray& arr1, NDArray& arr2, sd::pairwise::Ops op, sd::BroadcastOpsTuple broadcastOp) {
  if (arr1.isS() || arr2.isS()) {
    std::string errorMessage = "compoundAssignArray: you can't use this method on String arrays!";
    THROW_EXCEPTION(errorMessage.c_str());
  }

  if (!Environment::getInstance().isExperimentalBuild() && arr1.dataType() != arr2.dataType() &&
      (arr1.dataType() != DataType::BOOL || arr2.dataType() != BOOL)) {
    std::string errorMessage;
    errorMessage += "compoundAssignArray: Cannot operate on arrays of different types: ";
    errorMessage += DataTypeUtils::asString(arr1.dataType());
    errorMessage += " and ";
    errorMessage += DataTypeUtils::asString(arr2.dataType());
    THROW_EXCEPTION(errorMessage.c_str());
  }

  if (arr1.lengthOf() == arr2.lengthOf() && arr1.rankOf() == arr2.rankOf()) {
    NDArray::prepareSpecialUse({&arr1}, {&arr1, &arr2});
    NativeOpExecutioner::execPairwiseTransform(
        arr1.getContext(), op,
        arr1.buffer(), arr1.shapeInfo(), arr1.specialBuffer(), arr1.specialShapeInfo(),
        arr2.buffer(), arr2.shapeInfo(), arr2.specialBuffer(), arr2.specialShapeInfo(),
        arr1.buffer(), arr1.shapeInfo(), arr1.specialBuffer(), arr1.specialShapeInfo(), nullptr);
    NDArray::registerSpecialUse({&arr1}, {&arr1, &arr2});
  } else {
    sd::LongType *bShape = nullptr;
    if (!ShapeUtils::evalBroadcastShapeInfo(arr1.shapeInfo(), arr2.shapeInfo(), true, bShape, arr1.getContext()->getWorkspace())) {
      std::string errorMessage = "compoundAssignArray: the shapes are not suitable for broadcast operation!";
      THROW_EXCEPTION(errorMessage.c_str());
    }

    if (shape::equalsTypesAndShapesSoft(arr1.shapeInfo(), bShape)) {
      arr1.applyTrueBroadcast(broadcastOp, &arr2, &arr1, false);
    } else {
      NDArray result(bShape, true, arr1.getContext());
      arr1.applyTrueBroadcast(broadcastOp, &arr2, &result, false);
      arr1 = std::move(result);
    }
  }
}

// Compound assignment with arrays
void NDArray::operator+=(NDArray& other) {
  compoundAssignArray(*this, other, sd::pairwise::Add, sd::BroadcastOpsTuple::Add());
}

void NDArray::operator+=(NDArray&& other) {
  operator+=(other);
}

void NDArray::operator-=(NDArray& other) {
  compoundAssignArray(*this, other, sd::pairwise::Subtract, sd::BroadcastOpsTuple::Subtract());
}

void NDArray::operator-=(NDArray&& other) {
  operator-=(other);
}

void NDArray::operator*=(NDArray& other) {
  compoundAssignArray(*this, other, sd::pairwise::Multiply, sd::BroadcastOpsTuple::Multiply());
}

void NDArray::operator*=(NDArray&& other) {
  operator*=(other);
}

void NDArray::operator/=(NDArray& other) {
  compoundAssignArray(*this, other, sd::pairwise::Divide, sd::BroadcastOpsTuple::Divide());
}

void NDArray::operator/=(NDArray&& other) {
  operator/=(other);
}

// Unary negative operator
NDArray NDArray::operator-() & {
  NDArray result(shapeInfo(), false, getContext());

  NDArray::prepareSpecialUse({&result}, {this});
  NativeOpExecutioner::execTransformSame(getContext(), sd::transform::Neg,
                                         buffer(), shapeInfo(), specialBuffer(), specialShapeInfo(),
                                         result.buffer(), result.shapeInfo(), result.specialBuffer(),
                                         result.specialShapeInfo(), nullptr, nullptr, nullptr);
  NDArray::registerSpecialUse({&result}, {this});

  return result;
}

NDArray NDArray::operator-() && {
  NDArray::prepareSpecialUse({this}, {this});
  NativeOpExecutioner::execTransformSame(getContext(), sd::transform::Neg,
                                         buffer(), shapeInfo(), specialBuffer(), specialShapeInfo(),
                                         buffer(), shapeInfo(), specialBuffer(), specialShapeInfo(),
                                         nullptr, nullptr, nullptr);
  NDArray::registerSpecialUse({this}, {this});

  return std::move(*this);
}

template <typename T1, typename T2, typename>
NDArray operator-(T1 &&arr1, T2 &&arr2) {
  if (arr1.isS() || arr2.isS())
    THROW_EXCEPTION("operator-(T&& arr1, T&& arr2): you can't use this method on String arrays!");
  if (!Environment::getInstance().isExperimentalBuild() && arr1.dataType() != arr2.dataType() &&
      (arr1.dataType() != DataType::BOOL || arr2.dataType() != BOOL))
    throw sd::datatype_exception::build("operator-(T&& arr1, T&& arr2): Cannot multiply different types",
                                        arr1.dataType(), arr2.dataType());

  PointersManager pointersManager(arr1.getContext(), "operator-(T&& arr1, T&& arr2)");

  if (arr1.lengthOf() == arr2.lengthOf() && arr1.rankOf() == arr2.rankOf()) {
    const bool isArr1Rvalue = !std::is_reference<T1>::value && !arr1.isView();
    const bool isArr2Rvalue = !std::is_reference<T2>::value && !arr2.isView();

    NDArray *result = nullptr;
    if (isArr1Rvalue)
      result = const_cast<NDArray *>(&arr1);
    else if (isArr2Rvalue)
      result = const_cast<NDArray *>(&arr2);
    else
      result = new NDArray(arr1.shapeInfo(), DataTypeUtils::pickPairwiseResultType(arr1.shapeInfo(), arr2.shapeInfo()),
                           false, arr1.getContext());

    NDArray::prepareSpecialUse({result}, {&arr1, &arr2});
    NativeOpExecutioner::execPairwiseTransform(
        arr1.getContext(), sd::pairwise::Subtract, arr1.buffer(), arr1.shapeInfo(), arr1.specialBuffer(),
        arr1.specialShapeInfo(), arr2.buffer(), arr2.shapeInfo(), arr2.specialBuffer(), arr2.specialShapeInfo(),
        result->buffer(), result->shapeInfo(), result->specialBuffer(), result->specialShapeInfo(), nullptr);
    NDArray::registerSpecialUse({result}, {&arr1, &arr2});

    if (!isArr1Rvalue && !isArr2Rvalue) {
      NDArray res = std::move(*result);
      delete result;
      return res;
    }

    return std::move(*result);
  }

  NDArray* arr1Addr = &arr1;
  NDArray* arr2Addr = &arr2;
  NDArray *ret =  std::forward<T1>(arr1).applyTrueBroadcast(sd::BroadcastOpsTuple::Subtract(), arr2Addr);
  return *ret;
}



template SD_LIB_EXPORT NDArray operator-<NDArray &, NDArray &, void>(NDArray &arr1, NDArray &arr2);
template SD_LIB_EXPORT NDArray operator-<NDArray &, NDArray, void>(NDArray &arr1, NDArray &&arr2);
template SD_LIB_EXPORT NDArray operator-<NDArray, NDArray &, void>(NDArray &&arr1, NDArray &arr2);
template SD_LIB_EXPORT NDArray operator-<NDArray, NDArray, void>(NDArray &&arr1, NDArray &&arr2);


template <typename T1, typename T2, typename>
NDArray operator*(T1 &&arr1, T2 &&arr2) {
  if (arr1.isS() || arr2.isS()) {
    THROW_EXCEPTION("operator*(T&& arr1, T&& arr2): you can't use this method on String arrays!");
  }
  if (!Environment::getInstance().isExperimentalBuild() && arr1.dataType() != arr2.dataType() &&
      (arr1.dataType() != DataType::BOOL || arr2.dataType() != BOOL)) {
    std::string errorMessage;
    errorMessage += "operator*(T&& arr1, T&& arr2): Cannot multiply different types";
    errorMessage += " arr1.dataType()=";
    errorMessage += DataTypeUtils::asString(arr1.dataType());
    errorMessage += " arr2.dataType()=";
    errorMessage += DataTypeUtils::asString(arr2.dataType());
    errorMessage += " arr1.shapeInfo()=";
    errorMessage += ShapeUtils::shapeAsString(arr1.shapeInfo());
    errorMessage += " arr2.shapeInfo()=";
    errorMessage += ShapeUtils::shapeAsString(arr2.shapeInfo());
    errorMessage += " arr1.ordering()=";
    THROW_EXCEPTION(errorMessage.c_str());
  }
  PointersManager pointersManager(arr1.getContext(), "operator*(T&& arr1, T&& arr2)");

  if (arr1.lengthOf() == arr2.lengthOf() && arr1.rankOf() == arr2.rankOf()) {
    const bool isArr1Rvalue = !std::is_reference<T1>::value && !arr1.isView();
    const bool isArr2Rvalue = !std::is_reference<T2>::value && !arr2.isView();

    NDArray *result = nullptr;
    if (isArr1Rvalue) {
      result = const_cast<NDArray *>(&arr1);
    }    else if (isArr2Rvalue) {
      result = const_cast<NDArray *>(&arr2);
    } else {
      result = new NDArray(arr1.shapeInfo(), DataTypeUtils::pickPairwiseResultType(arr1.shapeInfo(), arr2.shapeInfo()),
                           false, arr1.getContext());
    }
    NDArray::prepareSpecialUse({result}, {&arr1, &arr2});
    NativeOpExecutioner::execPairwiseTransform(
        arr1.getContext(), sd::pairwise::Multiply, arr1.buffer(), arr1.shapeInfo(), arr1.specialBuffer(),
        arr1.specialShapeInfo(), arr2.buffer(), arr2.shapeInfo(), arr2.specialBuffer(), arr2.specialShapeInfo(),
        result->buffer(), result->shapeInfo(), result->specialBuffer(), result->specialShapeInfo(), nullptr);
    NDArray::registerSpecialUse({result}, {&arr1, &arr2});

    if (!isArr1Rvalue && !isArr2Rvalue) {
      NDArray res = std::move(*result);
      delete result;
      return res;
    }

    return std::move(*result);
  }

  NDArray* arr1Addr = &arr1;
  NDArray* arr2Addr = &arr2;
  NDArray *ret =  std::forward<T1>(arr1).applyTrueBroadcast(sd::BroadcastOpsTuple::Multiply(), arr2Addr);
  return *ret;
}
template SD_LIB_EXPORT NDArray operator*<NDArray &, NDArray &, void>(NDArray &arr1, NDArray &arr2);
template SD_LIB_EXPORT NDArray operator*<NDArray &, NDArray, void>(NDArray &arr1, NDArray &&arr2);
template SD_LIB_EXPORT NDArray operator*<NDArray, NDArray &, void>(NDArray &&arr1, NDArray &arr2);
template SD_LIB_EXPORT NDArray operator*<NDArray&&, NDArray &&, void>(NDArray &&arr1, NDArray &&arr2);
template SD_LIB_EXPORT NDArray sd::operator*<NDArray, NDArray, void>(sd::NDArray&&, sd::NDArray&&);


template <typename T1, typename T2, typename>
NDArray operator+(T1 &&arr1, T2 &&arr2) {
  if (arr1.isS() || arr2.isS())
    THROW_EXCEPTION("operator+(T&& arr1, T&& arr2): you can't use this method on String arrays!");
  if (arr1.dataType() != arr2.dataType() &&
      (arr1.dataType() != DataType::BOOL || arr2.dataType() != BOOL)) {
    std::string errorMessage;
    errorMessage += "operator+(T&& arr1, T&& arr2): Cannot multiply different types";
    errorMessage += " arr1.dataType()=";
    errorMessage += DataTypeUtils::asString(arr1.dataType());
    errorMessage += " arr2.dataType()=";
    errorMessage += DataTypeUtils::asString(arr2.dataType());
    errorMessage += " arr1.shapeInfo()=";
    errorMessage += ShapeUtils::shapeAsString(arr1.shapeInfo());
    errorMessage += " arr2.shapeInfo()=";
    errorMessage += ShapeUtils::shapeAsString(arr2.shapeInfo());
    errorMessage += " arr1.ordering()=";
    THROW_EXCEPTION(errorMessage.c_str());

  }
  PointersManager pointersManager(arr1.getContext(), "operator+(T&& arr1, T&& arr2)");

  if (arr1.lengthOf() == arr2.lengthOf() && arr1.rankOf() == arr2.rankOf()) {
    const bool isArr1Rvalue = !std::is_reference<T1>::value && !arr1.isView();
    const bool isArr2Rvalue = !std::is_reference<T2>::value && !arr2.isView();

    NDArray *result = nullptr;
    if (isArr1Rvalue)
      result = const_cast<NDArray *>(&arr1);
    else if (isArr2Rvalue)
      result = const_cast<NDArray *>(&arr2);
    else
      result = new NDArray(arr1.shapeInfo(), DataTypeUtils::pickPairwiseResultType(arr1.shapeInfo(), arr2.shapeInfo()),
                           false, arr1.getContext());

    NDArray::prepareSpecialUse({result}, {&arr1, &arr2});
    NativeOpExecutioner::execPairwiseTransform(
        arr1.getContext(), sd::pairwise::Add, arr1.buffer(), arr1.shapeInfo(), arr1.specialBuffer(),
        arr1.specialShapeInfo(), arr2.buffer(), arr2.shapeInfo(), arr2.specialBuffer(), arr2.specialShapeInfo(),
        result->buffer(), result->shapeInfo(), result->specialBuffer(), result->specialShapeInfo(), nullptr);
    NDArray::registerSpecialUse({result}, {&arr1, &arr2});

    if (!isArr1Rvalue && !isArr2Rvalue) {
      NDArray res = std::move(*result);
      delete result;
      return res;
    }

    return std::move(*result);
  }

  NDArray* arr1Addr = &arr1;
  NDArray* arr2Addr = &arr2;
  NDArray *ret =  std::forward<T1>(arr1).applyTrueBroadcast(sd::BroadcastOpsTuple::Add(), arr2Addr);
  return *ret;
}
template SD_LIB_EXPORT NDArray operator+<NDArray &, NDArray &, void>(NDArray &arr1, NDArray &arr2);
template SD_LIB_EXPORT NDArray operator+<NDArray &, NDArray, void>(NDArray &arr1, NDArray &&arr2);
template SD_LIB_EXPORT NDArray operator+<NDArray, NDArray &, void>(NDArray &&arr1, NDArray &arr2);
template SD_LIB_EXPORT NDArray operator+<NDArray, NDArray, void>(NDArray &&arr1, NDArray &&arr2);


template <typename T1, typename T2, typename>
NDArray operator/(T1 &&arr1, T2 &&arr2) {
  if (arr1.isS() || arr2.isS())
    THROW_EXCEPTION("operator/(T&& arr1, T&& arr2): you can't use this method on String arrays!");
  if (!Environment::getInstance().isExperimentalBuild() && arr1.dataType() != arr2.dataType() &&
      (arr1.dataType() != DataType::BOOL || arr2.dataType() != BOOL))
    throw sd::datatype_exception::build("operator/(T&& arr1, T&& arr2): Cannot multiply different types",
                                        arr1.dataType(), arr2.dataType());

  PointersManager pointersManager(arr1.getContext(), "operator/(T&& arr1, T&& arr2)");

  if (arr1.lengthOf() == arr2.lengthOf() && arr1.rankOf() == arr2.rankOf()) {
    const bool isArr1Rvalue = !std::is_reference<T1>::value && !arr1.isView();
    const bool isArr2Rvalue = !std::is_reference<T2>::value && !arr2.isView();

    NDArray *result = nullptr;
    if (isArr1Rvalue)
      result = const_cast<NDArray *>(&arr1);
    else if (isArr2Rvalue)
      result = const_cast<NDArray *>(&arr2);
    else
      result = new NDArray(arr1.shapeInfo(), DataTypeUtils::pickPairwiseResultType(arr1.shapeInfo(), arr2.shapeInfo()),
                           false, arr1.getContext());

    NDArray::prepareSpecialUse({result}, {&arr1, &arr2});
    NativeOpExecutioner::execPairwiseTransform(
        arr1.getContext(), sd::pairwise::Divide, arr1.buffer(), arr1.shapeInfo(), arr1.specialBuffer(),
        arr1.specialShapeInfo(), arr2.buffer(), arr2.shapeInfo(), arr2.specialBuffer(), arr2.specialShapeInfo(),
        result->buffer(), result->shapeInfo(), result->specialBuffer(), result->specialShapeInfo(), nullptr);
    NDArray::registerSpecialUse({result}, {&arr1, &arr2});

    if (!isArr1Rvalue && !isArr2Rvalue) {
      NDArray res = std::move(*result);
      delete result;
      return res;
    }

    return std::move(*result);
  }

  NDArray* arr1Addr = &arr1;
  NDArray* arr2Addr = &arr2;
  NDArray *ret =  std::forward<T1>(arr1).applyTrueBroadcast(sd::BroadcastOpsTuple::Divide(), arr2Addr);
  return *ret;
}
template SD_LIB_EXPORT NDArray operator/<NDArray &, NDArray &, void>(NDArray &arr1, NDArray &arr2);
template SD_LIB_EXPORT NDArray operator/<NDArray &, NDArray, void>(NDArray &arr1, NDArray &&arr2);
template SD_LIB_EXPORT NDArray operator/<NDArray, NDArray &, void>(NDArray &&arr1, NDArray &arr2);
template SD_LIB_EXPORT NDArray operator/<NDArray&&, NDArray &&, void>(NDArray &&arr1, NDArray &&arr2);
template SD_LIB_EXPORT NDArray operator/<NDArray, NDArray, void>(sd::NDArray&&, sd::NDArray&&);

} // namespace sd