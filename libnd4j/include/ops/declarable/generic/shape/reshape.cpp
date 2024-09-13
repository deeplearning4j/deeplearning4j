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
// Created by raver119 on 29/10/17.
//

#include <system/op_boilerplate.h>
#if NOT_EXCLUDED(OP_reshape)
#include <ops/declarable/CustomOperations.h>
namespace sd {
namespace ops {

//////////////////////////////////////////////////////////////////////////
// here iArgs is a vector with (optional) negative of order as first element:
// ({-order, dim1, dim2, dim3, ...})
CUSTOM_OP_IMPL(reshape, 1, 1, false, 0, -2) {
  auto x = INPUT_VARIABLE(0);
  auto z = OUTPUT_VARIABLE(0);
  printf("reshape x offset %lld z offset %lld\n", x->offset(), z->offset());
  fflush(stdout);
  // Special case: empty.reshape(<other empty shape>) -> return empty
  if (x->isEmpty()) {
    REQUIRE_TRUE(z->isEmpty(), 0, "Reshape: when input is empty, output must also be empty");
    return Status::OK;  // No op
  }
  x->syncToHost();
  x->printBufferRaw("X INPUT FOR RESHAPE RAW");
  x->printIndexedBuffer("X INPUT FOR RESHAPE INDEXED");
  //scalars can either be 0 or 1
  if(!x->isScalar() && !x->isEmpty())
  REQUIRE_TRUE(x->lengthOf() == z->lengthOf(), 0,
               "Reshape: lengths before and after reshape should match, but "
               "got %i vs %i",
               x->lengthOf(), z->lengthOf());

  if (Environment::getInstance().isDebugAndVerbose()) sd_printv("Reshape: new shape", z->getShapeAsVector());
  if(z->ordering() != 'c' && z->ordering() != 'f') {
    std::string errorMessage;
    errorMessage += "Reshape: new shape has unknown order: [";
    errorMessage += z->ordering();
    errorMessage += "]";
    THROW_EXCEPTION(errorMessage.c_str());
  }

  //only perform assign when we aren't using a view
  if(x->dataBuffer() != z->dataBuffer()) {
    std::vector<sd::LongType> shape = z->getShapeAsVector();
    NDArray &reshapedX = x->reshape(z->ordering(), shape,true);
    reshapedX.printBufferRaw("RESHAPE X RAW:");
    reshapedX.printIndexedBuffer("RESHAPED INDEX BUFFER:");
    printf("before z assign call: z offset %lld\n", z->offset());
    z->assign(reshapedX);
  } else {
    z->printBufferRaw("RESHAPE Z BUFFER RAW:");
    z->printIndexedBuffer("RESHAPE Z INDEXED BUFFER:");
  }

  return Status::OK;
}

DECLARE_TYPES(reshape) {
  getOpDescriptor()->setAllowedInputTypes(0, ANY)->setAllowedInputTypes(1, {ALL_INTS})->setSameMode(true);
}

bool handleOptionalOrder(std::vector<LongType> &reshapeArgs, char &ordering) {
  if (reshapeArgs.size() > 0) {
    // check if any optional negative ordering value is passed
    auto optional = reshapeArgs[0];
    if (optional < 0) {
      optional = abs(optional);
      // check if passed option is allowed. (-1 -> dynamic shape)
      // in that case we will return back
      if (optional == 1) return true;
      // in this case it should obey allowed orderings
      if (optional != 'c' && optional != 'f') return false;
      reshapeArgs.erase(reshapeArgs.begin());
      // ordering was passed and ok. let's assign
      ordering = optional;
    }
  }
  // skipped
  return true;
}

DECLARE_SHAPE_FN(reshape) {
  const auto x = INPUT_VARIABLE(0);
  std::vector<LongType> reshapeArgs;
  std::vector<LongType> shapeNew;
  char orderNew = 'c';
  /**
   * NOTE: The value here is negative as a flag.
   * A negative value signifies 1 of 3 values:
   * -1 -> dynamic shape
   * -99 -> c ordering
   * -102 -> f ordering
   *
   */
  if (block.width() == 1) {
    reshapeArgs = *block.getIArguments();
    if (!handleOptionalOrder(reshapeArgs, orderNew)) {
      THROW_EXCEPTION(
          "reshape:: Value passed in must be -99 or -102 for the ordering if "
          "an int array is present. -99 represents c ordering and -102 "
          "represents f ordering. This number is negative for the long array "
          "case to flag the difference between an ordering and a dimension "
          "being specified.");
    };
  } else {
    reshapeArgs = INPUT_VARIABLE(1)->getBufferAsVector<LongType>();
    if (block.numI() > 0) {
      // Note here that the ordering for this case can not be negative.
      // Negative is used in the long array case to be used as a flag to
      // differentiate between a 99 or 102 shaped array and
      // the ordering. You can't have a -99 or -102 shaped array.
      char potentialOrdering = (char)I_ARG(0);
      if (!handleOptionalOrder(reshapeArgs, orderNew)) {
        THROW_EXCEPTION(
            "reshape:: Value passed in must be -99 or -102 for the ordering if "
            "an int array is present. -99 represents c ordering and -102 "
            "represents f ordering. This number is negative for the long array "
            "case to flag the difference between an ordering and a dimension "
            "being specified.");
      };

      orderNew = -potentialOrdering;
    }
  }

  LongType newShapeLen = 1;
  int pos = -1;
  bool newShapeEmpty = false;

  for (int i = 0; i < reshapeArgs.size(); i++) {
    const int dim = reshapeArgs[i];
    if (dim == -1) {
      REQUIRE_TRUE(pos == -1, 0, "Reshape : Only one unknown dimension (-1) is allowed.");
      pos = i;
      shapeNew.push_back(1);
    } else if (dim == 0) {
      shapeNew.push_back(0);
      newShapeEmpty = true;
    } else {
      shapeNew.push_back(dim);
      newShapeLen *= dim;
    }
  }

  if (pos != -1) {
    LongType xLen = x->lengthOf();
    if (x->isEmpty()) {
      xLen = 1;
      for (LongType i = 0; i < x->rankOf(); ++i)  // take into account possible empty shapes
        if (x->sizeAt(i) > 0 || !newShapeEmpty) xLen *= x->sizeAt(i);
    }

    shapeNew[pos] = xLen / newShapeLen;
  }

  if(newShapeEmpty) {
    for(int i = 0; i < reshapeArgs.size(); i++) {
      if(reshapeArgs[i] < 0)
        reshapeArgs[i] = 1;
    }
    return SHAPELIST(ConstantShapeHelper::getInstance().emptyShapeInfoWithShape(x->dataType(), reshapeArgs));
  }


  auto len = shape::prodLong(shapeNew.data(), shapeNew.size());
  if(!x->isScalar() && !x->isEmpty())
  REQUIRE_TRUE(x->lengthOf() == len, 0,
               "Reshape: lengths before and after reshape should match, but "
               "got %i vs %i",
               x->lengthOf(), len);

  return SHAPELIST(ConstantShapeHelper::getInstance().createShapeInfo(x->dataType(), orderNew, shapeNew));
}

}  // namespace ops
}  // namespace sd

#endif
