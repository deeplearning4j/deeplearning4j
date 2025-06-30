/*
 *  ******************************************************************************
 *  *
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
 *  * See the NOTICE file distributed with this work for additional
 *  * information regarding copyright ownership.
 *  * Unless required by applicable law or agreed to in writing, software
 *  * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 *  * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 *  * License for the specific language governing permissions and limitations
 *  * under the License.
 *  *
 *  * SPDX-License-Identifier: Apache-2.0
 *  *****************************************************************************
 */
/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

//
//  @author sgazeos@gmail.com
//
#include <execution/Threads.h>
#include <ops/declarable/helpers/crop_and_resize.h>
#include <system/selective_rendering.h>
#if NOT_EXCLUDED(OP_crop_and_resize)
namespace sd {
namespace ops {
namespace helpers {

// ------------------------------------------------------------------------------------------------------------------ //
// ------------------------------------------------------------------------------------------------------------------ //
// crop and resize helper functor:
// \@param context - launch context for operation
// \@param images - batch of images (4D tensor) with shape {batch, width, height, channels} with given type
// \@param boxes - float boxes for crop
// \@param indices - integer boxes indices for crop
// \@param cropSize - integer size (newWidth, newHeight)
// \@param method - one of bilinear (0) or nearest neighbour (1) interpolation algorithm
// \@param extrapolationVal - radix to increase/decrease image
// \@param crops - output image batch (4D with given type)
//

void cropAndResizeFunctor(sd::LaunchContext *context, NDArray *images, NDArray *boxes,
                           NDArray *indices, NDArray *cropSize, int method, double extrapolationVal,
                          NDArray *crops) {
  auto imagesDType = images->dataType();
  auto boxesDType = boxes->dataType();
  auto indicesDType = indices->dataType();
#if SD_IS_TRIPLE_TYPE_COMPILED(imagesDType,boxesDType,indicesDType)
  BUILD_TRIPLE_SELECTOR(images->dataType(), boxes->dataType(), indices->dataType(), cropAndResizeFunctor_,
                        ( images, boxes, indices, cropSize, method, extrapolationVal, crops), SD_NUMERIC_TYPES,
                        SD_FLOAT_TYPES, SD_INTEGER_TYPES);
#endif
}



}  // namespace helpers
}  // namespace ops
}  // namespace sd
#endif