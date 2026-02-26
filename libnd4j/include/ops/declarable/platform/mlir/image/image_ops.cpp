/* ******************************************************************************
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
 * See the NOTICE file distributed with this work for additional
 * information regarding copyright ownership.
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

//
// MLIR-accelerated image processing operations
//

#include <ops/declarable/PlatformHelper.h>
#include <ops/declarable/OpRegistrator.h>
#include <system/platform_boilerplate.h>
#include <ops/declarable/platform/mlir/mlirUtils.h>

#if defined(HAVE_MLIR)

namespace sd {
namespace ops {
namespace platforms {

//////////////////////////////////////////////////////////////////////////
// resize_bilinear MLIR implementation
//////////////////////////////////////////////////////////////////////////

DECLARE_PLATFORM(resize_bilinear, ENGINE_CPU)

PLATFORM_IMPL(resize_bilinear, ENGINE_CPU) {
    auto* input = INPUT_VARIABLE(0);     // [batch, height, width, channels]
    auto* size = INPUT_VARIABLE(1);      // [newHeight, newWidth]
    auto* output = OUTPUT_VARIABLE(0);

    bool alignCorners = block.numB() > 0 ? B_ARG(0) : false;
    bool halfPixelCenters = block.numB() > 1 ? B_ARG(1) : false;

    std::vector<NDArray*> inputs = {input, size};
    std::vector<NDArray*> outputs = {output};

    auto status = executeMlirEx("resize_bilinear", block, inputs, outputs);

    if (status != Status::OK()) {
        return Status::CODE(ND4J_STATUS_BAD_ARGUMENTS, "MLIR resize_bilinear failed");
    }

    return Status::OK();
}

PLATFORM_CHECK(resize_bilinear, ENGINE_CPU) {
    auto* input = INPUT_VARIABLE(0);

    Requirements req("MLIR RESIZE_BILINEAR");

    req.expectTrue(mlirEnabled(), "MLIR enabled") &&
    req.expectIn(input->dataType(), {FLOAT32, DOUBLE, HALF, BFLOAT16}, "Supported dtype") &&
    req.expectTrue(input->lengthOf() >= MLIR_MIN_TENSOR_SIZE, "Size threshold");

    return req;
}

//////////////////////////////////////////////////////////////////////////
// resize_nearest_neighbor MLIR implementation
//////////////////////////////////////////////////////////////////////////

DECLARE_PLATFORM(resize_nearest_neighbor, ENGINE_CPU)

PLATFORM_IMPL(resize_nearest_neighbor, ENGINE_CPU) {
    auto* input = INPUT_VARIABLE(0);
    auto* size = INPUT_VARIABLE(1);
    auto* output = OUTPUT_VARIABLE(0);

    bool alignCorners = block.numB() > 0 ? B_ARG(0) : false;
    bool halfPixelCenters = block.numB() > 1 ? B_ARG(1) : false;

    std::vector<NDArray*> inputs = {input, size};
    std::vector<NDArray*> outputs = {output};

    auto status = executeMlirEx("resize_nearest_neighbor", block, inputs, outputs);

    if (status != Status::OK()) {
        return Status::CODE(ND4J_STATUS_BAD_ARGUMENTS, "MLIR resize_nearest_neighbor failed");
    }

    return Status::OK();
}

PLATFORM_CHECK(resize_nearest_neighbor, ENGINE_CPU) {
    auto* input = INPUT_VARIABLE(0);

    Requirements req("MLIR RESIZE_NEAREST_NEIGHBOR");

    req.expectTrue(mlirEnabled(), "MLIR enabled") &&
    req.expectIn(input->dataType(), {FLOAT32, DOUBLE, HALF, BFLOAT16}, "Supported dtype") &&
    req.expectTrue(input->lengthOf() >= MLIR_MIN_TENSOR_SIZE, "Size threshold");

    return req;
}

//////////////////////////////////////////////////////////////////////////
// resize_bicubic MLIR implementation
//////////////////////////////////////////////////////////////////////////

DECLARE_PLATFORM(resize_bicubic, ENGINE_CPU)

PLATFORM_IMPL(resize_bicubic, ENGINE_CPU) {
    auto* input = INPUT_VARIABLE(0);
    auto* size = INPUT_VARIABLE(1);
    auto* output = OUTPUT_VARIABLE(0);

    bool alignCorners = block.numB() > 0 ? B_ARG(0) : false;
    bool halfPixelCenters = block.numB() > 1 ? B_ARG(1) : false;

    std::vector<NDArray*> inputs = {input, size};
    std::vector<NDArray*> outputs = {output};

    auto status = executeMlirEx("resize_bicubic", block, inputs, outputs);

    if (status != Status::OK()) {
        return Status::CODE(ND4J_STATUS_BAD_ARGUMENTS, "MLIR resize_bicubic failed");
    }

    return Status::OK();
}

PLATFORM_CHECK(resize_bicubic, ENGINE_CPU) {
    auto* input = INPUT_VARIABLE(0);

    Requirements req("MLIR RESIZE_BICUBIC");

    req.expectTrue(mlirEnabled(), "MLIR enabled") &&
    req.expectIn(input->dataType(), {FLOAT32, DOUBLE, HALF, BFLOAT16}, "Supported dtype") &&
    req.expectTrue(input->lengthOf() >= MLIR_MIN_TENSOR_SIZE, "Size threshold");

    return req;
}

//////////////////////////////////////////////////////////////////////////
// crop_and_resize MLIR implementation
//////////////////////////////////////////////////////////////////////////

DECLARE_PLATFORM(crop_and_resize, ENGINE_CPU)

PLATFORM_IMPL(crop_and_resize, ENGINE_CPU) {
    auto* image = INPUT_VARIABLE(0);       // [batch, height, width, channels]
    auto* boxes = INPUT_VARIABLE(1);       // [numBoxes, 4] - normalized coords
    auto* boxIndices = INPUT_VARIABLE(2);  // [numBoxes]
    auto* cropSize = INPUT_VARIABLE(3);    // [cropHeight, cropWidth]
    auto* output = OUTPUT_VARIABLE(0);

    int method = block.numI() > 0 ? INT_ARG(0) : 0;  // 0=bilinear, 1=nearest
    double extrapolationValue = block.numT() > 0 ? T_ARG(0) : 0.0;

    std::vector<NDArray*> inputs = {image, boxes, boxIndices, cropSize};
    std::vector<NDArray*> outputs = {output};

    auto status = executeMlirEx("crop_and_resize", block, inputs, outputs);

    if (status != Status::OK()) {
        return Status::CODE(ND4J_STATUS_BAD_ARGUMENTS, "MLIR crop_and_resize failed");
    }

    return Status::OK();
}

PLATFORM_CHECK(crop_and_resize, ENGINE_CPU) {
    auto* image = INPUT_VARIABLE(0);

    Requirements req("MLIR CROP_AND_RESIZE");

    req.expectTrue(mlirEnabled(), "MLIR enabled") &&
    req.expectIn(image->dataType(), {FLOAT32, DOUBLE, HALF, BFLOAT16}, "Supported dtype") &&
    req.expectTrue(image->lengthOf() >= MLIR_MIN_TENSOR_SIZE, "Size threshold");

    return req;
}

//////////////////////////////////////////////////////////////////////////
// rgb_to_hsv MLIR implementation
//////////////////////////////////////////////////////////////////////////

DECLARE_PLATFORM(rgb_to_hsv, ENGINE_CPU)

PLATFORM_IMPL(rgb_to_hsv, ENGINE_CPU) {
    auto* input = INPUT_VARIABLE(0);   // [..., 3] RGB values in [0, 1]
    auto* output = OUTPUT_VARIABLE(0); // [..., 3] HSV values

    std::vector<NDArray*> inputs = {input};
    std::vector<NDArray*> outputs = {output};

    auto status = executeMlirEx("rgb_to_hsv", block, inputs, outputs);

    if (status != Status::OK()) {
        return Status::CODE(ND4J_STATUS_BAD_ARGUMENTS, "MLIR rgb_to_hsv failed");
    }

    return Status::OK();
}

PLATFORM_CHECK(rgb_to_hsv, ENGINE_CPU) {
    auto* input = INPUT_VARIABLE(0);

    Requirements req("MLIR RGB_TO_HSV");

    req.expectTrue(mlirEnabled(), "MLIR enabled") &&
    req.expectIn(input->dataType(), {FLOAT32, DOUBLE, HALF, BFLOAT16}, "Supported dtype") &&
    req.expectTrue(input->lengthOf() >= MLIR_MIN_TENSOR_SIZE, "Size threshold");

    return req;
}

//////////////////////////////////////////////////////////////////////////
// hsv_to_rgb MLIR implementation
//////////////////////////////////////////////////////////////////////////

DECLARE_PLATFORM(hsv_to_rgb, ENGINE_CPU)

PLATFORM_IMPL(hsv_to_rgb, ENGINE_CPU) {
    auto* input = INPUT_VARIABLE(0);   // [..., 3] HSV values
    auto* output = OUTPUT_VARIABLE(0); // [..., 3] RGB values in [0, 1]

    std::vector<NDArray*> inputs = {input};
    std::vector<NDArray*> outputs = {output};

    auto status = executeMlirEx("hsv_to_rgb", block, inputs, outputs);

    if (status != Status::OK()) {
        return Status::CODE(ND4J_STATUS_BAD_ARGUMENTS, "MLIR hsv_to_rgb failed");
    }

    return Status::OK();
}

PLATFORM_CHECK(hsv_to_rgb, ENGINE_CPU) {
    auto* input = INPUT_VARIABLE(0);

    Requirements req("MLIR HSV_TO_RGB");

    req.expectTrue(mlirEnabled(), "MLIR enabled") &&
    req.expectIn(input->dataType(), {FLOAT32, DOUBLE, HALF, BFLOAT16}, "Supported dtype") &&
    req.expectTrue(input->lengthOf() >= MLIR_MIN_TENSOR_SIZE, "Size threshold");

    return req;
}

//////////////////////////////////////////////////////////////////////////
// rgb_to_grayscale MLIR implementation
//////////////////////////////////////////////////////////////////////////

DECLARE_PLATFORM(rgb_to_grs, ENGINE_CPU)

PLATFORM_IMPL(rgb_to_grs, ENGINE_CPU) {
    auto* input = INPUT_VARIABLE(0);   // [..., 3] RGB values
    auto* output = OUTPUT_VARIABLE(0); // [..., 1] grayscale

    std::vector<NDArray*> inputs = {input};
    std::vector<NDArray*> outputs = {output};

    auto status = executeMlirEx("rgb_to_grayscale", block, inputs, outputs);

    if (status != Status::OK()) {
        return Status::CODE(ND4J_STATUS_BAD_ARGUMENTS, "MLIR rgb_to_grayscale failed");
    }

    return Status::OK();
}

PLATFORM_CHECK(rgb_to_grs, ENGINE_CPU) {
    auto* input = INPUT_VARIABLE(0);

    Requirements req("MLIR RGB_TO_GRAYSCALE");

    req.expectTrue(mlirEnabled(), "MLIR enabled") &&
    req.expectIn(input->dataType(), {FLOAT32, DOUBLE, HALF, BFLOAT16}, "Supported dtype") &&
    req.expectTrue(input->lengthOf() >= MLIR_MIN_TENSOR_SIZE, "Size threshold");

    return req;
}

//////////////////////////////////////////////////////////////////////////
// adjust_contrast MLIR implementation
//////////////////////////////////////////////////////////////////////////

DECLARE_PLATFORM(adjust_contrast, ENGINE_CPU)

PLATFORM_IMPL(adjust_contrast, ENGINE_CPU) {
    auto* input = INPUT_VARIABLE(0);
    auto* output = OUTPUT_VARIABLE(0);

    double factor = T_ARG(0);

    std::vector<NDArray*> inputs = {input};
    std::vector<NDArray*> outputs = {output};

    auto status = executeMlirEx("adjust_contrast", block, inputs, outputs);

    if (status != Status::OK()) {
        return Status::CODE(ND4J_STATUS_BAD_ARGUMENTS, "MLIR adjust_contrast failed");
    }

    return Status::OK();
}

PLATFORM_CHECK(adjust_contrast, ENGINE_CPU) {
    auto* input = INPUT_VARIABLE(0);

    Requirements req("MLIR ADJUST_CONTRAST");

    req.expectTrue(mlirEnabled(), "MLIR enabled") &&
    req.expectIn(input->dataType(), {FLOAT32, DOUBLE, HALF, BFLOAT16}, "Supported dtype") &&
    req.expectTrue(input->lengthOf() >= MLIR_MIN_TENSOR_SIZE, "Size threshold");

    return req;
}

//////////////////////////////////////////////////////////////////////////
// adjust_hue MLIR implementation
//////////////////////////////////////////////////////////////////////////

DECLARE_PLATFORM(adjust_hue, ENGINE_CPU)

PLATFORM_IMPL(adjust_hue, ENGINE_CPU) {
    auto* input = INPUT_VARIABLE(0);
    auto* output = OUTPUT_VARIABLE(0);

    double delta = T_ARG(0);  // hue adjustment in [-1, 1]

    std::vector<NDArray*> inputs = {input};
    std::vector<NDArray*> outputs = {output};

    auto status = executeMlirEx("adjust_hue", block, inputs, outputs);

    if (status != Status::OK()) {
        return Status::CODE(ND4J_STATUS_BAD_ARGUMENTS, "MLIR adjust_hue failed");
    }

    return Status::OK();
}

PLATFORM_CHECK(adjust_hue, ENGINE_CPU) {
    auto* input = INPUT_VARIABLE(0);

    Requirements req("MLIR ADJUST_HUE");

    req.expectTrue(mlirEnabled(), "MLIR enabled") &&
    req.expectIn(input->dataType(), {FLOAT32, DOUBLE, HALF, BFLOAT16}, "Supported dtype") &&
    req.expectTrue(input->lengthOf() >= MLIR_MIN_TENSOR_SIZE, "Size threshold");

    return req;
}

//////////////////////////////////////////////////////////////////////////
// adjust_saturation MLIR implementation
//////////////////////////////////////////////////////////////////////////

DECLARE_PLATFORM(adjust_saturation, ENGINE_CPU)

PLATFORM_IMPL(adjust_saturation, ENGINE_CPU) {
    auto* input = INPUT_VARIABLE(0);
    auto* output = OUTPUT_VARIABLE(0);

    double factor = T_ARG(0);

    std::vector<NDArray*> inputs = {input};
    std::vector<NDArray*> outputs = {output};

    auto status = executeMlirEx("adjust_saturation", block, inputs, outputs);

    if (status != Status::OK()) {
        return Status::CODE(ND4J_STATUS_BAD_ARGUMENTS, "MLIR adjust_saturation failed");
    }

    return Status::OK();
}

PLATFORM_CHECK(adjust_saturation, ENGINE_CPU) {
    auto* input = INPUT_VARIABLE(0);

    Requirements req("MLIR ADJUST_SATURATION");

    req.expectTrue(mlirEnabled(), "MLIR enabled") &&
    req.expectIn(input->dataType(), {FLOAT32, DOUBLE, HALF, BFLOAT16}, "Supported dtype") &&
    req.expectTrue(input->lengthOf() >= MLIR_MIN_TENSOR_SIZE, "Size threshold");

    return req;
}

//////////////////////////////////////////////////////////////////////////
// image_resize (general) MLIR implementation
//////////////////////////////////////////////////////////////////////////

DECLARE_PLATFORM(image_resize, ENGINE_CPU)

PLATFORM_IMPL(image_resize, ENGINE_CPU) {
    auto* input = INPUT_VARIABLE(0);
    auto* size = INPUT_VARIABLE(1);
    auto* output = OUTPUT_VARIABLE(0);

    // 0=bilinear, 1=nearest, 2=bicubic, 3=area, 4=lanczos3, 5=lanczos5, 6=gaussian
    int method = block.numI() > 0 ? INT_ARG(0) : 0;
    bool preserveAspectRatio = block.numB() > 0 ? B_ARG(0) : false;
    bool antialias = block.numB() > 1 ? B_ARG(1) : false;

    std::vector<NDArray*> inputs = {input, size};
    std::vector<NDArray*> outputs = {output};

    auto status = executeMlirEx("image_resize", block, inputs, outputs);

    if (status != Status::OK()) {
        return Status::CODE(ND4J_STATUS_BAD_ARGUMENTS, "MLIR image_resize failed");
    }

    return Status::OK();
}

PLATFORM_CHECK(image_resize, ENGINE_CPU) {
    auto* input = INPUT_VARIABLE(0);

    Requirements req("MLIR IMAGE_RESIZE");

    req.expectTrue(mlirEnabled(), "MLIR enabled") &&
    req.expectIn(input->dataType(), {FLOAT32, DOUBLE, HALF, BFLOAT16}, "Supported dtype") &&
    req.expectTrue(input->lengthOf() >= MLIR_MIN_TENSOR_SIZE, "Size threshold");

    return req;
}

} // namespace platforms
} // namespace ops
} // namespace sd

#endif // HAVE_MLIR
