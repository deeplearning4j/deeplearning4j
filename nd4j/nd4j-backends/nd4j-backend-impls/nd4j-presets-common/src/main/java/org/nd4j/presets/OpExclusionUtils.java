package org.nd4j.presets;/*
 *  ******************************************************************************
 *  *
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
 *  *  See the NOTICE file distributed with this work for additional
 *  *  information regarding copyright ownership.
 *  * Unless required by applicable law or agreed to in writing, software
 *  * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 *  * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 *  * License for the specific language governing permissions and limitations
 *  * under the License.
 *  *
 *  * SPDX-License-Identifier: Apache-2.0
 *  *****************************************************************************
 */

import org.bytedeco.javacpp.tools.Info;
import org.bytedeco.javacpp.tools.InfoMap;
import org.bytedeco.javacpp.tools.Logger;

import java.io.File;
import java.io.IOException;
import java.util.*;

public class OpExclusionUtils {


    private static final String[] COMMON_MACROS = {
            "thread_local", "SD_LIB_EXPORT", "SD_INLINE", "CUBLASWINAPI",
            "SD_HOST", "SD_DEVICE", "SD_KERNEL", "SD_HOST_DEVICE", "SD_ALL_OPS", "NOT_EXCLUDED"
    };

    private static final String[] SKIP_HEADERS = {
            "openblas_config.h", "cblas.h", "lapacke_config.h", "lapacke_mangling.h",
            "lapack.h", "lapacke.h", "lapacke_utils.h"
    };

    private static final String[] OBJECTIFY_HEADERS = {
            "NativeOps.h", "build_info.h"
    };

    private static final String[] OPAQUE_TYPES = {
            "NDArray", "TadPack", "ResultWrapper", "ShapeList", "VariablesSet",
            "Variable", "ConstantDataBuffer", "ConstantShapeBuffer", "ConstantOffsetsBuffer",
            "DataBuffer", "Context", "RandomGenerator", "LaunchContext"
    };

    private static final String[] SKIP_CLASSES = {
            "sd::graph::FlowPath", "sd::graph::Graph", "sd::ConstantDataBuffer", "sd::ConstantHolder",
            "sd::ConstantOffsetsBuffer", "sd::ShapeList", "sd::graph::GraphProfile", "sd::LaunchContext",
            "sd::TadDescriptor", "sd::ShapeDescriptor", "sd::TadPack", "sd::NDIndex", "sd::NDIndexPoint",
            "sd::NDIndexAll", "sd::NDIndexInterval", "sd::IndicesList", "sd::ArgumentsList",
            "sd::graph::GraphState", "sd::graph::VariableSpace"
    };

    private static final String[] EXCLUDED_FUNCTIONS = {
            "std::initializer_list", "cnpy::NpyArray", "sd::NDArray::applyLambda",
            "sd::NDArray::applyPairwiseLambda", "sd::graph::FlatResult", "sd::graph::FlatVariable",
            "sd::NDArray::subarray", "std::shared_ptr", "sd::PointerWrapper", "sd::PointerDeallocator",
            "instrumentFile", "setInstrumentOut", "closeInstrumentOut", "__cyg_profile_func_exit",
            "__cyg_profile_func_enter"
    };

    private static final String[] SKIP_OPS = {
            "sd::ops::platforms::PlatformHelper", "sd::ops::DeclarableOp", "sd::ops::BroadcastableOp",
            "sd::ops::BroadcastableBoolOp", "sd::ops::DeclarableListOp", "sd::ops::DeclarableReductionOp",
            "sd::ops::DeclarableCustomOp", "sd::ops::LogicOp", "sd::ops::Switch", "sd::ops::While",
            "sd::ops::Scope", "sd::ops::Conditional", "sd::ops::Return", "sd::ops::expose", "sd::ops::invoke",
            "sd::ops::sigmoid", "sd::ops::sigmoid_bp", "sd::ops::softsign", "sd::ops::softsign_bp",
            "sd::ops::tanh", "sd::ops::tanh_bp", "sd::ops::softplus", "sd::ops::softplus_bp", "sd::ops::relu",
            "sd::ops::relu_bp", "sd::ops::selu", "sd::ops::selu_bp", "sd::ops::lrelu", "sd::ops::lrelu_bp",
            "sd::ops::elu", "sd::ops::elu_bp", "sd::ops::cube", "sd::ops::cube_bp", "sd::ops::rectifiedtanh",
            "sd::ops::rectifiedtanh_bp", "sd::ops::rationaltanh", "sd::ops::rationaltanh_bp", "sd::ops::hardtanh",
            "sd::ops::hardtanh_bp", "sd::ops::hardsigmoid", "sd::ops::hardsigmoid_bp", "sd::ops::identity",
            "sd::ops::identity_bp", "sd::ops::identity_n", "sd::ops::crelu", "sd::ops::crelu_bp", "sd::ops::relu6",
            "sd::ops::relu6_bp", "sd::ops::prelu", "sd::ops::prelu_bp", "sd::ops::thresholdedrelu",
            "sd::ops::thresholdedrelu_bp", "sd::ops::lt_scalar", "sd::ops::gt_scalar", "sd::ops::lte_scalar",
            "sd::ops::gte_scalar", "sd::ops::eq_scalar", "sd::ops::neq_scalar", "sd::ops::Where", "sd::ops::where_np",
            "sd::ops::select", "sd::ops::choose", "sd::ops::is_non_decreasing", "sd::ops::is_strictly_increasing",
            "sd::ops::is_numeric_tensor", "sd::ops::boolean_not", "sd::ops::maximum", "sd::ops::maximum_bp",
            "sd::ops::minimum", "sd::ops::minimum_bp", "sd::ops::add", "sd::ops::add_bp", "sd::ops::subtract",
            "sd::ops::subtract_bp", "sd::ops::reversesubtract", "sd::ops::reversesubtract_bp", "sd::ops::reversemod",
            "sd::ops::reversemod_bp", "sd::ops::squaredsubtract", "sd::ops::squaredsubtract_bp", "sd::ops::multiply",
            "sd::ops::multiply_bp", "sd::ops::divide", "sd::ops::divide_bp", "sd::ops::divide_no_nan",
            "sd::ops::reversedivide", "sd::ops::reversedivide_bp", "sd::ops::floormod", "sd::ops::floormod_bp",
            "sd::ops::mod", "sd::ops::mod_bp", "sd::ops::floordiv", "sd::ops::floordiv_bp", "sd::ops::realdiv",
            "sd::ops::realdiv_bp", "sd::ops::truncatediv", "sd::ops::assign", "sd::ops::assign_bp", "sd::ops::meshgrid",
            "sd::ops::equals", "sd::ops::not_equals", "sd::ops::less_equal", "sd::ops::greater_equal", "sd::ops::less",
            "sd::ops::greater", "sd::ops::boolean_and", "sd::ops::boolean_or", "sd::ops::boolean_xor",
            "sd::ops::percentile", "sd::ops::tf_atan2", "sd::ops::Pow", "sd::ops::Pow_bp", "sd::ops::igamma",
            "sd::ops::igammac", "sd::ops::conv1d", "sd::ops::conv1d_bp", "sd::ops::conv2d", "sd::ops::conv2d_bp",
            "sd::ops::conv2d_input_bp", "sd::ops::sconv2d", "sd::ops::sconv2d_bp", "sd::ops::deconv2d",
            "sd::ops::deconv2d_bp", "sd::ops::deconv3d", "sd::ops::deconv3d_bp", "sd::ops::maxpool2d",
            "sd::ops::maxpool2d_bp", "sd::ops::avgpool2d", "sd::ops::avgpool2d_bp", "sd::ops::pnormpool2d",
            "sd::ops::pnormpool2d_bp", "sd::ops::im2col", "sd::ops::im2col_bp", "sd::ops::col2im",
            "sd::ops::upsampling2d", "sd::ops::upsampling2d_bp", "sd::ops::upsampling3d", "sd::ops::upsampling3d_bp",
            "sd::ops::ismax", "sd::ops::dilation2d", "sd::ops::conv3dnew", "sd::ops::conv3dnew_bp",
            "sd::ops::avgpool3dnew", "sd::ops::avgpool3dnew_bp", "sd::ops::maxpool3dnew", "sd::ops::maxpool3dnew_bp",
            "sd::ops::max_pool_with_argmax", "sd::ops::depthwise_conv2d", "sd::ops::depthwise_conv2d_bp",
            "sd::ops::pointwise_conv2d", "sd::ops::deconv2d_tf", "sd::ops::write_list", "sd::ops::stack_list",
            "sd::ops::read_list", "sd::ops::pick_list", "sd::ops::size_list", "sd::ops::create_list",
            "sd::ops::delete_list", "sd::ops::scatter_list", "sd::ops::split_list", "sd::ops::gather_list",
            "sd::ops::clone_list", "sd::ops::unstack_list", "sd::ops::sru", "sd::ops::sru_bp", "sd::ops::sru_bi",
            "sd::ops::sru_bi_bp", "sd::ops::lstmCell", "sd::ops::lstmLayerCell", "sd::ops::lstmLayerCellBp",
            "sd::ops::lstmBlockCell", "sd::ops::lstmBlock", "sd::ops::lstmLayer", "sd::ops::lstmLayer_bp",
            "sd::ops::sruCell", "sd::ops::gruCell", "sd::ops::gruCell_bp", "sd::ops::lstm", "sd::ops::gru",
            "sd::ops::gru_bp", "sd::ops::static_rnn", "sd::ops::dynamic_rnn", "sd::ops::static_bidirectional_rnn",
            "sd::ops::dynamic_bidirectional_rnn", "sd::ops::clipbyvalue", "sd::ops::clipbynorm", "sd::ops::clipbynorm_bp",
            "sd::ops::clipbyavgnorm", "sd::ops::clipbyavgnorm_bp", "sd::ops::cumsum", "sd::ops::cumprod", "sd::ops::tile",
            "sd::ops::tile_bp", "sd::ops::repeat", "sd::ops::invert_permutation", "sd::ops::concat", "sd::ops::concat_bp",
            "sd::ops::mergemax", "sd::ops::mergemax_bp", "sd::ops::mergemaxindex", "sd::ops::mergeadd",
            "sd::ops::mergeadd_bp", "sd::ops::mergeavg", "sd::ops::mergeavg_bp", "sd::ops::scatter_update",
            "sd::ops::Floor", "sd::ops::Log1p", "sd::ops::reverse", "sd::ops::reverse_bp", "sd::ops::gather",
            "sd::ops::pad", "sd::ops::eye", "sd::ops::gather_nd", "sd::ops::reverse_sequence", "sd::ops::trace",
            "sd::ops::random_shuffle", "sd::ops::clip_by_global_norm", "sd::ops::tri", "sd::ops::triu",
            "sd::ops::triu_bp", "sd::ops::mirror_pad", "sd::ops::cumsum_bp", "sd::ops::cumprod_bp", "sd::ops::flatten",
            "sd::ops::histogram_fixed_width", "sd::ops::standardize", "sd::ops::standardize_bp", "sd::ops::hashcode",
            "sd::ops::histogram", "sd::ops::argmax", "sd::ops::argmin", "sd::ops::argamax", "sd::ops::argamin",
            "sd::ops::norm", "sd::ops::matrix_set_diag", "sd::ops::matrix_diag", "sd::ops::betainc", "sd::ops::biasadd",
            "sd::ops::biasadd_bp", "sd::ops::diag", "sd::ops::diag_part", "sd::ops::matrix_diag_part", "sd::ops::qr",
            "sd::ops::listdiff", "sd::ops::scatter_add", "sd::ops::scatter_sub", "sd::ops::scatter_mul",
            "sd::ops::scatter_div", "sd::ops::scatter_upd", "sd::ops::scatter_max", "sd::ops::scatter_min",
            "sd::ops::scatter_nd", "sd::ops::scatter_nd_update", "sd::ops::scatter_nd_add", "sd::ops::scatter_nd_sub",
            "sd::ops::fill_as", "sd::ops::rint", "sd::ops::unique", "sd::ops::unique_with_counts", "sd::ops::tear",
            "sd::ops::unstack", "sd::ops::strided_slice", "sd::ops::strided_slice_bp", "sd::ops::create_view",
            "sd::ops::slice", "sd::ops::slice_bp", "sd::ops::range", "sd::ops::onehot", "sd::ops::confusion_matrix",
            "sd::ops::stack", "sd::ops::size", "sd::ops::rank", "sd::ops::broadcastgradientargs", "sd::ops::zeros_as",
            "sd::ops::ones_as", "sd::ops::square", "sd::ops::zeta", "sd::ops::polygamma", "sd::ops::lgamma",
            "sd::ops::digamma", "sd::ops::fill", "sd::ops::split_v", "sd::ops::split", "sd::ops::adjust_hue",
            "sd::ops::adjust_saturation", "sd::ops::adjust_contrast", "sd::ops::adjust_contrast_v2",
            "sd::ops::depth_to_space", "sd::ops::space_to_depth", "sd::ops::cross", "sd::ops::space_to_batch",
            "sd::ops::space_to_batch_nd", "sd::ops::batch_to_space", "sd::ops::batch_to_space_nd", "sd::ops::top_k",
            "sd::ops::in_top_k", "sd::ops::moments", "sd::ops::embedding_lookup", "sd::ops::dynamic_partition",
            "sd::ops::dynamic_partition_bp", "sd::ops::dynamic_stitch", "sd::ops::zero_fraction", "sd::ops::xw_plus_b",
            "sd::ops::xw_plus_b_bp", "sd::ops::stop_gradient", "sd::ops::parallel_stack", "sd::ops::normalize_moments",
            "sd::ops::sufficient_statistics", "sd::ops::weighted_cross_entropy_with_logits", "sd::ops::dropout",
            "sd::ops::dropout_bp", "sd::ops::alpha_dropout_bp", "sd::ops::bincount", "sd::ops::broadcast_dynamic_shape",
            "sd::ops::matrix_determinant", "sd::ops::log_matrix_determinant", "sd::ops::logdet", "sd::ops::lstsq",
            "sd::ops::solve_ls", "sd::ops::matrix_inverse", "sd::ops::triangular_solve", "sd::ops::solve", "sd::ops::lu",
            "sd::ops::sequence_mask", "sd::ops::segment_max", "sd::ops::segment_max_bp", "sd::ops::segment_min",
            "sd::ops::segment_min_bp", "sd::ops::segment_sum", "sd::ops::segment_sum_bp", "sd::ops::segment_prod",
            "sd::ops::segment_prod_bp", "sd::ops::segment_mean", "sd::ops::segment_mean_bp",
            "sd::ops::unsorted_segment_max", "sd::ops::unsorted_segment_max_bp", "sd::ops::unsorted_segment_min",
            "sd::ops::unsorted_segment_min_bp", "sd::ops::unsorted_segment_sum", "sd::ops::unsorted_segment_sum_bp",
            "sd::ops::unsorted_segment_prod", "sd::ops::unsorted_segment_prod_bp", "sd::ops::unsorted_segment_mean",
            "sd::ops::unsorted_segment_mean_bp", "sd::ops::unsorted_segment_sqrt_n", "sd::ops::unsorted_segment_sqrt_n_bp",
            "sd::ops::extract_image_patches", "sd::ops::draw_bounding_boxes", "sd::ops::roll", "sd::ops::lin_space",
            "sd::ops::reduce_sum", "sd::ops::reduce_sum_bp", "sd::ops::reduce_prod", "sd::ops::reduce_prod_bp",
            "sd::ops::reduce_min", "sd::ops::reduce_min_bp", "sd::ops::reduce_max", "sd::ops::reduce_max_bp",
            "sd::ops::reduce_norm1", "sd::ops::reduce_norm1_bp", "sd::ops::reduce_norm2", "sd::ops::reduce_norm2_bp",
            "sd::ops::reduce_sqnorm", "sd::ops::reduce_sqnorm_bp", "sd::ops::reduce_norm_max",
            "sd::ops::reduce_norm_max_bp", "sd::ops::reduce_mean", "sd::ops::reduce_mean_bp", "sd::ops::reduce_variance",
            "sd::ops::reduce_variance_bp", "sd::ops::reduce_stdev", "sd::ops::reduce_stdev_bp", "sd::ops::reduce_dot_bp",
            "sd::ops::reduce_logsumexp", "sd::ops::matrix_band_part", "sd::ops::Assert", "sd::ops::non_max_suppression",
            "sd::ops::non_max_suppression_v3", "sd::ops::non_max_suppression_overlaps", "sd::ops::cholesky",
            "sd::ops::nth_element", "sd::ops::check_numerics", "sd::ops::fake_quant_with_min_max_vars",
            "sd::ops::fake_quant_with_min_max_vars_per_channel", "sd::ops::compare_and_bitpack", "sd::ops::eig",
            "sd::ops::permute", "sd::ops::reshapeas", "sd::ops::linear_copy", "sd::ops::transpose", "sd::ops::shape_of",
            "sd::ops::shapes_of", "sd::ops::set_shape",
            "sd::ops::squeeze", "sd::ops::expand_dims", "sd::ops::flatten_2d",
            "sd::ops::reshape", "sd::ops::reshape_no_copy", "sd::ops::size_at", "sd::ops::order",
            "sd::ops::tile_to_shape", "sd::ops::tile_to_shape_bp", "sd::ops::broadcast_to",
            "sd::ops::evaluate_reduction_shape", "sd::ops::create", "sd::ops::set_seed", "sd::ops::get_seed",
            "sd::ops::randomuniform", "sd::ops::random_multinomial", "sd::ops::random_normal",
            "sd::ops::random_bernoulli", "sd::ops::random_exponential", "sd::ops::random_crop",
            "sd::ops::random_gamma", "sd::ops::random_poisson", "sd::ops::softmax", "sd::ops::softmax_bp",
            "sd::ops::lrn", "sd::ops::lrn_bp", "sd::ops::batchnorm", "sd::ops::batchnorm_bp",
            "sd::ops::apply_sgd", "sd::ops::fused_batch_norm", "sd::ops::log_softmax", "sd::ops::log_softmax_bp",
            "sd::ops::relu_layer", "sd::ops::layer_norm", "sd::ops::layer_norm_bp",
            "sd::ops::dot_product_attention", "sd::ops::dot_product_attention_bp",
            "sd::ops::dot_product_attention_v2", "sd::ops::dot_product_attention_v2_bp",
            "sd::ops::multi_head_dot_product_attention", "sd::ops::multi_head_dot_product_attention_bp",
            "sd::ops::matmul", "sd::ops::matmul_bp", "sd::ops::tensormmul", "sd::ops::tensormmul_bp",
            "sd::ops::axpy", "sd::ops::batched_gemm", "sd::ops::batched_gemm_bp", "sd::ops::svd",
            "sd::ops::sqrtm", "sd::ops::test_output_reshape", "sd::ops::test_scalar", "sd::ops::testreduction",
            "sd::ops::noop", "sd::ops::testop2i2o", "sd::ops::testcustom", "sd::ops::toggle_bits",
            "sd::ops::shift_bits", "sd::ops::rshift_bits", "sd::ops::cyclic_shift_bits",
            "sd::ops::cyclic_rshift_bits", "sd::ops::bitwise_and", "sd::ops::bitwise_or", "sd::ops::bitwise_xor",
            "sd::ops::bits_hamming_distance", "sd::ops::hinge_loss", "sd::ops::hinge_loss_grad",
            "sd::ops::huber_loss", "sd::ops::huber_loss_grad", "sd::ops::log_loss", "sd::ops::log_loss_grad",
            "sd::ops::l2_loss", "sd::ops::log_poisson_loss", "sd::ops::log_poisson_loss_grad",
            "sd::ops::mean_pairwssqerr_loss", "sd::ops::mean_pairwssqerr_loss_grad", "sd::ops::mean_sqerr_loss",
            "sd::ops::mean_sqerr_loss_grad", "sd::ops::sigm_cross_entropy_loss",
            "sd::ops::sigm_cross_entropy_loss_grad", "sd::ops::softmax_cross_entropy_loss",
            "sd::ops::softmax_cross_entropy_loss_grad", "sd::ops::absolute_difference_loss",
            "sd::ops::absolute_difference_loss_grad", "sd::ops::cosine_distance_loss",
            "sd::ops::cosine_distance_loss_grad", "sd::ops::softmax_cross_entropy_loss_with_logits",
            "sd::ops::softmax_cross_entropy_loss_with_logits_grad",
            "sd::ops::sparse_softmax_cross_entropy_loss_with_logits",
            "sd::ops::sparse_softmax_cross_entropy_loss_with_logits_grad", "sd::ops::ctc_loss",
            "sd::ops::ctc_loss_grad", "sd::ops::to_double", "sd::ops::to_float16", "sd::ops::to_float32",
            "sd::ops::to_int32", "sd::ops::to_int64", "sd::ops::to_uint32", "sd::ops::to_uint64",
            "sd::ops::cast", "sd::ops::min_max_datatype", "sd::ops::bitcast"
    };


    private static final String[] SHAPE_FUNCTIONS = {
            "ShapeInformation", "isViewConst", "isEmptyConst", "isView", "isEmpty",
            "detachShape", "copyShape", "shapeEquals", "strideEquals", "equalsSoft",
            "equalsTypesAndShapesSoft", "equalsStrict", "haveSameShapeAndStrides",
            "tadIndexForLinear", "tadLength", "tadElementWiseStride", "reshapeC",
            "shapeBuffer", "shapeBufferFortran", "updateStrides", "shapeCopy",
            "computeElementWiseStride", "permuteShapeBufferInPlace", "doPermuteShapeInfo",
            "createPermuteIndexes", "getOrder", "permute", "isVector", "isLikeVector",
            "isColumnVector", "isCommonVector", "isMatrix", "shapeOf", "slice",
            "sliceOfShapeBuffer", "slices", "shapeInfoLength", "shapeInfoByteLength",
            "index2coords", "subArrayIndex", "outerArrayOffsets", "calcOffsets",
            "shapeOldScalar", "checkStridesEwsAndOrder", "calcSubArrsShapeInfoAndOffsets",
            "calcSubArrShapeInfoAndOffset", "excludeUnitiesFromShapeInfo",
            "calcStridesFortran", "calcStrides", "subArrayOffset", "tadOffset",
            "doPermuteSwap", "stride", "extra", "sizeAt", "setStrideConst", "setOrder",
            "setElementWiseStride", "setExtra", "order", "type", "elementWiseStride",
            "reductionIndexElementWiseStride", "everyIndexBut", "ensureVectorShape",
            "createScalarShapeInfo", "keep", "lengthPerSlice", "sliceOffsetForTensor",
            "tadForBlockIndex", "tadsPerBlock", "tadIndex", "reductionIndexForTad",
            "tadsPerReduceIndex", "reductionIndexForLinear", "prodLong", "getOffset",
            "getOffsetBroadcast", "index2coordsCPU", "coords2index", "getIndexOffset",
            "printShapeInfo", "printShapeInfoLinear", "printIntArray", "printArray",
            "maxIndToMinInd", "rearMostLeftOverItem", "shapeToString", "setStride",
            "fillStrides", "permuteShapeBuffer", "oneDimEqualToLength", "isRowVector",
            "numOfNonUnitDims", "rank", "ews", "length", "strideAt", "setShape",
            "isScalar", "indexOffset", "sweepShapeInfoBuffer", "tensorsAlongDimension",
            "shapeInfoString", "shapeBufferOfNpy"
    };




    public static void processOps(Logger logger, java.util.Properties properties, InfoMap infoMap) {
        boolean funcTrace = System.getProperty("libnd4j.calltrace", "OFF").equalsIgnoreCase("ON");
        System.out.println("Func trace: " + funcTrace);
        processCustomOps(logger, properties, infoMap);
    }

    public static String[] getCommonMacros() {
        return new String[]{
                "thread_local", "SD_LIB_EXPORT", "SD_INLINE", "CUBLASWINAPI",
                "SD_HOST", "SD_DEVICE", "SD_KERNEL", "SD_HOST_DEVICE", "SD_ALL_OPS", "NOT_EXCLUDED"
        };
    }

    public static String[] getSkipHeaders() {
        return new String[]{
                "openblas_config.h", "cblas.h", "lapacke_config.h", "lapacke_mangling.h",
                "lapack.h", "lapacke.h", "lapacke_utils.h"
        };
    }

    public static String[] getObjectifyHeaders() {
        return new String[]{"NativeOps.h", "build_info.h"};
    }

    public static String[] getOpaqueTypes() {
        return new String[]{
                "NDArray", "TadPack", "ResultWrapper", "ShapeList", "VariablesSet",
                "Variable", "ConstantDataBuffer", "ConstantShapeBuffer", "ConstantOffsetsBuffer",
                "DataBuffer", "Context", "RandomGenerator", "LaunchContext"
        };
    }

    public static String[] getSkipClasses() {
        return new String[]{
                "sd::graph::FlowPath", "sd::graph::Graph", "sd::ConstantDataBuffer", "sd::ConstantHolder",
                "sd::ConstantOffsetsBuffer", "sd::ShapeList", "sd::graph::GraphProfile", "sd::LaunchContext",
                "sd::TadDescriptor", "sd::ShapeDescriptor", "sd::TadPack", "sd::NDIndex", "sd::NDIndexPoint",
                "sd::NDIndexAll", "sd::NDIndexInterval", "sd::IndicesList", "sd::ArgumentsList",
                "sd::graph::GraphState", "sd::graph::VariableSpace"
        };
    }

    public static String[] getExcludedFunctions() {
        return new String[]{
                "std::initializer_list", "cnpy::NpyArray", "sd::NDArray::applyLambda",
                "sd::NDArray::applyPairwiseLambda", "sd::graph::FlatResult", "sd::graph::FlatVariable",
                "sd::NDArray::subarray", "std::shared_ptr", "sd::PointerWrapper", "sd::PointerDeallocator",
                "instrumentFile", "setInstrumentOut", "closeInstrumentOut", "__cyg_profile_func_exit",
                "__cyg_profile_func_enter"
        };
    }


    public static String[] getPrefixedShapeFunctions() {
        return Arrays.stream(SHAPE_FUNCTIONS)
                .map(func -> "shape::" + func)
                .toArray(String[]::new);
    }

    public static String[] getPrefixedSkipOps() {
        return Arrays.stream(SKIP_OPS)
                .map(op -> "sd::ops::" + op)
                .toArray(String[]::new);
    }




    private static void processCustomOps(Logger logger, java.util.Properties properties, InfoMap infoMap) {
        String separator = properties.getProperty("platform.path.separator");
        String[] includePaths = properties.getProperty("platform.includepath").split(separator);
        File file = null;
        File opFile = null;
        boolean foundCustom = false;
        boolean foundOps = false;
        for (String path : includePaths) {
            if(!foundCustom) {
                file = new File(path, "ops/declarable/CustomOperations.h");
                if (file.exists()) {
                    foundCustom = true;
                }
            }

            if(!foundOps) {
                opFile = new File(path, "generated/include_ops.h");
                if (opFile.exists()) {
                    foundOps = true;
                }
            }

            if(foundCustom && foundOps) {
                break;
            }
        }

        boolean allOps = false;
        Set<String> opsToExclude = new HashSet<>();
        try (Scanner scanner = new Scanner(opFile, "UTF-8")) {
            while (scanner.hasNextLine()) {
                String line = scanner.nextLine().trim();
                if(line.contains("#ifndef") || line.contains("#endif")) {
                    break;
                }

                if(line.contains("SD_ALL_OPS")) {
                    allOps = true;
                    System.out.println("All ops found.");
                    break;
                }

                String[] lineSplit = line.split(" ");
                if(lineSplit.length < 2) {
                    System.err.println("Unable to add op to exclude. Invalid op found: " + line);
                } else {
                    String opName = lineSplit[1].replace("OP_","");
                    opsToExclude.add(opName);
                    //usually gradient ops are co located in the same block
                    opsToExclude.add(opName + "_bp");
                }
            }
        } catch (IOException e) {
            throw new RuntimeException("Could not parse CustomOperations.h and headers", e);
        }

        if(opsToExclude.isEmpty()) {
            System.out.println("No ops found for exclusion setting all ops to true");
            allOps = true;
        }

        List<File> files = new ArrayList<>();
        List<String> opTemplates = new ArrayList<>();
        if(file == null) {
            throw new IllegalStateException("No file found in include paths. Please ensure one of the include paths leads to path/ops/declarable/CustomOperations.h");
        }
        files.add(file);
        File[] headers = new File(file.getParent(), "headers").listFiles();
        if(headers == null) {
            throw new IllegalStateException("No headers found for file " + file.getAbsolutePath());
        }

        files.addAll(Arrays.asList(headers));
        Collections.sort(files);

        for (File f : files) {
            try (Scanner scanner = new Scanner(f, "UTF-8")) {
                while (scanner.hasNextLine()) {
                    String line = scanner.nextLine().trim();
                    if (line.startsWith("DECLARE_")) {
                        try {
                            int start = line.indexOf('(') + 1;
                            int end = line.indexOf(',');
                            if (end < start) {
                                end = line.indexOf(')');
                            }
                            String name = line.substring(start, end).trim();
                            opTemplates.add(name);
                        } catch(Exception e) {
                            throw new RuntimeException("Could not parse line from CustomOperations.h and headers: \"" + line + "\"", e);
                        }
                    }
                }
            } catch (IOException e) {
                throw new RuntimeException("Could not parse CustomOperations.h and headers", e);
            }
        }

        Collections.sort(opTemplates);
        logger.info("Ops found in CustomOperations.h and headers: " + opTemplates);
        //we will be excluding some ops based on the ops defined in the generated op inclusion file
        if(!allOps) {
            logger.info("Found ops to only include " + opsToExclude);
            for(String op : opTemplates)
                if(!opsToExclude.contains(op)) {
                    logger.info("Excluding op " + op);
                    infoMap.put(new Info("NOT_EXCLUDED(OP_" + op + ")")
                            .skip(true)
                            .define(false));
                } else {
                    logger.info("Including " + op);
                    infoMap.put(new Info("NOT_EXCLUDED(OP_" + op + ")").define(true));
                    infoMap.put(new Info("NOT_EXCLUDED(OP_" + op + "_bp)").define(true));
                }
        }
    }
}