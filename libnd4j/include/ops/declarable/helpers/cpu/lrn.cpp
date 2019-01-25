/*******************************************************************************
 * Copyright (c) 2015-2018 Skymind, Inc.
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

//
//  @author raver119@gmail.com
//

#include <ops/declarable/helpers/lrn.h>
#include <Status.h>

namespace nd4j {
namespace ops {
namespace helpers {

#ifdef HAVE_MKLDNN
using namespace mkldnn;

static void getMKLDNNMemoryDescLrn(const NDArray* src, const NDArray* diff_src,
        mkldnn::memory::desc* lrn_src_md, mkldnn::memory::desc* lrn_diff_src_md, int axis) {
    const Nd4jLong* shape = src->getShapeInfo();
    long rank = shape[0];
    long dim1 = axis; // MKL-DNN supports only 1 axis, which has to be the "channel" one
    long dim2 = axis >= 2 ? 1 : 2;
    long dim3 = axis >= 3 ? 2 : 3;
    mkldnn::memory::dims lrn_src_tz = { (int)shape[1], (int)shape[dim1 + 1], rank > 2 ? (int)shape[dim2 + 1] : 1, rank > 3 ? (int)shape[dim3 + 1] : 1};

    auto type = mkldnn::memory::data_type::f32;
    auto format = axis == 1 ? mkldnn::memory::format::nchw : mkldnn::memory::format::nhwc;

    if (src != nullptr && src->getBuffer() != nullptr && lrn_src_md != nullptr) {
        *lrn_src_md = mkldnn::memory::desc({ lrn_src_tz }, type, format);
        // lrn_src_md->data.format = mkldnn_blocked; // unsupported for lrn, leave nchw or nhwc for now
        lrn_src_md->data.layout_desc.blocking.strides[0][0] = src->stridesOf()[0];
        lrn_src_md->data.layout_desc.blocking.strides[0][1] = src->stridesOf()[dim1];
        lrn_src_md->data.layout_desc.blocking.strides[0][2] = rank > 2 ? src->stridesOf()[dim2] : 1;
        lrn_src_md->data.layout_desc.blocking.strides[0][3] = rank > 3 ? src->stridesOf()[dim3] : 1;
    }

    if (diff_src != nullptr && diff_src->getBuffer() != nullptr && lrn_diff_src_md != nullptr) {
        *lrn_diff_src_md = mkldnn::memory::desc({ lrn_src_tz }, type, format);
        // lrn_diff_src_md->data.format = mkldnn_blocked; // unsupported for lrn, leave nchw or nhwc for now
        lrn_diff_src_md->data.layout_desc.blocking.strides[0][0] = diff_src->stridesOf()[0];
        lrn_diff_src_md->data.layout_desc.blocking.strides[0][1] = diff_src->stridesOf()[dim1];
        lrn_diff_src_md->data.layout_desc.blocking.strides[0][2] = rank > 2 ? diff_src->stridesOf()[dim2] : 1;
        lrn_diff_src_md->data.layout_desc.blocking.strides[0][3] = rank > 3 ? diff_src->stridesOf()[dim3] : 1;
    }
}
#endif

    // FIXME: double
    int lrnFunctor(nd4j::graph::Context& block, NDArray* input, NDArray* output, int depth, double bias, double alpha, double beta) {

        double dividor;

        int totalLength = input->lengthOf();
        int lastDim = input->sizeAt(-1);
        int chunkCount = totalLength / lastDim;

#ifdef HAVE_MKLDNN
    if (block.isUseMKLDNN() && nd4j::MKLDNNStream::isSupported({input, output})) {
        std::vector<nd4j::MKLDNNStream>& streams = block.getMKLDNNStreams();
        if (streams.empty()) {
            streams.push_back(MKLDNNStream("lrn"));
        }

        if (streams[0].checkAndReset({input}, {output}, {bias, alpha, beta}, {depth})) {
            mkldnn_memory_desc_t empty;
            mkldnn::memory::desc lrn_src_md(empty);

            getMKLDNNMemoryDescLrn(input, nullptr, &lrn_src_md, nullptr, input->rankOf() - 1);

            auto lrn_desc = lrn_forward::desc(prop_kind::forward_inference, lrn_across_channels, lrn_src_md, (2 * depth + 1), alpha * (2 * depth + 1), beta, bias);

            auto lrn_prim_desc = lrn_forward::primitive_desc(lrn_desc, streams[0].getEngine());
            auto lrn_src_memory = mkldnn::memory(lrn_prim_desc.src_primitive_desc(), input->buffer());
            auto lrn_dst_memory = mkldnn::memory(lrn_prim_desc.dst_primitive_desc(), output->buffer());
            streams[0].setMemory({lrn_src_memory, lrn_dst_memory});
            streams[0].setOperation(lrn_forward(lrn_prim_desc, lrn_src_memory, lrn_dst_memory));
        }

        streams[0].submitAndWait();
        return ND4J_STATUS_OK;
    }
#endif
    nd4j_debug("MKL-DNN is not used for lrn!\n", 0);

        std::unique_ptr<ResultSet> listOut(output->allTensorsAlongDimension({output->rankOf() - 1}));
        std::unique_ptr<ResultSet> listInput(input->allTensorsAlongDimension({input->rankOf() - 1}));
        if (chunkCount != listOut->size()) 
            return ND4J_STATUS_VALIDATION;
        for (int c = 0; c < chunkCount; c++) {
            for (int e = 0; e < lastDim; e++) {
                int begin = nd4j::math::nd4j_max(0, e - depth);
                int end = nd4j::math::nd4j_min(depth + e + 1, lastDim);
                double quadSum = 0;

                for (int pos = begin; pos < end; ++pos) {
                    double val = listInput->at(c)->e<double>(pos);
                    quadSum += val * val;
                }
                double dividor = nd4j::math::nd4j_pow<double, double, double>(bias + alpha * quadSum, beta);
                listOut->at(c)->p<double>(e,  listInput->at(c)->e<double>(e) / dividor);
            }
        }

        return Status::OK();
    }

    int lrnFunctorEx(nd4j::graph::Context& block, NDArray* input, NDArray* output, NDArray* unitScale, NDArray* scale, int depth, double bias, double alpha, double beta) {
    
        depth = nd4j::math::nd4j_min<Nd4jLong>(depth, input->sizeAt(1));

        int halfDepth = depth / 2;
        halfDepth = nd4j::math::nd4j_max(halfDepth, 0);
        const int channel =  input->sizeAt(1);

#ifdef HAVE_MKLDNN_DISABLED
//XXX: need to get output to match exactly with MKL-DNN
    if (block.isUseMKLDNN() && nd4j::MKLDNNStream::isSupported({input, scale, output})) {
        std::vector<nd4j::MKLDNNStream>& streams = block.getMKLDNNStreams();
        if (streams.empty()) {
            streams.push_back(MKLDNNStream("lrn_bp"));
        }

        if (streams[0].checkAndReset({input, scale}, {output}, {bias, alpha, beta}, {depth})) {
            mkldnn_memory_desc_t empty;
            mkldnn::memory::desc lrn_src_md(empty), lrn_diff_src_md(empty);

            getMKLDNNMemoryDescLrn(input, scale, &lrn_src_md, &lrn_diff_src_md, 1);

            auto lrn_desc = lrn_forward::desc(prop_kind::forward, lrn_across_channels, lrn_src_md, (2 * halfDepth + 1), alpha * (2 * halfDepth + 1), beta, bias);
            auto lrn_back_desc = lrn_backward::desc(lrn_across_channels, lrn_src_md, lrn_diff_src_md, (2 * halfDepth + 1), alpha * (2 * halfDepth + 1), beta, bias);

            auto lrn_prim_desc = lrn_forward::primitive_desc(lrn_desc, streams[0].getEngine());
            auto lrn_back_prim_desc = lrn_backward::primitive_desc(lrn_back_desc, streams[0].getEngine(), lrn_prim_desc);
            auto lrn_src_memory = mkldnn::memory(lrn_prim_desc.src_primitive_desc(), input->buffer());
            auto lrn_dst_memory = mkldnn::memory(lrn_back_prim_desc.diff_dst_primitive_desc(), scale->buffer());
            auto lrn_diff_src_memory = mkldnn::memory(lrn_back_prim_desc.diff_src_primitive_desc(), output->buffer());
            streams[0].setMemory({lrn_src_memory, lrn_dst_memory, lrn_diff_src_memory});
            streams[0].setOperation(lrn_backward(lrn_back_prim_desc, lrn_src_memory, lrn_dst_memory, lrn_diff_src_memory));
        }

        streams[0].submitAndWait();
        return ND4J_STATUS_OK;
    }
#endif
    nd4j_debug("MKL-DNN is not used for lrn_bp!\n", 0);

        std::unique_ptr<NDArray> activitySqr(input->dup('c'));//NDArrayFactory<T>::createUninitialized(input));
        std::unique_ptr<NDArray> sumPart(activitySqr->dup('c'));

        input->applyPairwiseTransform(pairwise::Multiply, input, activitySqr.get(), nullptr);
#pragma omp parallel for if (halfDepth + 1 > Environment::getInstance()->elementwiseThreshold()) schedule(static)         
        for (int i = 1; i < halfDepth + 1; i++) {
            IndicesList indA({NDIndex::all(), NDIndex::interval(i, channel), NDIndex::all(), NDIndex::all()});
            IndicesList indB({NDIndex::all(), NDIndex::interval(0, channel - i), NDIndex::all(), NDIndex::all()});

            std::unique_ptr<NDArray> tmp(sumPart->subarray(indA));
            std::unique_ptr<NDArray> addVal(activitySqr->subarray(indB));

            tmp->applyPairwiseTransform(pairwise::Add, *addVal.get(), nullptr);


            std::unique_ptr<NDArray> tmp2(sumPart->subarray(indB));
            std::unique_ptr<NDArray> addVal2(activitySqr->subarray(indA));

            tmp2->applyPairwiseTransform(pairwise::Add, *addVal2.get(), nullptr);
        }

        /*
         *  // taken from java
            unitScale = sumPart.mul(alpha).addi(k).leverageTo(ComputationGraph.workspaceExternal);
            // y = x * unitScale**-beta
            scale = Transforms.pow(unitScale, -beta).leverageTo(ComputationGraph.workspaceExternal);
            activations = input.mul(scale).leverageTo(ComputationGraph.workspaceExternal);
         */
        if (unitScale != nullptr && scale != nullptr) {
            sumPart->applyScalar(scalar::Multiply, alpha, unitScale, nullptr);
            unitScale->applyScalar(scalar::Add, bias);

            float p = static_cast<float>(-beta);
            unitScale->applyScalar(scalar::Pow, p, scale, nullptr);
            input->applyPairwiseTransform(pairwise::Multiply, scale, output, nullptr);
        }

        return Status::OK();
    }

}
}
}
