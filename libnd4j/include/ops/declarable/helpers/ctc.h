/*******************************************************************************
 * Copyright (c) 2021 Deeplearning4j Contributors
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
 *******************************************************************************/

//
// @author AbdelRauf
//

#ifndef LIBND4J_HELPERS_CTCLOSS_H
#define LIBND4J_HELPERS_CTCLOSS_H

#include <ops/declarable/helpers/helpers.h>
#include <graph/Context.h>
#include <type_traits>
#include <math/platformmath.h>
namespace sd    {
namespace ops     {
namespace helpers {

    //#define LOGIT_SOFTMAX_NORMALIZATION 1

    template <typename T>
    constexpr T negative_infinity()
    {
        return -DataTypeUtils::infOrMax<T>();
    }

    //choose ptr[index*element_stride]
    template <bool HasStride, typename Type>
    typename std::enable_if<HasStride == true, Type &>::type
    element(Type *ptr, int index, int element_stride)
    {
        return ptr[index * element_stride];
    }

    //choose ptr[index] assuming element_stride is 1
    template <bool HasStride, typename Type>
    typename std::enable_if<HasStride == false, Type &>::type
    element(Type *ptr, int index, int element_stride)
    {
        return ptr[index];
    }

    template <typename T>
    T local_log(T x)
    {
        if (x > 0)
        {
            return (sd::math::p_log<T>(x));
        }
        return (negative_infinity<T>());
    }

    template <typename T>
    T log_sum_exp(T x1, T x2)
    {
        //substituting this : std::log(std::exp(arg1 - cMax) + std::exp(arg2 - cMax)) + cMax
        //if arg1==cMax : std::log(1 + std::exp(arg2 - cMax)) + cMax
        if (x1 >= x2)
        {
            //x1 is max
            return (x1 + local_log(1 + sd::math::p_exp<T>(x2 - x1)));
        }
        //x2 is max
        return (x2 + local_log(1 + sd::math::p_exp<T>(x1 - x2)));
    }

    template <typename T>
    T log_sum_exp(T arg1, T arg2, T arg3)
    {
        auto c_max = std::max(arg1, arg2);
        c_max = std::max(c_max, arg3);
        if (negative_infinity<T>() == c_max)
        {
            c_max = 0;
        }
        return sd::math::p_log(sd::math::p_exp(arg1 - c_max) + sd::math::p_exp(arg2 - c_max) + sd::math::p_exp(arg3 - c_max)) + c_max;
    }

    template <bool HasElementStride, typename Type, typename IndexType>
    Type softmax_normalization_term(const Type* log_p, const uint64_t len_c, const uint64_t element_stride)
    {
        Type max_p;
        for (auto c = 0; c < len_c; ++c) {
            max_p = std::max(max_p, element<HasElementStride>(log_p, c, element_stride));
        }
        // Get normalization term of softmax: log(sum(exp(logit[j]-max_p))).
        Type logsumexp = Type(0.0);
        for (auto c = 0; c < len_c; ++c) {
            logsumexp += sd::math::p_exp(element<HasElementStride>(log_p, c, element_stride) - max_p);
        }
        logsumexp = sd::math::p_log(logsumexp);
        return max_p + logsumexp;
    }

    /**
     * @brief Implementation of CTC loss function
     *  References:
        Connectionist Temporal Classification - Labeling Unsegmented Sequence Data
        with Recurrent Neural Networks:
        [Graves et al., 2006](https://dl.acm.org/citation.cfm?id=1143891)
        ([pdf](http://www.cs.toronto.edu/~graves/icml_2006.pdf))
     *
     * @param block Context
     * @param logits NDArray {BATCH_LEN, MAX_FRAME_LEN, CLASS_LEN }. It should include a blank label as well.
     * NOTE: log softmax of rnn output. so we expect softmax normalized 
     * @param targetLabels NDArray {BATCH_LEN, MAX_TARGET_LEN}
     * @param logitsLengths NDArray {BATCH_LEN} Length of input sequence in logits
     * @param targetLabelLengths NDArray {BATCH_LEN} Length of label sequence in labels
     * @param logLosses NDArray {BATCH_LEN} or EMPTY. if empty it will be skipped. negative log probabilities of loss
     * @param gradients NDArray {BATCH_LEN, MAX_FRAME_LEN, CLASS_LEN } or EMPTY. gradients
     * @param blankIndex index of the blank label in logits
     */
    void ctcLoss(graph::Context& block, const NDArray &logitsInput, const NDArray &targetLabels, const NDArray &logitsLengths, const NDArray &targetLabelLengths, NDArray &logLosses, NDArray &gradients, int blankIndex);


    /**
     * @brief Implementation of CTC beam search
     *
     * @param logit NDArray {BATCH_LEN, MAX_FRAME_LEN, CLASS_LEN }. log probabilities. It should include a blank label as well.
     * @param sequence_length NDArray {BATCH_LEN} length of frames. type integer
     * @param result_sequences NDArray {BATCH_LEN, NBEST, MAX_FRAME_LEN} result sequences.
     *  NOTE: result_sequences NdArray should be c order and have ews == 1. type integer.
     * @param result_probs NDArray {BATCH_LEN, NBEST} negative log probabilities for each sequence. 
     *  NOTE: result_probs NdArray should be c order and have ews == 1
     * @param result_sequences_length NDArray {BATCH_LEN, NBEST} the length of each sequence in result_sequences. 
     *  NOTE: result_sequences_length NdArray should be c order and have ews == 1
     * @param blank_index the index of the blank label in logits
     * @param beam_width  the width of the beam search.
     * @param nbest_len the number of top best results that should be returned.  if it is greather than beam_width it will be defaulted to beam_width size.
     * @param normalize_logits when its true it will normalize logits. by default it is assumed logit contains already normalized log-probabilities
     * NOTE:
     * maximum value of integer type  should be >= CLASS_LEN to make sense. And also user should consider frame lengthes as well.
     */
void beamSearch(const NDArray& logit, const NDArray& sequence_length, NDArray& result_sequences, NDArray& result_probs, NDArray& result_sequences_length, int blank_index, int beam_width , int nbest_len, bool normalize_logits);
}
}
}


#endif
