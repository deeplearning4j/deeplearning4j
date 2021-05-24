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
// @author AbdelRauf
//

#ifndef LIBND4J_HEADERS_DECODER_H
#define LIBND4J_HEADERS_DECODER_H

#include <ops/declarable/headers/common.h>

namespace sd {
namespace ops {

      /**
       * Implementation of CTC beam search
       *
       * Input arrays:
       *    0: logits - logits NDArray logit NDArray {BATCH_LEN, MAX_FRAME_LEN, CLASS_LEN }. It should include a blank label as well. type float
       *    1: sequence_length -  NDArray {BATCH_LEN} length of frames. type integer
       *
       * Input integer arguments (IArgs):
       *    0: blank_index the index of the blank label in logits. default is last class. CLASS_LEN-1
       *    1: beam_width  the width of the beam search. default is 25
       *    2: nbest_len  the number of top best results that should be returned. default is 1
       *    NOTE:  if it is > beam_width it will be defaulted to beam_width size.
       * Input bool argument (BArgs):
       *    0: normalize_logit when its true it will normalize logits. by default it is assumed logit contains already normalized log-probabilities
       * Output array:
       *    0: result_sequences NDArray {BATCH_LEN, NBEST, MAX_FRAME_LEN} result sequences.
       *    NOTE: result_sequences NdArray should be c order and have ews == 1. type integer
       *    1: result_probs NDArray {BATCH_LEN, NBEST} negative log probabilities for each sequence. type float
       *    NOTE: result_probs NdArray should be c order and have ews == 1
       *    2: result_sequence_length NDArray {BATCH_LEN, NBEST} the length of the each sequence. type integer
       *    NOTE: result_sequence_length NdArray should be c order and have ews == 1
       * 
       *  NOTE:
       *   maximum value of integer indexing type should be >= CLASS_LEN to make sense. And also it should consider frame lengthes as well.
       *   For now this case is mostly fine as only Indexing types are allowed as integer. 
       */
        #if NOT_EXCLUDED(OP_ctc_beam)
        DECLARE_CUSTOM_OP(ctc_beam, 2, 3, false, 0, -2); 
        #endif


}
}

#endif
