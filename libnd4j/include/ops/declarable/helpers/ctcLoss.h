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

namespace sd    {
namespace ops     {
namespace helpers {

	/**
	 * @brief Implementation of CTC loss function
	 *  References:
        Connectionist Temporal Classification - Labeling Unsegmented Sequence Data
        with Recurrent Neural Networks:
        [Graves et al., 2006](https://dl.acm.org/citation.cfm?id=1143891)
        ([pdf](http://www.cs.toronto.edu/~graves/icml_2006.pdf))
	 *
	 * @param block Context
	 * @param logits NDArray {BATCH_LEN, FRAME_LEN, CLASS_LEN }. log softmax of  rnn output. It should include a blank label as well.
	 * @param targetLabels NDArray {BATCH_LEN, MAX_TARGET_LEN}
	 * @param logitsLengths NDArray {BATCH_LEN} Length of input sequence in logits
	 * @param targetLabelLengths NDArray {BATCH_LEN} Length of label sequence in labels
	 * @param logLosses NDArray {BATCH_LEN} or EMPTY. if empty it will be skipped. negative log probabilities of loss
	 * @param gradients NDArray {BATCH_LEN, FRAME_LEN, CLASS_LEN } or EMPTY. gradients
	 * @param blankIndex index of the blank label in logits
	 */
	void ctcLoss(graph::Context& block, const NDArray &logitsInput, const NDArray &targetLabels, const NDArray &logitsLengths, const NDArray &targetLabelLengths, NDArray &logLosses, NDArray &gradients, int blankIndex);

}
}
}


#endif // LIBND4J_ADDBIAS_H