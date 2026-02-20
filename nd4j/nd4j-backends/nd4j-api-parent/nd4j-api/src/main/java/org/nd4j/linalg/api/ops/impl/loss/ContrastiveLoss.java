/*
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

package org.nd4j.linalg.api.ops.impl.loss;

import lombok.Getter;
import lombok.NoArgsConstructor;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.DynamicCustomOp;

import org.nd4j.linalg.api.ops.impl.loss.bp.ContrastiveLossBp;

import java.util.Arrays;
import java.util.Collections;
import java.util.List;

/**
 * InfoNCE Contrastive Loss for CLIP-style contrastive alignment.
 *
 * Given L2-normalized image and text embeddings, computes the symmetric
 * cross-entropy over the cosine similarity matrix:
 *   similarity = image_embeds @ text_embeds^T * temperature
 *   loss = (CE_rows + CE_cols) / (2 * batch)
 *
 * Inputs:
 *   0: imageEmbeddings [batch, embedDim] - L2-normalized
 *   1: textEmbeddings  [batch, embedDim] - L2-normalized
 *
 * Float args:
 *   0: temperature (default: 1.0)
 *
 * Output:
 *   0: scalar loss value
 *
 * @author Eclipse Deeplearning4j Contributors
 */
@NoArgsConstructor
public class ContrastiveLoss extends DynamicCustomOp {

    @Getter private double temperature = 1.0;

    public ContrastiveLoss(INDArray imageEmbeddings, INDArray textEmbeddings) {
        this(imageEmbeddings, textEmbeddings, 1.0);
    }

    public ContrastiveLoss(INDArray imageEmbeddings, INDArray textEmbeddings, double temperature) {
        super(new INDArray[]{imageEmbeddings, textEmbeddings}, null);
        this.temperature = temperature;
        addTArgument(temperature);
    }

    public ContrastiveLoss(SameDiff sameDiff, SDVariable imageEmbeddings, SDVariable textEmbeddings) {
        this(sameDiff, imageEmbeddings, textEmbeddings, 1.0);
    }

    public ContrastiveLoss(SameDiff sameDiff, SDVariable imageEmbeddings, SDVariable textEmbeddings, double temperature) {
        super(null, sameDiff, new SDVariable[]{imageEmbeddings, textEmbeddings}, false);
        this.temperature = temperature;
        addTArgument(temperature);
    }

    @Override
    public void configureFromArguments() {
        super.configureFromArguments();
        if (tArguments.size() > 0) this.temperature = tArguments.get(0);
    }

    @Override
    public String opName() {
        return "contrastive_loss";
    }

    @Override
    public List<SDVariable> doDiff(List<SDVariable> grad) {
        return Arrays.asList(new ContrastiveLossBp(sameDiff, arg(0), arg(1), temperature).outputVariables());
    }

    @Override
    public List<DataType> calculateOutputDataTypes(List<DataType> inputDataTypes) {
        return Collections.singletonList(inputDataTypes.get(0));
    }

    @Override
    public int getNumOutputs() {
        return 1;
    }
}
