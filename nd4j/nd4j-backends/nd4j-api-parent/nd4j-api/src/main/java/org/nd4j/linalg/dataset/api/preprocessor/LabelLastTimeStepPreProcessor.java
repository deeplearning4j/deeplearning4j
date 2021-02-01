/*
 *  ******************************************************************************
 *  *
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
 *  * Unless required by applicable law or agreed to in writing, software
 *  * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 *  * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 *  * License for the specific language governing permissions and limitations
 *  * under the License.
 *  *
 *  * SPDX-License-Identifier: Apache-2.0
 *  *****************************************************************************
 */

package org.nd4j.linalg.dataset.api.preprocessor;

import org.nd4j.common.base.Preconditions;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.DataSet;
import org.nd4j.linalg.dataset.api.DataSetPreProcessor;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.BooleanIndexing;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.nd4j.linalg.indexing.conditions.Conditions;

/**
 * Used to extract the labels from a 3d format (shape: [minibatch, nOut, sequenceLength]) to a 2d format (shape: [minibatch, nOut])
 * where the values are the last time step of the labels.<br>
 * <br>
 * For example, for 2 sequences:<br>
 * [a, b, c, 0, 0]<br>
 * [p, q, r, s, t]<br>
 * (where a/b/p etc represet a vector of size numOutputs), and each row is the sequence for each <br
 * with label mask:<br>
 * [1, 1, 1, 0, 0]<br>
 * [1, 1, 1, 1, 1]<br>
 * The new labels would be a rank 2 array of shape [minibatch, nOut] with values:<br>
 * [c]<br>
 * [t]<br>
 * <br>
 * This preprocessor can be used for example to convert from "single non-masked time step" labels format (produced by
 * RecordReaderDataSetIterator, used in RnnOutputLayer) to 2d labels format (used in OutputLayer).
 *
 * @author Alex Black
 */
public class LabelLastTimeStepPreProcessor implements DataSetPreProcessor {
    @Override
    public void preProcess(DataSet toPreProcess) {

        INDArray label3d = toPreProcess.getLabels();
        Preconditions.checkState(label3d.rank() == 3, "LabelLastTimeStepPreProcessor expects rank 3 labels, got rank %s labels with shape %ndShape", label3d.rank(), label3d);

        INDArray lMask = toPreProcess.getLabelsMaskArray();
        //If no mask: assume that examples for each minibatch are all same length
        INDArray labels2d;
        if(lMask == null){
            labels2d = label3d.get(NDArrayIndex.all(), NDArrayIndex.all(), NDArrayIndex.point(label3d.size(2)-1)).dup();
        } else {
            //Use the label mask to work out the last time step...
            INDArray lastIndex = BooleanIndexing.lastIndex(lMask, Conditions.greaterThan(0), 1);
            long[] idxs = lastIndex.data().asLong();

            //Now, extract out:
            labels2d = Nd4j.create(DataType.FLOAT, label3d.size(0), label3d.size(1));

            //Now, get and assign the corresponding subsets of 3d activations:
            for (int i = 0; i < idxs.length; i++) {
                long lastStepIdx = idxs[i];
                Preconditions.checkState(lastStepIdx >= 0, "Invalid last time step index: example %s in minibatch is entirely masked out" +
                        " (label mask is all 0s, meaning no label data is present for this example)", i);
                //TODO can optimize using reshape + pullRows
                labels2d.putRow(i, label3d.get(NDArrayIndex.point(i), NDArrayIndex.all(), NDArrayIndex.point(lastStepIdx)));
            }
        }

        toPreProcess.setLabels(labels2d);
        toPreProcess.setLabelsMaskArray(null);  //Remove label mask if present
    }
}
