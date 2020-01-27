/*******************************************************************************
 * Copyright (c) 2015-2019 Skymind, Inc.
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

package org.deeplearning4j.rl4j.observation;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.DataSet;
import org.nd4j.linalg.factory.Nd4j;

/**
 * Presently only a dummy container. Will contain observation channels when done.
 */
public class Observation {
    // TODO: Presently only a dummy container. Will contain observation channels when done.

    private final DataSet data;

    public Observation(INDArray[] data) {
        this(data, false);
    }

    public Observation(INDArray[] data, boolean shouldReshape) {
        INDArray features = Nd4j.concat(0, data);
        if(shouldReshape) {
            features = reshape(features);
        }
        this.data = new org.nd4j.linalg.dataset.DataSet(features, null);
    }

    // FIXME: Remove -- only used in unit tests
    public Observation(INDArray data) {
        this.data = new org.nd4j.linalg.dataset.DataSet(data, null);
    }

    private INDArray reshape(INDArray source) {
        long[] shape = source.shape();
        long[] nshape = new long[shape.length + 1];
        nshape[0] = 1;
        System.arraycopy(shape, 0, nshape, 1, shape.length);

        return source.reshape(nshape);
    }

    private Observation(DataSet data) {
        this.data = data;
    }

    public Observation dup() {
        return new Observation(new org.nd4j.linalg.dataset.DataSet(data.getFeatures().dup(), null));
    }

    public INDArray getData() {
        return data.getFeatures();
    }
}
