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

package org.deeplearning4j.optimize.listeners;

import org.deeplearning4j.nn.api.Model;
import org.deeplearning4j.optimize.api.BaseTrainingListener;
import org.deeplearning4j.optimize.api.TrainingListener;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;

/**
 * A group of listeners
 * @author Adam Gibson
 * @deprecated Not required - DL4J networks can use multiple listeners simultaneously
 */
@Deprecated
public class ComposableIterationListener extends BaseTrainingListener implements Serializable {
    private Collection<TrainingListener> listeners = new ArrayList<>();

    public ComposableIterationListener(TrainingListener... TrainingListener) {
        listeners.addAll(Arrays.asList(TrainingListener));
    }

    public ComposableIterationListener(Collection<TrainingListener> listeners) {
        this.listeners = listeners;
    }

    @Override
    public void iterationDone(Model model, int iteration, int epoch) {
        for (TrainingListener listener : listeners)
            listener.iterationDone(model, iteration, epoch);
    }
}
