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

package org.deeplearning4j.rl4j.observation.transforms;

import org.deeplearning4j.rl4j.observation.Observation;
import org.deeplearning4j.rl4j.observation.SimpleObservation;
import org.nd4j.base.Preconditions;

import java.util.ArrayList;
import java.util.List;

/**
 * The SignalingTransform will call its list of listeners when the transform is called
 *
 * @author Alexandre Boulanger
 */
public class SignalingTransform extends PassthroughTransform {

    private final List<TransformListener> listeners;

    public SignalingTransform() {
        super();
        listeners = new ArrayList<>();
    }

    private SignalingTransform(Builder builder) {
        listeners = builder.listeners;
    }

    /**
     * Adds a listener to the call list. This list will be called whenever transform() is called.
     * @param listener
     */
    public void addListener(TransformListener listener) {
        Preconditions.checkNotNull(listener);
        listeners.add(listener);
    }

    @Override
    public void reset() {
        super.reset();
        signalOnReset();
    }

    @Override
    protected Observation handle(Observation input) {
        signalOnTransform(input);
        return input;
    }

    @Override
    protected boolean getIsReady() {
        return true;
    }

    private void signalOnTransform(Observation observation) {
        for (TransformListener listener : listeners) {
            listener.onTransform(observation);
        }
    }

    private void signalOnReset() {
        for (TransformListener listener : listeners) {
            listener.onReset();
        }
    }

    public static Builder builder() {
        return new Builder();
    }

    public static class Builder {
        private List<TransformListener> listeners = new ArrayList<>();

        public Builder listener(TransformListener listener) {
            Preconditions.checkNotNull(listener);
            listeners.add(listener);

            return this;
        }

        public SignalingTransform build() {
            return new SignalingTransform(this);
        }
    }
}
