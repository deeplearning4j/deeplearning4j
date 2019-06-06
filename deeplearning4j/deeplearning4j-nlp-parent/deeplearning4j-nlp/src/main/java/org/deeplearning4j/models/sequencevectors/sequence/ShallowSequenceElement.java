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

package org.deeplearning4j.models.sequencevectors.sequence;

/**
 * This is special shallow SequenceElement implementation, that doesn't hold labels or any other custom user-defined data
 *
 * @author raver119@gmail.com
 */
public class ShallowSequenceElement extends SequenceElement {

    public ShallowSequenceElement(double frequency, long id) {
        this.storageId = id;
        this.elementFrequency.set(frequency);
    }

    @Override
    public String getLabel() {
        return null;
    }

    @Override
    public String toJSON() {
        return null;
    }
}
