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

package org.datavec.local.transforms;

import org.datavec.api.writable.Writable;
import org.nd4j.linalg.function.Function;

import java.util.List;

/**
 * Used for filtering empty records
 *
 * @author Adam Gibson
 */
public class SequenceEmptyRecordFunction implements Function<List<List<Writable>>, Boolean> {
    @Override
    public Boolean apply(List<List<Writable>> v1) {
        return v1.isEmpty();
    }
}
