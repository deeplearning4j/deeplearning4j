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

package org.datavec.local.transforms.sequence;

import org.datavec.api.writable.Writable;
import org.nd4j.linalg.function.Function;

import java.util.Collections;
import java.util.List;

/**
 * Very simple function to convert an example to sequence of length 1
 *
 * @author Alex Black
 */
public class ConvertToSequenceLengthOne implements Function<List<Writable>, List<List<Writable>>> {
    @Override
    public List<List<Writable>> apply(List<Writable> writables) {
        return Collections.singletonList(writables);
    }
}
