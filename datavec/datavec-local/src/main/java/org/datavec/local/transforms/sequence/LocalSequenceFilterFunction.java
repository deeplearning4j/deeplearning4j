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

import lombok.AllArgsConstructor;
import org.datavec.api.transform.filter.Filter;
import org.datavec.api.writable.Writable;
import org.nd4j.linalg.function.Function;

import java.util.List;

/**
 * Created by Alex on 5/03/2016.
 */
@AllArgsConstructor
public class LocalSequenceFilterFunction implements Function<List<List<Writable>>, Boolean> {

    private final Filter filter;

    @Override
    public Boolean apply(List<List<Writable>> v1) {
        return !filter.removeSequence(v1); //return true to keep example (Filter: return true to remove)
    }
}
