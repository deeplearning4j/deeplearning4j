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

package org.datavec.api.writable.comparator;

import lombok.EqualsAndHashCode;
import org.datavec.api.writable.Writable;

@EqualsAndHashCode
public class LongWritableComparator implements WritableComparator {
    @Override
    public int compare(Writable o1, Writable o2) {
        return Long.compare(o1.toLong(), o2.toLong());
    }

    @Override
    public String toString() {
        return "LongWritableComparator()";
    }
}
