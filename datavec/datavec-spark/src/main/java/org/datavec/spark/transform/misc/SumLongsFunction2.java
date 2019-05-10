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

package org.datavec.spark.transform.misc;

import org.apache.spark.api.java.function.Function2;

/**
 * Created by Alex on 03/09/2016.
 */
public class SumLongsFunction2 implements Function2<Long, Long, Long> {
    @Override
    public Long call(Long l1, Long l2) throws Exception {
        return l1 + l2;
    }
}
