/*
 *  ******************************************************************************
 *  *
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
 *  *  See the NOTICE file distributed with this work for additional
 *  *  information regarding copyright ownership.
 *  * Unless required by applicable law or agreed to in writing, software
 *  * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 *  * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 *  * License for the specific language governing permissions and limitations
 *  * under the License.
 *  *
 *  * SPDX-License-Identifier: Apache-2.0
 *  *****************************************************************************
 */

package org.datavec.spark.transform.analysis;

import org.apache.spark.api.java.function.FlatMapFunction;
import org.datavec.api.writable.Writable;

import java.util.Iterator;
import java.util.List;

/**
 * SequenceFlatMapFunction: very simple function used to flatten a sequence
 * Typically used only internally for certain analysis operations
 *
 * @author Alex Black
 */
public class SequenceFlatMapFunction implements FlatMapFunction<List<List<Writable>>, List<Writable>> {

    @Override
    public Iterator<List<Writable>> call(List<List<Writable>> collections) throws Exception {
        return collections.iterator();
    }

}
