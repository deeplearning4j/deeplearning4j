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
package org.nd4j.samediff.frameworkimport

import org.nd4j.autodiff.samediff.internal.AbstractDependencyTracker
import org.nd4j.autodiff.samediff.internal.IDependencyMap

/**
 * Concrete implementation of AbstractDependencyTracker for ImportGraph operations and variables
 */
class ImportDependencyTracker : AbstractDependencyTracker<String, String>() {

    override fun newTMap(): IDependencyMap<String, *> {
        return StringDependencyMap<String>()
    }

    override fun newTSet(): Set<String> {
        return LinkedHashSet()
    }

    override fun toStringT(t: String): String {
        return t
    }

    override fun toStringD(d: String): String {
        return d
    }
}