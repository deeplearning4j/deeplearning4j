/*
 *  ******************************************************************************
 *  *
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
 *  * See the NOTICE file distributed with this work for additional
 *  * information regarding copyright ownership.
 *  * Unless required by applicable law or agreed to in writing, software
 *  * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 *  * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 *  * License for the specific language governing permissions and limitations
 *  * under the License.
 *  *
 *  * SPDX-License-Identifier: Apache-2.0
 *  *****************************************************************************
 */

package org.nd4j.codegen.api.generator

import org.nd4j.codegen.api.Language
import org.nd4j.codegen.api.NamespaceOps
import java.io.File
import java.io.IOException

interface Generator {
    fun language(): Language?

    @Throws(IOException::class)
    fun generateNamespaceNd4j(namespace: NamespaceOps?, config: GeneratorConfig?, directory: File?, className: String?)

    @Throws(IOException::class)
    fun generateNamespaceSameDiff(namespace: NamespaceOps?, config: GeneratorConfig?, directory: File?, className: String?)
}