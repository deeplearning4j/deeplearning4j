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

package org.nd4j.codegen.api.doc

import org.nd4j.codegen.api.CodeComponent

enum class DocScope {
    ALL, CLASS_DOC_ONLY, CREATORS_ONLY, CONSTRUCTORS_ONLY;

    fun applies(codeComponent: CodeComponent): Boolean {
        return when (this) {
            ALL -> true
            CLASS_DOC_ONLY -> codeComponent === CodeComponent.CLASS_DOC
            CREATORS_ONLY -> codeComponent === CodeComponent.OP_CREATOR
            CONSTRUCTORS_ONLY -> codeComponent === CodeComponent.CONSTRUCTOR
        }
    }
}