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

package org.datavec.api.io.labels;

import org.datavec.api.writable.Writable;

import java.io.Serializable;
import java.net.URI;

public interface PathLabelGenerator extends Serializable {

    Writable getLabelForPath(String path);

    Writable getLabelForPath(URI uri);

    /**
     * If true: infer the set of possible label classes, and convert these to integer indexes. If when true, the
     * returned Writables should be text writables.<br>
     * <br>
     * For regression use cases (or PathLabelGenerator classification instances that do their own label -> integer
     * assignment), this should return false.
     *
     * @return whether label classes should be inferred
     */
    boolean inferLabelClasses();

}
