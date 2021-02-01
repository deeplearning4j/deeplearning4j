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

package org.nd4j.autodiff.samediff;

import lombok.Data;

import java.io.Closeable;

/**
 * Used with {@link SameDiff#withNameScope(String)}
 *
 * @author Alex Black
 */
@Data
public class NameScope implements Closeable {
    private final SameDiff sameDiff;
    private final String name;

    public NameScope(SameDiff sameDiff, String name){
        this.sameDiff = sameDiff;
        this.name = name;
    }

    @Override
    public void close() {
        sameDiff.closeNameScope(this);
    }

    @Override
    public String toString(){
        return "NameScope(" + name + ")";
    }
}
