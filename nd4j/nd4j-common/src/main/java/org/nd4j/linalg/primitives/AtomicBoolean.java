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

package org.nd4j.linalg.primitives;

public class AtomicBoolean extends java.util.concurrent.atomic.AtomicBoolean {

    public AtomicBoolean(boolean initialValue){
        super(initialValue);
    }

    public AtomicBoolean(){
        this(false);
    }

    @Override
    public boolean equals(Object o){
        if(o instanceof AtomicBoolean){
            return get() == ((AtomicBoolean)o).get();
        } else if(o instanceof Boolean){
            return get() == ((Boolean)o);
        }
        return false;
    }

    @Override
    public int hashCode(){
        return get() ? 1 : 0;
    }

}
