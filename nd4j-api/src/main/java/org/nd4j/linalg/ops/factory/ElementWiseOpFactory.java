/*
 * Copyright 2015 Skymind,Inc.
 *
 *    Licensed under the Apache License, Version 2.0 (the "License");
 *    you may not use this file except in compliance with the License.
 *    You may obtain a copy of the License at
 *
 *        http://www.apache.org/licenses/LICENSE-2.0
 *
 *    Unless required by applicable law or agreed to in writing, software
 *    distributed under the License is distributed on an "AS IS" BASIS,
 *    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *    See the License for the specific language governing permissions and
 *    limitations under the License.
 */

package org.nd4j.linalg.ops.factory;

import org.nd4j.linalg.ops.ElementWiseOp;

/**
 * Create element wise operations
 *
 * @author Adam Gibson
 */
public interface ElementWiseOpFactory {

    /**
     * Create element wise operations
     *
     * @return the element wise operation to create
     */
    ElementWiseOp create();


    /**
     * Create with the given arguments
     *
     * @param args
     * @return
     */
    ElementWiseOp create(Object[] args);


}
