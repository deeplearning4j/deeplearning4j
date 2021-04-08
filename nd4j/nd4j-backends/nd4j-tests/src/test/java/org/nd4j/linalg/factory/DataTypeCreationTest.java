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

package org.nd4j.linalg.factory;

import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.nd4j.common.tests.tags.NativeTag;
import org.nd4j.common.tests.tags.TagNames;
import org.nd4j.linalg.BaseNd4jTestWithBackends;
import org.nd4j.linalg.api.buffer.DataType;

import static org.junit.jupiter.api.Assertions.assertEquals;

@Tag(TagNames.RNG)
@NativeTag
public class DataTypeCreationTest  extends BaseNd4jTestWithBackends  {

    @Test
    public void testDataTypeCreation() {
        for(DataType dataType : new DataType[] {DataType.DOUBLE,DataType.FLOAT}) {
            assertEquals(dataType,Nd4j.create(dataType,1,2).dataType());
            assertEquals(dataType,Nd4j.rand(dataType,1,2).dataType());
            assertEquals(dataType,Nd4j.randn(dataType,1,2).dataType());
            assertEquals(dataType,Nd4j.rand(dataType,'c',1,2).dataType());
            assertEquals(dataType,Nd4j.randn(dataType,'c',1,2).dataType());

        }
    }

}
