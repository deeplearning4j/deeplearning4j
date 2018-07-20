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

package org.nd4j.linalg;

import org.junit.Ignore;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;

/**
 * TODO !! TO REMOVE : only for getting a better understanding of indexes !!
 */
@Ignore // temporary ignored
public class TempNdArrayIndexTest {

    public static void main(String[] args) {
        //deliberately not doing a square matrix
        INDArray cOrder = Nd4j.linspace(1, 12, 12).reshape('c', 4, 3);

        System.out.println("==========================");
        System.out.println("C order..");
        System.out.println("==========================");
        System.out.println(cOrder);
        System.out.println(cOrder.shapeInfoToString());

        System.out.println("==========================");
        System.out.println("Shape and stride of a view from the above c order array");
        System.out.println("==========================");
        INDArray cOrderView = cOrder.get(NDArrayIndex.interval(2, 4), NDArrayIndex.interval(0, 2));
        System.out.println(cOrderView);
        System.out.println("This array is a view? " + cOrderView.isView());
        System.out.println(cOrderView.shapeInfoToString());

        System.out.println("==========================");
        System.out.println("Shape and stride of a view from the above c order array");
        System.out.println("==========================");
        INDArray cOrderView2 = cOrder.get(NDArrayIndex.interval(2, 4), NDArrayIndex.interval(1, 3));
        System.out.println(cOrderView2);
        System.out.println("This array is a view? " + cOrderView2.isView());
        System.out.println(cOrderView2.shapeInfoToString());

        System.out.println("==========================");
        System.out.println("One way to build an f order array");
        System.out.println("==========================");
        INDArray fOrder = Nd4j.linspace(1, 12, 12).reshape('f', 4, 3);
        System.out.println(fOrder);
        System.out.println(fOrder.shapeInfoToString());
    }
}
