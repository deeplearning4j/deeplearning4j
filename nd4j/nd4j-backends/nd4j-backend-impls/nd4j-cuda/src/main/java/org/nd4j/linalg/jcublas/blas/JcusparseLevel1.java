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

package org.nd4j.linalg.jcublas.blas;

import org.nd4j.linalg.api.blas.impl.SparseBaseLevel1;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.ndarray.INDArray;

/**
 * @author Audrey Loeffel
 */
public class JcusparseLevel1 extends SparseBaseLevel1 {

    @Override
    protected double ddoti(long N, INDArray X, DataBuffer indx, INDArray Y) {
        return 0;
    }

    @Override
    protected double sdoti(long N, INDArray X, DataBuffer indx, INDArray Y) {
        return 0;
    }

    @Override
    protected double hdoti(long N, INDArray X, DataBuffer indx, INDArray Y) {
        return 0;
    }

    @Override
    protected double snrm2(long N, INDArray X, int incx) {
        return 0;
    }

    @Override
    protected double dnrm2(long N, INDArray X, int incx) {
        return 0;
    }

    @Override
    protected double hnrm2(long N, INDArray X, int incx) {
        return 0;
    }

    @Override
    protected double dasum(long N, INDArray X, int incx) {
        return 0;
    }

    @Override
    protected double sasum(long N, INDArray X, int incx) {
        return 0;
    }

    @Override
    protected double hasum(long N, INDArray X, int incx) {
        return 0;
    }

    @Override
    protected int isamax(long N, INDArray X, int incx) {
        return 0;
    }

    @Override
    protected int idamax(long N, INDArray X, int incx) {
        return 0;
    }

    @Override
    protected int ihamax(long N, INDArray X, int incx) {
        return 0;
    }

    @Override
    protected int isamin(long N, INDArray X, int incx) {
        return 0;
    }

    @Override
    protected int idamin(long N, INDArray X, int incx) {
        return 0;
    }

    @Override
    protected int ihamin(long N, INDArray X, int incx) {
        return 0;
    }

    @Override
    protected void daxpyi(long N, double alpha, INDArray X, DataBuffer pointers, INDArray Y) {

    }

    @Override
    protected void saxpyi(long N, double alpha, INDArray X, DataBuffer pointers, INDArray Y) {

    }

    @Override
    protected void haxpyi(long N, double alpha, INDArray X, DataBuffer pointers, INDArray Y) {

    }

    @Override
    protected void droti(long N, INDArray X, DataBuffer indexes, INDArray Y, double c, double s) {

    }

    @Override
    protected void sroti(long N, INDArray X, DataBuffer indexes, INDArray Y, double c, double s) {

    }

    @Override
    protected void hroti(long N, INDArray X, DataBuffer indexes, INDArray Y, double c, double s) {

    }

    @Override
    protected void dscal(long N, double a, INDArray X, int incx) {

    }

    @Override
    protected void sscal(long N, double a, INDArray X, int incx) {

    }

    @Override
    protected void hscal(long N, double a, INDArray X, int incx) {

    }
}
