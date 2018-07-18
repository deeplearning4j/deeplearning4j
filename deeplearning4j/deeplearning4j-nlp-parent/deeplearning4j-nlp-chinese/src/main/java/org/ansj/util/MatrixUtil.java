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

package org.ansj.util;

public class MatrixUtil {

    /**
     * 向量求和
     * 
     * @param dbs
     * @return
     */
    public static double sum(double[] dbs) {
        double value = 0;
        for (double d : dbs) {
            value += d;
        }
        return value;
    }

    public static int sum(int[] dbs) {
        int value = 0;
        for (int d : dbs) {
            value += d;
        }
        return value;
    }

    public static double sum(double[][] w) {

        double value = 0;
        for (double[] dbs : w) {
            value += sum(dbs);
        }
        return value;
    }

    public static void dot(double[] feature, double[] feature1) {
        if (feature1 == null) {
            return;
        }
        for (int i = 0; i < feature1.length; i++) {
            feature[i] += feature1[i];
        }
    }

    public static void dot(float[] feature, float[] feature1) {
        if (feature1 == null) {
            return;
        }

        if (feature == null) {
            return;
        }

        int min = Math.min(feature.length, feature1.length);

        for (int i = 0; i < min; i++) {
            feature[i] += feature1[i];
        }
    }
}
