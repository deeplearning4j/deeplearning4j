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

package org.ansj.domain;

import java.io.Serializable;

/**
 * 人名标注pojo类
 * 
 * @author ansj
 * 
 */
public class PersonNatureAttr implements Serializable {
    /**
     * 
     */
    private static final long serialVersionUID = -8443825231800208197L;

    // public int B = -1;//0 姓氏
    // public int C = -1;//1 双名的首字
    // public int D = -1;//2 双名的末字
    // public int E = -1;//3 单名
    // public int N = -1; //4任意字
    // public int L = -1;//11 人名的下文
    // public int M = -1;//12 两个中国人名之间的成分
    // public int m = -1;//44 可拆分的姓名
    // String[] parretn = {"BC", "BCD", "BCDE", "BCDEN"}
    // double[] factory = {"BC", "BCD", "BCDE", "BCDEN"}

    public static final PersonNatureAttr NULL = new PersonNatureAttr();

    private int[][] locFreq = null;

    public int split;
    // 12
    public int begin;
    // 11+12
    public int end;

    public int allFreq;

    // 是否有可能是名字的第一个字
    public boolean flag;

    /**
     * 设置
     * 
     * @param index
     * @param freq
     */
    public void addFreq(int index, int freq) {
        switch (index) {
            case 11:
                this.end += freq;
                allFreq += freq;
                break;
            case 12:
                this.end += freq;
                this.begin += freq;
                allFreq += freq;
                break;
            case 44:
                this.split += freq;
                allFreq += freq;
                break;
        }
    }

    /**
     * 得道某一个位置的词频
     * 
     * @param length
     * @param loc
     * @return
     */
    public int getFreq(int length, int loc) {
        if (locFreq == null)
            return 0;
        if (length > 3)
            length = 3;
        if (loc > 4)
            loc = 4;
        return locFreq[length][loc];
    }

    /**
     * 词频记录表
     * 
     * @param ints
     */
    public void setlocFreq(int[][] ints) {
        for (int i = 0; i < ints.length; i++) {
            if (ints[i][0] > 0) {
                flag = true;
                break;
            }
        }
        locFreq = ints;
    }

    @Override
    public String toString() {
        StringBuilder sb = new StringBuilder();
        sb.append("begin=" + begin);
        sb.append(",");
        sb.append("end=" + end);
        sb.append(",");
        sb.append("split=" + split);
        return sb.toString();
    }
}
