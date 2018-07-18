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

public class NumNatureAttr implements Serializable {

    /**
     * 
     */
    private static final long serialVersionUID = 1L;

    public static final NumNatureAttr NULL = new NumNatureAttr();

    // 是有可能是一个数字
    public int numFreq = -1;

    // 数字的结尾
    public int numEndFreq = -1;

    // 最大词性是否是数字
    public boolean flag = false;

    public NumNatureAttr() {}
}
