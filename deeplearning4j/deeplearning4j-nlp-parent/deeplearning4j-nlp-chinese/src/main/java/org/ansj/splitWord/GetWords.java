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

package org.ansj.splitWord;

public interface GetWords {
    /**
     * 全文全词全匹配
     * 
     * @param str
     *            传入的需要分词的句子
     * @return 返还分完词后的句子
     */
    public String allWords();

    /**
     * 同一个对象传入词语
     * 
     * @param temp
     *            传入的句子
     */
    public void setStr(String temp);

    /**
     * 
     * @return
     */

    public void setChars(char[] chars, int start, int end);

    public int getOffe();
}
