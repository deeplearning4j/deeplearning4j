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

package org.ansj.app.summary.pojo;

import org.ansj.app.keyword.Keyword;

import java.util.List;

/**
 * 摘要结构体封装
 * 
 * @author ansj
 * 
 */
public class Summary {

    /**
     * 关键词
     */
    private List<Keyword> keyWords = null;

    /**
     * 摘要
     */
    private String summary;

    public Summary(List<Keyword> keyWords, String summary) {
        this.keyWords = keyWords;
        this.summary = summary;
    }

    public List<Keyword> getKeyWords() {
        return keyWords;
    }

    public String getSummary() {
        return summary;
    }

}
