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

package org.ansj.recognition.impl;

import org.ansj.domain.Result;
import org.ansj.domain.Term;
import org.ansj.library.DicLibrary;
import org.ansj.recognition.Recognition;
import org.nlpcn.commons.lang.tire.domain.Forest;

import java.util.List;

/**
 * 
 * 用户自定词典识别 多本词典加入后将不再具有先后顺序,合并后统一规划.如果需要先后顺序请分别每个词典调用 Result.Recognition().Recognition() 这种方式 TODO:这种写灵活性是够了,但是速度不咋地.发愁........该不该这么写.先保留吧..也许在下一个版本中来做把
 * 
 * @author Ansj
 *
 */
public class DicRecognition implements Recognition {

    private static final long serialVersionUID = 7487741700410080896L;

    private Forest[] forests = null;

    public DicRecognition() {
        forests = DicLibrary.gets(DicLibrary.DEFAULT);
    }

    public DicRecognition(String[] keys) {
        forests = DicLibrary.gets(keys);
    }

    /**
     * @param forests
     */
    public DicRecognition(Forest[] forests) {
        this.forests = forests;
    }

    public DicRecognition(Forest forest) {
        this.forests = new Forest[] {forest};
    }

    @Override
    public void recognition(Result result) {
        for (Forest forest : forests) {
            if (forest == null) {
                continue;
            }
            recognition(result, forest);
        }
    }

    private void recognition(Result result, Forest forest) {
        List<Term> terms = result.getTerms();

    }

}
