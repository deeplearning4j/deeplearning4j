/*
 *  ******************************************************************************
 *  *
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
 *  * Unless required by applicable law or agreed to in writing, software
 *  * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 *  * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 *  * License for the specific language governing permissions and limitations
 *  * under the License.
 *  *
 *  * SPDX-License-Identifier: Apache-2.0
 *  *****************************************************************************
 */

package org.deeplearning4j.text.tokenization.tokenizer.preprocessor;

import lombok.NonNull;
import org.deeplearning4j.text.tokenization.tokenizer.TokenPreProcess;
import org.nd4j.common.base.Preconditions;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.List;

/**
 * CompositePreProcessor is a {@link TokenPreProcess} that applies multiple preprocessors sequentially
 * @author Alex Black
 */
public class CompositePreProcessor implements TokenPreProcess {

    private List<TokenPreProcess> preProcessors;

    public CompositePreProcessor(@NonNull TokenPreProcess... preProcessors){
        Preconditions.checkState(preProcessors.length > 0, "No preprocessors were specified (empty input)");
        this.preProcessors = Arrays.asList(preProcessors);
    }

    public CompositePreProcessor(@NonNull Collection<? extends TokenPreProcess> preProcessors){
        Preconditions.checkState(!preProcessors.isEmpty(), "No preprocessors were specified (empty input)");
        this.preProcessors = new ArrayList<>(preProcessors);
    }

    @Override
    public String preProcess(String token) {
        String s = token;
        for(TokenPreProcess tpp : preProcessors){
            s = tpp.preProcess(s);
        }
        return s;
    }
}
