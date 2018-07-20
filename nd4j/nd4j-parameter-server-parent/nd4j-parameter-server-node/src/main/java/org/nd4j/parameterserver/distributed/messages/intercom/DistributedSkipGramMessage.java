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

package org.nd4j.parameterserver.distributed.messages.intercom;

import lombok.NonNull;
import org.nd4j.parameterserver.distributed.messages.BaseVoidMessage;
import org.nd4j.parameterserver.distributed.messages.DistributedMessage;
import org.nd4j.parameterserver.distributed.messages.requests.SkipGramRequestMessage;

/**
 * @author raver119@gmail.com
 */
public class DistributedSkipGramMessage extends BaseVoidMessage implements DistributedMessage {

    // learning rate for this sequence
    protected double alpha;

    // current word & lastWord
    protected int w1;
    protected int w2;

    // following fields are for hierarchic softmax
    // points & codes for current word
    protected int[] points;
    protected byte[] codes;

    protected short negSamples;

    protected long nextRandom;


    protected DistributedSkipGramMessage() {
        super(23);
    }

    public DistributedSkipGramMessage(@NonNull SkipGramRequestMessage message) {
        this();

        this.w1 = message.getW1();
        this.w2 = message.getW2();
        this.points = message.getPoints();
        this.codes = message.getCodes();

        this.negSamples = message.getNegSamples();
        this.nextRandom = message.getNextRandom();
    }

    @Override
    public void processMessage() {

    }
}
