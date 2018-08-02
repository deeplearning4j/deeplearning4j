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

import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.apache.commons.math3.util.FastMath;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.parameterserver.distributed.enums.ExecutionMode;
import org.nd4j.parameterserver.distributed.logic.storage.WordVectorStorage;
import org.nd4j.parameterserver.distributed.messages.BaseVoidMessage;
import org.nd4j.parameterserver.distributed.messages.DistributedMessage;
import org.nd4j.parameterserver.distributed.messages.aggregations.InitializationAggregation;

/**
 * @author raver119@gmail.com
 */
@NoArgsConstructor
@Builder
@Data
@Slf4j
public class DistributedInitializationMessage extends BaseVoidMessage implements DistributedMessage {

    protected int vectorLength;
    protected int numWords;
    protected long seed;
    protected boolean useHs;
    protected boolean useNeg;
    protected int columnsPerShard;

    public DistributedInitializationMessage(int vectorLength, int numWords, long seed, boolean useHs, boolean useNeg,
                    int columnsPerShard) {
        super(4);
        this.vectorLength = vectorLength;
        this.numWords = numWords;
        this.seed = seed;
        this.useHs = useHs;
        this.useNeg = useNeg;
        this.columnsPerShard = columnsPerShard;
    }

    /**
     * This method initializes shard storage with given data
     */
    @Override
    public void processMessage() {
        // protection check, we definitely don't want double spending here
        INDArray syn0 = storage.getArray(WordVectorStorage.SYN_0);
        INDArray syn1 = storage.getArray(WordVectorStorage.SYN_1);
        INDArray syn1Neg = storage.getArray(WordVectorStorage.SYN_1_NEGATIVE);
        INDArray expTable = storage.getArray(WordVectorStorage.EXP_TABLE);
        if (syn0 == null) {
            log.info("sI_{} is starting initialization...", transport.getShardIndex());

            // we initialize only syn0/syn1/syn1neg and expTable
            // negTable will be initalized at driver level and will be shared via message
            Nd4j.getRandom().setSeed(seed * (shardIndex + 1));

            if (voidConfiguration.getExecutionMode() == ExecutionMode.AVERAGING) {
                // each shard has full own copy
                columnsPerShard = vectorLength;
            } else if (voidConfiguration.getExecutionMode() == ExecutionMode.SHARDED) {
                // each shard will have only part of the data
                if (voidConfiguration.getNumberOfShards() - 1 == shardIndex) {
                    int modulo = vectorLength % voidConfiguration.getNumberOfShards();
                    if (modulo != 0) {
                        columnsPerShard += modulo;
                        log.info("Got inequal split. using higher number of elements: {}", columnsPerShard);
                    }
                }
            }

            int[] shardShape = new int[] {numWords, columnsPerShard};

            syn0 = Nd4j.rand(shardShape, 'c').subi(0.5).divi(vectorLength);

            if (useHs)
                syn1 = Nd4j.create(shardShape, 'c');

            if (useNeg)
                syn1Neg = Nd4j.create(shardShape, 'c');

            // we handle full exp table here
            expTable = initExpTable(100000);


            storage.setArray(WordVectorStorage.SYN_0, syn0);

            if (useHs)
                storage.setArray(WordVectorStorage.SYN_1, syn1);
            if (useNeg)
                storage.setArray(WordVectorStorage.SYN_1_NEGATIVE, syn1Neg);

            storage.setArray(WordVectorStorage.EXP_TABLE, expTable);

            InitializationAggregation ia = new InitializationAggregation((short) voidConfiguration.getNumberOfShards(),
                            transport.getShardIndex());
            ia.setOriginatorId(this.originatorId);
            transport.sendMessage(ia);
        }
    }

    protected INDArray initExpTable(int tableWidth) {
        double[] expTable = new double[tableWidth];
        for (int i = 0; i < expTable.length; i++) {
            double tmp = FastMath.exp((i / (double) expTable.length * 2 - 1) * 6);
            expTable[i] = tmp / (tmp + 1.0);
        }

        return Nd4j.create(expTable);
    }
}
