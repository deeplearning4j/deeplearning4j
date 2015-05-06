/*
 *
 *  * Copyright 2015 Skymind,Inc.
 *  *
 *  *    Licensed under the Apache License, Version 2.0 (the "License");
 *  *    you may not use this file except in compliance with the License.
 *  *    You may obtain a copy of the License at
 *  *
 *  *        http://www.apache.org/licenses/LICENSE-2.0
 *  *
 *  *    Unless required by applicable law or agreed to in writing, software
 *  *    distributed under the License is distributed on an "AS IS" BASIS,
 *  *    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  *    See the License for the specific language governing permissions and
 *  *    limitations under the License.
 *
 */

package org.deeplearning4j.iterativereduce.impl.multilayer;



import java.io.DataOutputStream;
import java.io.IOException;
import java.util.Collection;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.util.ToolRunner;
import org.deeplearning4j.scaleout.api.ir.ParameterVectorUpdateable;
import org.deeplearning4j.iterativereduce.runtime.ComputableMaster;
import org.deeplearning4j.iterativereduce.runtime.yarn.appmaster.ApplicationMaster;
import org.nd4j.linalg.factory.Nd4j;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Master node:
 *
 *
 */
public class Master implements ComputableMaster<ParameterVectorUpdateable> {

    ParameterVectorUpdateable lastMasterUpdate = null;
    protected Configuration conf = null;
    protected static final Logger log = LoggerFactory.getLogger(Master.class);

    /**
     * Q: "is compute() called before complete() is called in last epoch?"
     *
     *
     */
    @Override
    public void complete(DataOutputStream osStream) throws IOException {
        log.info( "IR DBN Master Node: Complete!" );
        Nd4j.write(lastMasterUpdate.get(),osStream);
    }


    /**
     * Master::Compute
     *
     * This is where the worker parameter averaged updates come in and are processed
     *
     */
    @Override
    public ParameterVectorUpdateable compute(
            Collection<ParameterVectorUpdateable> workerUpdates,
            Collection<ParameterVectorUpdateable> masterUpdates) {

        log.info( "--------------- Master::Compute() -------------- " );
        ParameterVectorUpdateable first = null;
        for(ParameterVectorUpdateable update : workerUpdates) {
            if(first == null)
                first = update;
            else
                first.get().addi(update.get());
        }

        first.get().divi(workerUpdates.size());
        lastMasterUpdate = first;
        return first;
    }



    @Override
    public ParameterVectorUpdateable getResults() {
        return this.lastMasterUpdate;
    }

    @Override
    public void setup(Configuration c) {

    }

    public static void main(String[] args) throws Exception {
        Master pmn = new Master();
        ApplicationMaster<ParameterVectorUpdateable> am = new ApplicationMaster<>(
                pmn, ParameterVectorUpdateable.class);

        ToolRunner.run(am, args);
    }

}