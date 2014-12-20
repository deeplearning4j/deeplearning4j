package org.deeplearning4j.iterativereduce.impl;



import java.io.DataOutputStream;
import java.io.IOException;
import java.util.Collection;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.util.ToolRunner;
import org.deeplearning4j.iterativereduce.runtime.ComputableMaster;
import org.deeplearning4j.iterativereduce.runtime.yarn.appmaster.ApplicationMaster;



public class Master implements ComputableMaster<ParameterVectorUpdateable> {

    ParameterVectorUpdateable lastMasterUpdate = null;
    protected Configuration conf = null;



    /**
     * Q: "is compute() called before complete() is called in last epoch?"
     *
     *
     */
    @Override
    public void complete(DataOutputStream osStream) throws IOException {

        System.out.println( "IR DBN Master Node: Complete!" );
        //this.dbn_averaged_master.write( osStream );


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

        System.out.println( "--------------- Master::Compute() -------------- " );

        return null;
    }



    @Override
    public ParameterVectorUpdateable getResults() {

        //	System.out.println("\n\nMaster > getResults() -----------------------------------\n\n");
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