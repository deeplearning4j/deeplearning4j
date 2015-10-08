package org.nd4j.linalg.api.parallel.tasks.cpu;

import org.nd4j.linalg.api.ops.Op;
import org.nd4j.linalg.api.parallel.tasks.Task;

public abstract class BaseCPUAction extends AbstractCPUTask<Void> {

    public BaseCPUAction(int threshold, int n, int offsetX, int offsetZ, int incrX, int incrZ){
        super(threshold, n, offsetX, offsetZ, incrX, incrZ );
    }

    public BaseCPUAction(Op op, int threshold){
        super(op,threshold);
    }

    public BaseCPUAction(int threshold, int tadIdx, int tadDim){
        super(threshold,tadIdx,tadDim);
    }


    @Override
    public Void blockUntilComplete(){
        if(future==null && subTasks == null ){
            //invokeAsync hasn't been called?
            invokeAsync();
        }
        if(future!=null){
            try{
                future.get();
            }catch(Exception e ){
                throw new RuntimeException(e);
            }
        } else {
            for( Task<?> t : subTasks ){
                t.blockUntilComplete();
            }
        }
        return null;
    }

}
