package org.nd4j.linalg.api.parallel.bufferops;

import lombok.AllArgsConstructor;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.ScalarOp;
import org.nd4j.linalg.api.ops.executioner.OpExecutionerUtil;
import org.nd4j.linalg.api.ops.impl.scalar.*;
import org.nd4j.linalg.api.parallel.bufferops.impl.scalar.ScalarOpDataBufferAction;
import org.nd4j.linalg.api.parallel.bufferops.impl.transform.*;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.RecursiveAction;

/**VectorOpDataBufferAction: for executing row/column vector operations on a DataBuffer in parallel.
 * Implements parallel addiRowVector, muliColumnVector, etc operations
 * @author Alex Black
 */
@AllArgsConstructor
public class VectorOpDataBufferAction extends RecursiveAction {

    private final INDArray vector;
    private final INDArray array;
    private final char op;
    private final int dim;  //0 = xiRowVector, 1 = xiColumnVector
    private final int parallelThreshold;

    @Override
    protected void compute() {

        if(array.isVector()){
            //Two cases: either 'vector' is a scalar, or doing a single vector <op> vector
            if(vector.isScalar()){
                //Edge case: 'vector' is a scalar
                ScalarOp scalarOp = null;
                switch(op){
                    case 'a':
                        scalarOp = new ScalarAdd(array,vector.getDouble(0));
                        break;
                    case 's':
                        scalarOp = new ScalarSubtraction(array,vector.getDouble(0));
                        break;
                    case 'm':
                        scalarOp = new ScalarMultiplication(array,vector.getDouble(0));
                        break;
                    case 'd':
                        scalarOp = new ScalarDivision(array,vector.getDouble(0));
                        break;
                    case 'h':
                        scalarOp = new ScalarReverseSubtraction(array,vector.getDouble(0));
                        break;
                    case 't':
                        scalarOp = new ScalarReverseDivision(array,vector.getDouble(0));
                        break;
                    case 'p':
                        scalarOp = new ScalarSet(array,vector.getDouble(0));
                        break;
                    default:
                        throw new RuntimeException("Unknown op: " + op);
                }
                new ScalarOpDataBufferAction(scalarOp,parallelThreshold,array.length(),array.data(),array.data(),
                        array.offset(),array.offset(),array.elementWiseStride(),array.elementWiseStride()).invoke();
                return;
            } else {
                //Only one op required
                int n = vector.length();
                int offsetV = vector.offset();
                int incrV = vector.elementWiseStride();
                int offsetA = array.offset();
                int incrA = array.elementWiseStride();

                DataBuffer da = array.data();
                DataBuffer dv = vector.data();

                switch(op){
                    case 'a':
                        new AddOpDataBufferAction(null,parallelThreshold,n,da,dv,da,
                                offsetA,offsetV,offsetA,incrA,incrV,incrA).invoke();
                        return;
                    case 's':
                        new SubOpDataBufferAction(null,parallelThreshold,n,da,dv,da,
                                offsetA,offsetV,offsetA,incrA,incrV,incrA).invoke();
                        return;
                    case 'm':
                        new MulOpDataBufferAction(null,parallelThreshold,n,da,dv,da,
                                offsetA,offsetV,offsetA,incrA,incrV,incrA).invoke();
                        return;
                    case 'd':
                        new DivOpDataBufferAction(null,parallelThreshold,n,da,dv,da,
                                offsetA,offsetV,offsetA,incrA,incrV,incrA).invoke();
                        return;
                    case 'h':
                        new RSubOpDataBufferAction(null,parallelThreshold,n,da,dv,da,
                                offsetA,offsetV,offsetA,incrA,incrV,incrA).invoke();
                        return;
                    case 't':
                        new RDivOpDataBufferAction(null,parallelThreshold,n,da,dv,da,
                                offsetA,offsetV,offsetA,incrA,incrV,incrA).invoke();
                        return;
                    case 'p':
                        new CopyOpDataBufferAction(null,parallelThreshold,n,da,dv,da,
                                offsetA,offsetV,offsetA,incrA,incrV,incrA).invoke();
                        return;
                    default:
                        throw new RuntimeException("Unknown op: " + op);
                }
            }

        } else {
            //Usual case
            int nOps = array.size(dim);
            List<RecursiveAction> blockList = new ArrayList<>(nOps);

            DataBuffer dVector = vector.data();
            DataBuffer dArray = array.data();

            OpExecutionerUtil.Tensor1DStats t1d = OpExecutionerUtil.get1DTensorStats(array, (dim == 1 ? 0 : 1));
            int incrA = t1d.getElementWiseStride();


            int n = vector.length();
            int offsetV = vector.offset();
            int incrV = vector.elementWiseStride();

            for( int i=0; i<nOps; i++ ){
                int arrayVectorOffset = t1d.firstTensorOffset+i*t1d.getTensorStartSeparation();
                switch(op){
                    case 'a':
                        RecursiveAction a = new AddOpDataBufferAction(null,parallelThreshold,n,dArray,dVector,dArray,
                                arrayVectorOffset,offsetV,arrayVectorOffset,incrA,incrV,incrA);
                        a.fork();
                        blockList.add(a);
                        break;
                    case 's':
                        RecursiveAction s = new SubOpDataBufferAction(null,parallelThreshold,n,dArray,dVector,dArray,
                                arrayVectorOffset,offsetV,arrayVectorOffset,incrA,incrV,incrA);
                        s.fork();
                        blockList.add(s);
                        break;
                    case 'm':
                        RecursiveAction m = new MulOpDataBufferAction(null,parallelThreshold,n,dArray,dVector,dArray,
                                arrayVectorOffset,offsetV,arrayVectorOffset,incrA,incrV,incrA);
                        m.fork();
                        blockList.add(m);
                        break;
                    case 'd':
                        RecursiveAction d = new DivOpDataBufferAction(null,parallelThreshold,n,dArray,dVector,dArray,
                                arrayVectorOffset,offsetV,arrayVectorOffset,incrA,incrV,incrA);
                        d.fork();
                        blockList.add(d);
                        break;
                    case 'h':   //reverse subtraction
                        RecursiveAction h = new RSubOpDataBufferAction(null,parallelThreshold,n,dArray,dVector,dArray,
                                arrayVectorOffset,offsetV,arrayVectorOffset,incrA,incrV,incrA);
                        h.fork();
                        blockList.add(h);
                        break;
                    case 't':   //reverse division
                        RecursiveAction t = new RDivOpDataBufferAction(null,parallelThreshold,n,dArray,dVector,dArray,
                                arrayVectorOffset,offsetV,arrayVectorOffset,incrA,incrV,incrA);
                        t.fork();
                        blockList.add(t);
                        break;
                    case 'p':   //put/copy
                        RecursiveAction p = new CopyOpDataBufferAction(null,parallelThreshold,n,dArray,dVector,dArray,
                                arrayVectorOffset,offsetV,arrayVectorOffset,incrA,incrV,incrA);
                        p.fork();
                        blockList.add(p);
                        break;
                    default:
                        throw new RuntimeException("Unknown op: " + op);
                }
            }

            //Block until all complete
            for(RecursiveAction a : blockList ) a.join();
        }

    }
}
