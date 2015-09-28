package org.nd4j.linalg.api.parallel.ops;

import lombok.AllArgsConstructor;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.ops.Op;
import org.nd4j.linalg.factory.Nd4j;

import java.util.concurrent.RecursiveAction;

/**@author Alex Black
 */
public class BufferOps {

    @AllArgsConstructor
    public static abstract class BaseDataBufferTask extends RecursiveAction {
        protected final int threshold;
        protected final int n;
        protected final DataBuffer x;
        protected final DataBuffer y;
        protected final DataBuffer z;
        protected final int offsetX;
        protected final int offsetY;
        protected final int offsetZ;
        protected final int incrX;
        protected final int incrY;
        protected final int incrZ;


        @Override
        protected void compute() {
            if(n>threshold){
                //Split task
                int nFirst = n/2;
                BaseDataBufferTask t1 = getSubTask(threshold,nFirst,x,y,z,offsetX,offsetY,offsetZ,incrX,incrY,incrZ);

                int nSecond = n - nFirst;  //handle odd cases for integer division: i.e., 5/2=2; 5 -> (2,3)
                int offsetX2 = offsetX + nFirst*incrX;
                int offsetY2 = offsetY + nFirst*incrY;
                int offsetZ2 = offsetZ + nFirst*incrZ;
                BaseDataBufferTask t2 = getSubTask(threshold,nSecond,x,y,z,offsetX2,offsetY2,offsetZ2,incrX,incrY,incrZ);

                t1.fork();
                t2.fork();
                t1.join();
                t2.join();
            } else {
                doTask();
            }
        }

        public abstract void doTask();

        public abstract BaseDataBufferTask getSubTask(int threshold, int n, DataBuffer x, DataBuffer y, DataBuffer z,
                                                  int offsetX, int offsetY, int offsetZ,
                                                  int incrX, int incrY, int incrZ );
    }

    public static class AddOpDataBufferTask extends BaseDataBufferTask{
        public AddOpDataBufferTask(int threshold, int n, DataBuffer x, DataBuffer y, DataBuffer z, int offsetX, int offsetY, int offsetZ, int incrX, int incrY, int incrZ){
            super(threshold,n,x,y,z,offsetX,offsetY,offsetZ,incrX,incrY,incrZ);
        }

        @Override
        public void doTask() {
            //Task: Z = X+Y
            if(x == z){
                //can use axpy
                Nd4j.getBlasWrapper().level1().axpy(n, 1.0, y, offsetY, incrY, z, offsetZ, incrZ);
            } else {
                //use loop
                if(x.dataType() == DataBuffer.Type.FLOAT){
                    float[] xf = (float[])x.array();
                    float[] yf = (float[])y.array();
                    float[] zf = (float[])z.array();
                    if(incrX == 1 && incrY == 1 && incrZ == 1) {
                        for (int i = 0; i < n; i++) {
                            zf[offsetZ+i] = xf[offsetX+i] + yf[offsetY+i];
                        }
                    } else {
                        for( int i=0; i<n; i++){
                            zf[offsetZ+i*incrZ] = xf[offsetX+i*incrX] + yf[offsetY+i*incrY];
                        }
                    }
                } else {
                    double[] xd = (double[])x.array();
                    double[] yd = (double[])y.array();
                    double[] zd = (double[])z.array();
                    if(incrX == 1 && incrY == 1 && incrZ == 1) {
                        for (int i = 0; i < n; i++) {
                            zd[offsetZ+i] = xd[offsetX+i] + yd[offsetY+i];
                        }
                    } else {
                        for( int i=0; i<n; i++){
                            zd[offsetZ+i*incrZ] = xd[offsetX+i*incrX] + yd[offsetY+i*incrY];
                        }
                    }
                }
            }
        }

        @Override
        public BaseDataBufferTask getSubTask(int threshold, int n, DataBuffer x, DataBuffer y, DataBuffer z, int offsetX, int offsetY, int offsetZ, int incrX, int incrY, int incrZ) {
            return new AddOpDataBufferTask(threshold,n,x,y,z,offsetX,offsetY,offsetZ,incrX,incrY,incrZ);
        }
    }

    public static class SubOpDataBufferTask extends BaseDataBufferTask {
        public SubOpDataBufferTask(int threshold, int n, DataBuffer x, DataBuffer y, DataBuffer z, int offsetX, int offsetY, int offsetZ, int incrX, int incrY, int incrZ){
            super(threshold,n,x,y,z,offsetX,offsetY,offsetZ,incrX,incrY,incrZ);
        }

        @Override
        public void doTask() {
            //Task: Z = X+Y
            if(x == z){
                //can use axpy
                Nd4j.getBlasWrapper().level1().axpy(n,-1.0,y,offsetY,incrY,z,offsetZ,incrZ);
            } else {
                //use loop
                if(x.dataType() == DataBuffer.Type.FLOAT){
                    float[] xf = (float[])x.array();
                    float[] yf = (float[])y.array();
                    float[] zf = (float[])z.array();
                    if(incrX == 1 && incrY == 1 && incrZ == 1) {
                        for (int i = 0; i < n; i++) {
                            zf[offsetZ+i] = xf[offsetX+i] - yf[offsetY+i];
                        }
                    } else {
                        for( int i=0; i<n; i++){
                            zf[offsetZ+i*incrZ] = xf[offsetX+i*incrX] - yf[offsetY+i*incrY];
                        }
                    }
                } else {
                    double[] xd = (double[])x.array();
                    double[] yd = (double[])y.array();
                    double[] zd = (double[])z.array();
                    if(incrX == 1 && incrY == 1 && incrZ == 1) {
                        for (int i = 0; i < n; i++) {
                            zd[offsetZ+i] = xd[offsetX+i] - yd[offsetY+i];
                        }
                    } else {
                        for( int i=0; i<n; i++){
                            zd[offsetZ+i*incrZ] = xd[offsetX+i*incrX] - yd[offsetY+i*incrY];
                        }
                    }
                }
            }
        }

        @Override
        public BaseDataBufferTask getSubTask(int threshold, int n, DataBuffer x, DataBuffer y, DataBuffer z, int offsetX, int offsetY, int offsetZ, int incrX, int incrY, int incrZ) {
            return new SubOpDataBufferTask(threshold,n,x,y,z,offsetX,offsetY,offsetZ,incrX,incrY,incrZ);
        }
    }


    public static class MulOpDataBufferTask extends BaseDataBufferTask {
        public MulOpDataBufferTask(int threshold, int n, DataBuffer x, DataBuffer y, DataBuffer z, int offsetX, int offsetY, int offsetZ, int incrX, int incrY, int incrZ){
            super(threshold,n,x,y,z,offsetX,offsetY,offsetZ,incrX,incrY,incrZ);
        }

        @Override
        public void doTask() {
            //Task: Z = X*Y

            if(x.dataType() == DataBuffer.Type.FLOAT){
                float[] xf = (float[])x.array();
                float[] yf = (float[])y.array();
                if(incrX == 1 && incrY == 1 && incrZ == 1) {
                    if(x==z){
                        for (int i = 0; i < n; i++) {
                            xf[offsetX + i] *= yf[offsetY + i];
                        }
                    } else {
                        float[] zf = (float[])z.array();
                        for (int i = 0; i < n; i++) {
                            zf[offsetZ + i] = xf[offsetX + i] * yf[offsetY + i];
                        }
                    }
                } else {
                    if(x==z){
                        for (int i = 0; i < n; i++) {
                            xf[offsetX + i * incrX] *= yf[offsetY + i * incrY];
                        }
                    } else {
                        float[] zf = (float[])z.array();
                        for (int i = 0; i < n; i++) {
                            zf[offsetZ + i * incrZ] = xf[offsetX + i * incrX] * yf[offsetY + i * incrY];
                        }
                    }
                }
            } else {
                double[] xd = (double[])x.array();
                double[] yd = (double[])y.array();
                if(incrX == 1 && incrY == 1 && incrZ == 1) {
                    if(x==z){
                        for (int i = 0; i < n; i++) {
                            xd[offsetX + i] *= yd[offsetY + i];
                        }
                    } else {
                        double[] zd = (double[])z.array();
                        for (int i = 0; i < n; i++) {
                            zd[offsetZ + i] = xd[offsetX + i] * yd[offsetY + i];
                        }
                    }
                } else {
                    if(x==z) {
                        for (int i = 0; i < n; i++) {
                            xd[offsetX + i * incrX] *= yd[offsetY + i * incrY];
                        }
                    } else {
                        double[] zd = (double[])z.array();
                        for (int i = 0; i < n; i++) {
                            zd[offsetZ + i * incrZ] = xd[offsetX + i * incrX] * yd[offsetY + i * incrY];
                        }
                    }
                }
            }
        }

        @Override
        public BaseDataBufferTask getSubTask(int threshold, int n, DataBuffer x, DataBuffer y, DataBuffer z, int offsetX, int offsetY, int offsetZ, int incrX, int incrY, int incrZ) {
            return new MulOpDataBufferTask(threshold,n,x,y,z,offsetX,offsetY,offsetZ,incrX,incrY,incrZ);
        }
    }

    public static class DivOpDataBufferTask extends BaseDataBufferTask {
        public DivOpDataBufferTask(int threshold, int n, DataBuffer x, DataBuffer y, DataBuffer z, int offsetX, int offsetY, int offsetZ, int incrX, int incrY, int incrZ){
            super(threshold,n,x,y,z,offsetX,offsetY,offsetZ,incrX,incrY,incrZ);
        }

        @Override
        public void doTask() {
            //Task: Z = X/Y

            if(x.dataType() == DataBuffer.Type.FLOAT){
                float[] xf = (float[])x.array();
                float[] yf = (float[])y.array();
                if(incrX == 1 && incrY == 1 && incrZ == 1) {
                    if(x==z){
                        for (int i = 0; i < n; i++) {
                            xf[offsetX + i] /= yf[offsetY + i];
                        }
                    } else {
                        float[] zf = (float[])z.array();
                        for (int i = 0; i < n; i++) {
                            zf[offsetZ + i] = xf[offsetX + i] / yf[offsetY + i];
                        }
                    }
                } else {
                    if(x==z){
                        for (int i = 0; i < n; i++) {
                            xf[offsetX + i * incrX] /= yf[offsetY + i * incrY];
                        }
                    } else {
                        float[] zf = (float[])z.array();
                        for (int i = 0; i < n; i++) {
                            zf[offsetZ + i * incrZ] = xf[offsetX + i * incrX] / yf[offsetY + i * incrY];
                        }
                    }
                }
            } else {
                double[] xd = (double[])x.array();
                double[] yd = (double[])y.array();
                if(incrX == 1 && incrY == 1 && incrZ == 1) {
                    if(x==z){
                        for (int i = 0; i < n; i++) {
                            xd[offsetX + i] /= yd[offsetY + i];
                        }
                    } else {
                        double[] zd = (double[])z.array();
                        for (int i = 0; i < n; i++) {
                            zd[offsetZ + i] = xd[offsetX + i] / yd[offsetY + i];
                        }
                    }
                } else {
                    if(x==z) {
                        for (int i = 0; i < n; i++) {
                            xd[offsetX + i * incrX] /= yd[offsetY + i * incrY];
                        }
                    } else {
                        double[] zd = (double[])z.array();
                        for (int i = 0; i < n; i++) {
                            zd[offsetZ + i * incrZ] = xd[offsetX + i * incrX] / yd[offsetY + i * incrY];
                        }
                    }
                }
            }
        }

        @Override
        public BaseDataBufferTask getSubTask(int threshold, int n, DataBuffer x, DataBuffer y, DataBuffer z, int offsetX, int offsetY, int offsetZ, int incrX, int incrY, int incrZ) {
            return new DivOpDataBufferTask(threshold,n,x,y,z,offsetX,offsetY,offsetZ,incrX,incrY,incrZ);
        }
    }

    public static class CopyOpDataBufferTask extends BaseDataBufferTask {
        public CopyOpDataBufferTask(int threshold, int n, DataBuffer x, DataBuffer y, DataBuffer z, int offsetX, int offsetY, int offsetZ, int incrX, int incrY, int incrZ){
            super(threshold,n,x,y,z,offsetX,offsetY,offsetZ,incrX,incrY,incrZ);
        }

        @Override
        public void doTask() {
            //Task: Z = X
            if(x==z) return;    //No op
            Nd4j.getBlasWrapper().level1().copy(n,x,offsetX,incrX,z,offsetZ,incrZ);
        }

        @Override
        public BaseDataBufferTask getSubTask(int threshold, int n, DataBuffer x, DataBuffer y, DataBuffer z, int offsetX, int offsetY, int offsetZ, int incrX, int incrY, int incrZ) {
            return new CopyOpDataBufferTask(threshold,n,x,y,z,offsetX,offsetY,offsetZ,incrX,incrY,incrZ);
        }
    }

    public static class OpDataBufferTask extends BaseDataBufferTask {
        private final Op op;
        public OpDataBufferTask(Op op, int threshold, int n, DataBuffer x, DataBuffer y, DataBuffer z, int offsetX, int offsetY, int offsetZ, int incrX, int incrY, int incrZ){
            super(threshold,n,x,y,z,offsetX,offsetY,offsetZ,incrX,incrY,incrZ);
            this.op = op;
        }

        @Override
        public void doTask() {
            //Task: Z = X/Y

            if(x.dataType() == DataBuffer.Type.FLOAT){
                float[] xf = (float[])x.array();
                float[] yf = (float[])y.array();
                if(incrX == 1 && incrY == 1 && incrZ == 1) {
                    if(x==z){
                        for (int i = 0; i < n; i++) {
                            int xIdx = offsetX + i;
                            xf[xIdx] = op.op(xf[xIdx],yf[offsetY + i]);
                        }
                    } else {
                        float[] zf = (float[])z.array();
                        for (int i = 0; i < n; i++) {

                            zf[offsetZ + i] = op.op(xf[offsetX + i], yf[offsetY + i]);
                        }
                    }
                } else {
                    if(x==z){
                        for (int i = 0; i < n; i++) {
                            int xIdx = offsetX + i * incrX;
                            xf[xIdx] = op.op(xf[xIdx], yf[offsetY + i * incrY]);
                        }
                    } else {
                        float[] zf = (float[])z.array();
                        for (int i = 0; i < n; i++) {
                            zf[offsetZ + i * incrZ] = op.op(xf[offsetX + i * incrX], yf[offsetY + i * incrY]);
                        }
                    }
                }
            } else {
                double[] xd = (double[])x.array();
                double[] yd = (double[])y.array();
                if(incrX == 1 && incrY == 1 && incrZ == 1) {
                    if(x==z){
                        for (int i = 0; i < n; i++) {
                            int xIdx = offsetX+i;
                            xd[xIdx] = op.op(xd[xIdx], yd[offsetY + i]);
                        }
                    } else {
                        double[] zd = (double[])z.array();
                        for (int i = 0; i < n; i++) {
                            zd[offsetZ + i] = op.op(xd[offsetX + i], yd[offsetY + i]);
                        }
                    }
                } else {
                    if(x==z) {
                        for (int i = 0; i < n; i++) {
                            int xIdx = offsetX + i * incrX;
                            xd[xIdx] = op.op(xd[xIdx], yd[offsetY + i * incrY]);
                        }
                    } else {
                        double[] zd = (double[])z.array();
                        for (int i = 0; i < n; i++) {
                            zd[offsetZ + i * incrZ] = op.op(xd[offsetX + i * incrX],yd[offsetY + i * incrY]);
                        }
                    }
                }
            }
        }

        @Override
        public BaseDataBufferTask getSubTask(int threshold, int n, DataBuffer x, DataBuffer y, DataBuffer z, int offsetX, int offsetY, int offsetZ, int incrX, int incrY, int incrZ) {
            return new OpDataBufferTask(op,threshold,n,x,y,z,offsetX,offsetY,offsetZ,incrX,incrY,incrZ);
        }
    }


}
