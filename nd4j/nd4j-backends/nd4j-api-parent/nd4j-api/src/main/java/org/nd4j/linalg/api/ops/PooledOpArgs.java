package org.nd4j.linalg.api.ops;

import org.nd4j.linalg.api.ndarray.INDArray;

public class PooledOpArgs {
    protected INDArray x, y, z;
    protected Object[] extraArgs;
    private int argsCount;

    public PooledOpArgs() {
        argsCount = 0;
    }

    public void setX(INDArray x) {
        this.x = x;
        ++argsCount;
    }

    public void setY(INDArray y) {
        this.y = y;
        ++argsCount;
    }

    public void setZ(INDArray z) {
        this.z  = z;
        ++argsCount;
    }

    public int getArgsCount() {
        return argsCount;
    }
}
