package org.nd4j.nativeblas;

import org.bytedeco.javacpp.Pointer;

public abstract class ResultWrapperAbstraction extends Pointer {

    public ResultWrapperAbstraction(Pointer p) {
        super(p);
    }

    public abstract long size();

    public abstract Pointer pointer();
}
