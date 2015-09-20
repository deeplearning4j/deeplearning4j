package org.nd4j.linalg.api.shape.loop.two;

import org.nd4j.linalg.api.buffer.DataBuffer;

/**
 * Created by agibsonccc on 9/14/15.
 */
public class CopyLoopFunction implements LoopFunction2 {
    @Override
    public void perform(int i,RawArrayIterationInformation2 info, DataBuffer a, int aOffset, DataBuffer b, int bOffset) {
        a.put(aOffset, b.getDouble(bOffset));
    }
}
