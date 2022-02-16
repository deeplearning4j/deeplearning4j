package org.nd4j.autodiff.samediff;

import org.nd4j.autodiff.samediff.internal.InferenceSession;

public interface InferenceFactory {

    InferenceSession create(SameDiff sameDiff);
}