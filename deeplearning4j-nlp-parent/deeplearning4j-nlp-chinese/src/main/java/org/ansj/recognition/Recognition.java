package org.ansj.recognition;

import org.ansj.domain.Result;

import java.io.Serializable;

/**
 * 词语结果识别接口,用来通过规则方式识别词语,对结果的二次加工
 * 
 * @author Ansj
 *
 */
public interface Recognition extends Serializable {
    public void recognition(Result result);
}
