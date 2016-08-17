package org.nd4j.linalg.jcublas.ops.executioner;

import org.nd4j.linalg.api.ops.Op;
import org.nd4j.linalg.api.ops.executioner.DefaultOpExecutioner;
import org.nd4j.linalg.api.ops.executioner.OpExecutioner;

import java.util.ArrayList;
import java.util.Deque;
import java.util.List;

/**
 * mGRID implementation for OpExecutioner interface
 *
 * PLEASE NOTE:  WORK IN PROGRESS, DO NOT EVER USE THIS EXECUTIONER IN PRODUCTION
 * @author raver119@gmail.com
 */
public class GridExecutioner extends DefaultOpExecutioner {

    List<Deque<Op>> deviceQueues = new ArrayList<>();
}
