package org.nd4j.autodiff.execution;

import lombok.Data;
import lombok.NoArgsConstructor;
import lombok.ToString;
import lombok.extern.slf4j.Slf4j;

import java.util.ArrayList;
import java.util.List;

/**
 * Intermediate Node representation
 *
 * @author raver119@gmail.com
 */
@Data
@Slf4j
@NoArgsConstructor
@ToString(exclude = {"opExecAction"})
public class Node {
    private int id;
    private String name;
    private List<Integer> input = new ArrayList<>();
    private List<Integer> output = new ArrayList<>();
    private List<Integer> unresolved = new ArrayList<>();
    private int[] originalOutput;
}
