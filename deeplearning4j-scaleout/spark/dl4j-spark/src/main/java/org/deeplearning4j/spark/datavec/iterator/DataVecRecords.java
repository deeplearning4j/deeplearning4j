package org.deeplearning4j.spark.datavec.iterator;

import lombok.AllArgsConstructor;
import lombok.Data;
import org.datavec.api.writable.Writable;

import java.io.Serializable;
import java.util.List;

@AllArgsConstructor
@Data
public class DataVecRecords implements Serializable {
    private List<List<Writable>> records;
    private List<List<List<Writable>>> seqRecords;
}
