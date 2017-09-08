package org.deeplearning4j.spark.datavec.iterator;

import lombok.AllArgsConstructor;
import lombok.Data;
import org.datavec.api.writable.Writable;

import java.io.Serializable;
import java.util.List;

@AllArgsConstructor
@Data
public class DataVecRecord implements Serializable {
    private int readerIdx;
    private List<Writable> record;
    private List<List<Writable>> seqRecord;
}
