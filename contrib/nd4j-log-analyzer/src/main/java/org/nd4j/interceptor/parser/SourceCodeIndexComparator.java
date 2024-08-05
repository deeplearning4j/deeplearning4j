package org.nd4j.interceptor.parser;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;
import lombok.AllArgsConstructor;
import org.nd4j.shade.jackson.databind.ObjectMapper;
import org.nd4j.shade.jackson.databind.SerializationFeature;
import org.nd4j.shade.jackson.databind.annotation.JsonDeserialize;
import org.nd4j.shade.jackson.databind.annotation.JsonSerialize;

import java.io.File;
import java.io.IOException;
import java.util.HashMap;
import java.util.Map;

@Data
@Builder
@NoArgsConstructor
@AllArgsConstructor
@JsonSerialize(using = SourceCodeIndexComparatorSerializer.class)
@JsonDeserialize(using = SourceCodeIndexComparatorDeserializer.class)

public class SourceCodeIndexComparator {
    private SourceCodeIndexer index1;
    private SourceCodeIndexer index2;
    private ObjectMapper objectMapper;
    private Map<SourceCodeLine, SourceCodeLine> comparisonResult;
    private Map<SourceCodeLine, SourceCodeLine> reverseComparisonResult;

    public SourceCodeIndexComparator(File indexFile1, File indexFile2) throws IOException {
        objectMapper = new ObjectMapper().enable(SerializationFeature.INDENT_OUTPUT);
        this.index1 = objectMapper.readValue(indexFile1, SourceCodeIndexer.class);
        this.index2 = objectMapper.readValue(indexFile2, SourceCodeIndexer.class);
    }

    public void compareIndexes() {
        comparisonResult = new HashMap<>();
        reverseComparisonResult = new HashMap<>();

        for (String className : index1.getIndex().rowKeySet()) {
            for (Integer lineNumber : index1.getIndex().columnKeySet()) {
                SourceCodeLine line1 = index1.getSourceCodeLine(className, lineNumber);
                SourceCodeLine line2 = index2.getSourceCodeLine(className,lineNumber);

                if (line2 != null && line1.getLine().equals(line2.getLine())) {
                    comparisonResult.put(line1, line2);
                    reverseComparisonResult.put(line2, line1);
                }
            }
        }
    }

    public void saveComparisonResult(String fileName) {
        try {
            objectMapper.writeValue(new File(fileName), comparisonResult);
        } catch (IOException e) {
            e.printStackTrace();
        }
    }
}