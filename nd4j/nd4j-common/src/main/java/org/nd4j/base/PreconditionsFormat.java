package org.nd4j.base;

import java.util.List;

public interface PreconditionsFormat {

    List<String> formatTags();

    String format(String tag, Object arg);

}
