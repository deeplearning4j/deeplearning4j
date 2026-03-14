package org.nd4j.linalg.dataset.curation.filtering;

import java.io.Serializable;
import java.util.Collections;
import java.util.List;

/**
 * Result of text quality filtering, containing acceptance status and rejection reasons.
 */
public class FilterResult implements Serializable {
    private static final long serialVersionUID = 1L;

    private final boolean accepted;
    private final List<String> rejectionReasons;

    private FilterResult(boolean accepted, List<String> rejectionReasons) {
        this.accepted = accepted;
        this.rejectionReasons = rejectionReasons;
    }

    public static FilterResult accept() {
        return new FilterResult(true, Collections.emptyList());
    }

    public static FilterResult reject(String reason) {
        return new FilterResult(false, Collections.singletonList(reason));
    }

    public static FilterResult reject(List<String> reasons) {
        return new FilterResult(false, Collections.unmodifiableList(reasons));
    }

    public boolean isAccepted() {
        return accepted;
    }

    public List<String> getRejectionReasons() {
        return rejectionReasons;
    }

    @Override
    public String toString() {
        if (accepted) return "FilterResult{ACCEPTED}";
        return "FilterResult{REJECTED: " + rejectionReasons + "}";
    }
}
