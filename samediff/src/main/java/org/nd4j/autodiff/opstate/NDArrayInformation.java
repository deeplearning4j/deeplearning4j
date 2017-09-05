package org.nd4j.autodiff.opstate;

import lombok.Builder;
import lombok.Data;
import lombok.EqualsAndHashCode;
import org.nd4j.autodiff.ArrayField;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.io.Serializable;
import java.util.Arrays;
import java.util.UUID;

/**
 * Created by agibsonccc on 4/6/17.
 */
@Data()
@Builder
public class NDArrayInformation implements Serializable {
    private int[] shape;
    private String id;
    private OpState owner;
    private Number scalarValue;
    private String arrId;

    /**
     * Create appropriate
     * {@link NDArrayInformation}
     * given the ndarray
     * including the shape
     * and a new id
     * @param arr the input array
     * @return
     */
    public static NDArrayInformation newInfo(INDArray arr) {
        return NDArrayInformation.builder()
                .shape(arr.shape())
                .arrId(UUID.randomUUID().toString())
                .scalarValue(arr.isScalar() ? arr.getDouble(0) : null)
                .build();
    }

    /**
     * Get the arr id (possibly generating a
     * new one lazily)
     * @return
     */
    public String getArrId() {
        if(arrId == null)
            arrId = UUID.randomUUID().toString();
        return arrId;

    }

    /**
     * Return the scalar for this
     * {@link NDArrayInformation}
     * which is either the field itself,
     * or if that's null, the owner's
     * scalar value
     * @return
     */
    public Number scalar() {
        if(scalarValue != null)
            return scalarValue;

        if(owner == null)
            throw new IllegalStateException("No owner set.");
        return owner.getScalarValue();
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;

        NDArrayInformation that = (NDArrayInformation) o;

        if (!Arrays.equals(shape, that.shape)) return false;
        if (id != null ? !id.equals(that.id) : that.id != null) return false;
        if (scalarValue != null ? !scalarValue.equals(that.scalarValue) : that.scalarValue != null) return false;
        return arrId != null ? arrId.equals(that.arrId) : that.arrId == null;
    }

    @Override
    public int hashCode() {
        int result = super.hashCode();
        result = 31 * result + Arrays.hashCode(shape);
        result = 31 * result + (id != null ? id.hashCode() : 0);
        result = 31 * result + (owner != null ? owner.toString().hashCode() : 0);
        result = 31 * result + (scalarValue != null ? scalarValue.hashCode() : 0);
        result = 31 * result + (arrId != null ? arrId.hashCode() : 0);
        return result;
    }

    @Override
    public String toString() {
        return "NDArrayInformation{" +
                "shape=" + Arrays.toString(shape) +
                ", id='" + id + '\'' +
                '}';
    }
}
