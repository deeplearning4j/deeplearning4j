package org.nd4j.autodiff.opstate;

import lombok.Builder;
import lombok.Data;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.weightinit.WeightInit;
import org.nd4j.weightinit.WeightInitScheme;
import org.nd4j.weightinit.impl.ZeroInitScheme;

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
    @Builder.Default
    private WeightInitScheme weightInitScheme = new ZeroInitScheme('f');




    /**
     * Create appropriate
     * {@link NDArrayInformation}
     * given the ndarray
     * including the shape
     * and a new id
     * @param shape  the shape of the array
     *  @param weightInitScheme the init scheme to use
     * @return
     */
    public static NDArrayInformation newInfo(int[] shape,WeightInitScheme weightInitScheme) {
        String id = UUID.randomUUID().toString();
        return NDArrayInformation.builder()
                .shape(shape).weightInitScheme(weightInitScheme)
                .arrId(id)
                .id(id)
                .build();
    }



    /**
     * Create appropriate
     * {@link NDArrayInformation}
     * given the ndarray
     * including the shape
     * and a new id
     * @return
     */
    public static NDArrayInformation newInfo(int[] shape) {
        return newInfo(shape,new ZeroInitScheme('f'));
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
            return null;
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
        if (arrId != null ? !arrId.equals(that.arrId) : that.arrId != null) return false;
        return weightInitScheme != null ? weightInitScheme.equals(that.weightInitScheme) : that.weightInitScheme == null;
    }

    @Override
    public int hashCode() {
        int result = super.hashCode();
        result = 31 * result + Arrays.hashCode(shape);
        result = 31 * result + (id != null ? id.hashCode() : 0);
        result = 31 * result + (scalarValue != null ? scalarValue.hashCode() : 0);
        result = 31 * result + (arrId != null ? arrId.hashCode() : 0);
        result = 31 * result + (weightInitScheme != null ? weightInitScheme.hashCode() : 0);
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
