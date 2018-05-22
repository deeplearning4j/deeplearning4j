package vizdoom;

public class Label{
    public int objectId;
    public String objectName;
    public byte value;
    public double objectPositionX;
    public double objectPositionY;
    public double objectPositionZ;

    Label(int id, String name, byte value, double positionX, double positionY, double positionZ){
        this.objectId = objectId;
        this.objectName = objectName;
        this.value = value;
        this.objectPositionX = positionX;
        this.objectPositionY = positionY;
        this.objectPositionZ = positionZ;
    }
}
