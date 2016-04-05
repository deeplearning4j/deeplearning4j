/// <reference path="../typedefs/jquery.d.ts" />
/// <reference path="Style.ts" />
/// <reference path="ComponentType.ts" />


abstract class Component {

    private componentType: ComponentType;

    constructor(componentType: ComponentType){
        this.componentType = componentType;
    }

    public getComponentType(){
        return this.componentType;
    }
}






