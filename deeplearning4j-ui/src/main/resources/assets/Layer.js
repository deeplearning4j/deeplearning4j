/*
    This is js-side Layer definition
*/

function Layer(object) {
    this.id = object.id;
    this.name = object.name;
    this.layerType = object.layerType;
    this.x = object.x;
    this.y = object.y;

    // text properties

    this.mainLine = object.description.mainLine;
    this.subLine = object.description.subLine;

    // now we parse connections
}