/*
    This is js-side Layer definition
*/

function Layer(object) {
    this.id = parseInt(object.id);
    this.name = object.name;
    this.layerType = object.layerType;
    this.x = parseInt(object.x);
    this.y = parseInt(object.y);

    // text properties

    this.mainLine = object.description.mainLine;
    this.subLine = object.description.subLine;

    // now we parse connections
    this.connections = [];
    for (var i = 0; i < object.connections.length; i++) {
        var connection = new Connection(object.connections[i]);
        this.connections.push(connection);
    }
}