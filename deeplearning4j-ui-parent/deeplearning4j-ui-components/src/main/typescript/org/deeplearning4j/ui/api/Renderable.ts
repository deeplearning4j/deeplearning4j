/// <reference path="../typedefs/jquery.d.ts" />

interface Renderable {

    render: (addToObject: JQuery) => void;

    //TODO: Implement an update method. Can be used to update an existing chart, table etc with new data,
    // without redrawing the whole thing from scratch
    //update: (jsonObj: any) => void;
}