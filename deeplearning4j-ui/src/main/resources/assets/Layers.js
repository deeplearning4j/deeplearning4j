function Layers() {
    this.layers = [];
    this.totalLayers = 0;
    this.maximumX = 0;
    this.maximumY = 0;


    this.attach = function( layer) {
        this.layers.push(layer);
        this.totalLayers++;
        if (this.maximumX < layer.x) this.maximumX = layer.x;
        if (this.maximumY < layer.y) this.maximumY = layer.y;
    }

    this.getLayersForY = function(y) {
        var result = [];
            for (var i = 0; i < this.layers.length; i++) {
                if (this.layers[i].y == y) {
                    result.push(this.layers[i]);
                }
            }
         return result;
    }

    this.getLayersForX = function (x) {
        var result = [];
        for (var i = 0; i < this.layers.length; i++) {
            if (this.layers[i].x == x) {
                result.push(this.layers[i]);
            }
        }
        return result;
    }

    this.getLayerForXY = function (x, y) {
        var layerX = this.getLayersForX(x);
        for (var i = 0; i < layerX.length; i++) {
            if (layerX[i].y == y) return layerX[i];
        }
        return null;
    }

    this.getLayerForYX = function (x, y) {
            var layerY = this.getLayersForY(y);
            for (var i = 0; i < layerY.length; i++) {
                if (layerY[i].x == x) return layerY[i];
            }
            return null;
        }
}