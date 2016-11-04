function renderModelGraph(){
    $.ajax({
        url: "/train/model/graph",
        async: true,
        error: function (query, status, error) {
            console.log("Error getting data: " + error);
        },
        success: function (data) {
            console.log("Keys: " + Object.keys(data));

            createGraph(data);
        }
    });
}


function createGraph(data){

    //Generate the elements data
    var vertexNames = data["vertexNames"];    //List<String>
    var vertexTypes = data["vertexTypes"];    //List<String>
    var vertexInputs = data["vertexInputs"];  //int[][]
    var vertexInfos = data["vertexInfo"];     //List<Map<String,String>>
    var vertexCount = vertexNames.length;

    var nodes = [];
    var edges = [];
    for(var i=0; i<vertexNames.length; i++ ){
        var obj = {
            id: i,
            name: vertexTypes[i] + '\n(' + vertexNames[i] +')',
            faveColor: '#6FB1FC',   //TODO
            faveShape: 'triangle',   //TODO
            onclick: "renderLayerTable()"
        };
        nodes.push({ data: obj} );

        //Edges:
        var inputsToCurrent = vertexInputs[i];
        for(var j=0; j<inputsToCurrent.length; j++ ){
            var e = {
                source: inputsToCurrent[j],
                target: i,
                faveColor: '#A9A9A9',   //TODO
                strength: 100
            };
            edges.push({ data: e} );
        }
    }

    var elementsToRender = {
        nodes: nodes,
        edges: edges
    };


    $('#layers').cytoscape({
        layout: {
            name: 'dagre',
            padding: 10
        },

        style: cytoscape.stylesheet()
            .selector('node')
            .css({
                'shape': 'data(faveShape)',
                'width': '100',
                'height': '50',
                'content': 'data(name)',
                'text-valign': 'center',
                'text-outline-width': 2,
                'text-outline-color': 'data(faveColor)',
                'background-color': 'data(faveColor)',
                'color': '#fff',
                'text-wrap': 'wrap',
                'font-size': '17px'
            })
            .selector(':selected')
            .css({
                'border-width': 3,
                'border-color': '#333'
            })
            .selector('edge')
            .css({
                'curve-style': 'bezier',
                'opacity': 0.666,
                'width': 'mapData(strength, 70, 100, 2, 6)',
                'target-arrow-shape': 'triangle',
                'source-arrow-shape': 'circle',
                'line-color': 'data(faveColor)',
                'source-arrow-color': 'data(faveColor)',
                'target-arrow-color': 'data(faveColor)'
            })
            .selector('edge.questionable')
            .css({
                'line-style': 'dotted',
                'target-arrow-shape': 'diamond'
            })
            .selector('.faded')
            .css({
                'opacity': 0.25,
                'text-opacity': 0
            }),

        elements: elementsToRender,

        ready: function () {
            window.cy = this;
            if (vertexCount <= 4) {
                cy.zoom(1.6);
                cy.center();
            } else if (vertexCount > 4 && vertexCount <=6) {
                cy.zoom(1);
                cy.center();
            }
            else {
                cy.zoom(1);
                cy.panBy({x: -50, y:0});
            }
            cy.panningEnabled(true);
            cy.autoungrabify(true);
            cy.zoomingEnabled(false);
        }
    });

    cy.on('click', 'node', function (evt) {
        setSelectedVertex(this.id());
    });

}