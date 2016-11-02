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
    var layerNames = data["layerNames"];    //List<String>
    var layerTypes = data["layerTypes"];    //List<String>
    var layerInputs = data["layerInputs"];  //int[][]
    var layerInfos = data["layerInfo"];     //List<Map<String,String>>

    var nodes = [];
    var edges = [];
    for(var i=0; i<layerNames.length; i++ ){
        var obj = {
            id: i,
            name: layerNames[i],
            faveColor: '#6FB1FC',   //TODO
            faveShape: 'triangle'   //TODO
        };
        nodes.push({ data: obj} );

        //Edges:
        var inputsToCurrent = layerInputs[i];
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
                'width': 'mapData(weight, 40, 80, 20, 60)',
                'content': 'data(name)',
                'text-valign': 'center',
                'text-outline-width': 2,
                'text-outline-color': 'data(faveColor)',
                'background-color': 'data(faveColor)',
                'color': '#fff'
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
            cy.panningEnabled(true);
            cy.autoungrabify(true);
            cy.maxZoom(5);
            cy.minZoom(1);
            // cy.wheelSensitivity(0);
        }
    });

    // cy.on('tap', 'node', function () {
    //     window.location.href = this.data('href');
    // });

    cy.on('click', 'node', function (evt) {
        console.log('CLICKED: id=' + this.id() + ", name=" + this.data.name);
        console.log('CLICKED: id=' + this.id() + ", name=" + this.name);
    });

}