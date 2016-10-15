function onNavClick(source, errorMsg) {
    console.log("Clicked: " + source);
    var reqURL = "train/" + source;
    $.ajax({
        url: reqURL,
        success: function (data) {
            $("#mainContentDiv").html(data);
        },
        error: function (query, status, error) {
            $.notify(errorMsg, "error");
        }
    });
}
//# sourceMappingURL=dl4j-play-ui.js.map