
function languageSelect(languageCode, redirect){
    //language code: iso639 code

    $.ajax({
        url: "/setlang/" + languageCode,
        async: true,
        error: function (query, status, error) {
            console.log("Error getting data: " + error);
        },
        success: function (data) {
            window.location.replace('/train/' + redirect)
        }
    });
}


function formatBytes(bytes, precision){
    var index = 0;
    var newValue = bytes;
    while(newValue > 1024){
        newValue = newValue/1024;
        index++;
    }

    var unit = "";
    switch (index){
        case 0:
            unit = "B";
            break;
        case 1:
            unit = "kB";
            break;
        case 2:
            unit = "MB";
            break;
        case 3:
            unit = "GB";
            break;
        case 4:
            unit = "TB";
            break;
    }

    return newValue.toFixed(precision) + " " + unit;
}

/* ---------- Language Dropdown ---------- */

	$('.dropmenu').click(function(e){
		e.preventDefault();
		$(this).parent().find('ul').slideToggle();
	});