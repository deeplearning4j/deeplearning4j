
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
