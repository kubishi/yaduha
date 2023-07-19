// on btn-translator click, get input-translator-english value and sent to /api/translator/translate
$(document).ready(function() {
    $("#btn-translator").on("click", function() {
        // disable button and add a bootstrap spinner
        $("#btn-translator").prop("disabled", true);
        $("#btn-translator").html('<span class="spinner-border spinner-border-sm" role="status" aria-hidden="true"></span> Loading...');
        
        $("#output-translator-english").text("");
        $("#output-translator-paiute").text("");
        $("#translation-section").hide();

        var english = $("#input-translator-english").val();
        fetch("/api/translator/translate", {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify({
                english: english
            })
        }).then(res => res.json()).then(response => {
            $("#output-translator-english").text(response.english);
            $("#output-translator-paiute").text(response.paiute);
            $("#translation-section").show();
        }).catch(error => {
            console.log(error);
        }).finally(() => {
            // re-enable button and remove bootstrap spinner
            $("#btn-translator").prop("disabled", false);
            $("#btn-translator").html("Translate");
        });
    });
    
});