CURRENT_HREF = window.location.href
PATHNAME = window.location.pathname
var urlpath  = CURRENT_HREF.replace(PATHNAME,'');
URL_ROOT = urlpath


function get_input_text(id_input){
    var text = document.getElementById(id_input).value
    if (text == "") {
        alert("please input text in textarea")
    }
    return text
}
function get_result_predict(text){
    id_input = 'input-text'
    text = get_input_text(id_input)
    // console.log(text)
    predict(text).done(function(r){
        var result = r['result'] 
        var status = 'negative'
        if(result> 0.5){
            status = 'positive'
        }
        // console.log(result)
      
        var html = '<div class="text-light small fw-semibold mb-1">'+status+'</div> <div class="progress" style="height: 6px;"><div class="progress-bar" role="progressbar" style="width: '+result*100+'%;" aria-valuenow="60" aria-valuemin="0" aria-valuemax="100"></div></div> '
        document.getElementById("resultsId").innerHTML = html; 

    }
    );
}
function predict(text) {
    var content = { "text": text}
    var api = URL_ROOT + "/sentiment-analyst/predict"
    var type="POST"
    return ajax_api(content,api,  type)

}



function ajax_api(content, path, type){
    return $.ajax({
        url: path,
        data: 
            JSON.stringify(
                content
            ),
        type: type,
        contentType: "application/json; charset=utf-8",
    });
}


