
/* show uploaded image */
function readURL(input) {
    if (input.files && input.files[0]) {
        var reader = new FileReader();
        reader.onload = function (e) {
            $('#imageResult')
                .attr('src', e.target.result);
        };
        reader.readAsDataURL(input.files[0]);
    }
}

$(function () {
    $('#upload').on('change', function (e) {
        readURL(input);
        var form_data = new FormData();
        form_data.append('file', $('#upload').prop('files')[0]);
        /* reset result text */
        document.getElementById('result').innerHTML = "";
        $.ajax({
            type: 'POST',
            url: '/predict/',
            data: form_data,
            contentType: false,
            cache: false,
            processData: false,
            success: function(response) {
                showResult(response);
            },
        });
    });
});


/* use example image */

$(function () {
    $('#use-example').on('click', function () {
        $('#imageResult').attr('src', '/static/img/example.png');
        /* reset label and result text */
        var infoArea = document.getElementById( 'upload-label' );
        infoArea.textContent = 'Choose file';
        document.getElementById('result').innerHTML = "";
        $.ajax({
            type: 'POST',
            url: '/example/',
            data: null,
            success: function (response) {
             showResult(response);
            },
          });
    });
});


/* show uploaded image name */
var input = document.getElementById( 'upload' );
var infoArea = document.getElementById( 'upload-label' );

input.addEventListener( 'change', showFileName );
function showFileName( event ) {
  var input = event.srcElement;
  var fileName = input.files[0].name;
  infoArea.textContent = 'File name: ' + fileName;
}


/* show result text */
function showResult(result) {
    var text = "";
    if(result == 'Positive') {
        text = 'Predicted result: <em class="positive-result"> Positive </em>'
    } else {
        text = 'Predicted result: <em class="negative-result"> Negative </em>'
    }
    document.getElementById('result').innerHTML = text;
}
