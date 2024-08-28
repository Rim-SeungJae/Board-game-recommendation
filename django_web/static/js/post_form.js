$(document).ready(function(){
  $("#submit").click(function(){
    let value1 = $('#id_title').val();
    let value2 = $('#id_content').val();
    if(!value1)
    {
      $('#id_title').css("border-color", "red")
    }
    else{
      $('#id_title').css("border-color", "#ced4da")
    }
    if(!value2)
    {
      $('#id_content').css("border-color", "red")
    }
    else{
      $('#id_content').css("border-color", "#ced4da")
    }
  })
});
