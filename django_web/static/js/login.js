$(document).ready(function(){
  $("#submit").click(function(){
    let value1 = $('#id_username').val();
    let value2 = $('#id_password').val();
    if(!value1)
    {
      $('#id_username').css("border-color", "red")
    }
    else{
      $('#id_username').css("border-color", "#ced4da")
    }
    if(!value2)
    {
      $('#id_password').css("border-color", "red")
    }
    else{
      $('#id_password').css("border-color", "#ced4da")
    }
  })
});
