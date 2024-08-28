$(document).ready(function(){
  $("#submit").click(function(){
    let value1 = $('#id_username').val();
    let value2 = $('#id_password1').val();
    let value3 = $('#id_password2').val();
    if(!value1)
    {
      $('#id_username').css("border-color", "red")
    }
    else{
      $('#id_username').css("border-color", "#ced4da")
    }
    if(!value2)
    {
      $('#id_password1').css("border-color", "red")
    }
    else{
      $('#id_password1').css("border-color", "#ced4da")
    }
    if(!value3)
    {
      $('#id_password2').css("border-color", "red")
    }
    else{
      $('#id_password2').css("border-color", "#ced4da")
    }
  })
});
