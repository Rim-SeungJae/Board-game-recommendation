$(document).ready(function(){
  $("#submit").click(function(){
    let value1 = $('#id_search_word').val();
    if(!value1)
    {
      $('#id_search_word').css("border-color", "red")
    }
    else{
      $('#id_search_word').css("border-color", "#ced4da")
    }
  })
});
