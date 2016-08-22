// var fs = require('fs');
// var midiConverter = require('midi-converter');

// function midi_json(){
//   var midiSong = fs.readFileSync('example.mid', 'binary');
//   var jsonSong = midiConverter.midiToJson(midiSong);
//   fs.writeFileSync('example.json', JSON.stringify(jsonSong));
// }

/*
  Creating a button where you can upload your own file to the model
*/
function handleFile(){
  input = document.getElementById('file');
  if (!input) {
      alert("Um, couldn't find the fileinput element.");
  }
  else if (!input.files) {
    alert("This browser doesn't seem to support the `files` property of file inputs.");
  }
  else if (!input.files[0]) {
    alert("Please select a file before clicking 'Load'");               
  }
  else {
    file = input.files[0];
    fr = new FileReader();
    fr.onload = receivedText;
    // console.log(receivedText());
    fr.readAsText(file);
    // fr.readAsDataURL(file);
  }
  if((/\.(mid)$/i).test(input.files[0].name)){
    console.log("This is a MIDI file");
  }
  console.log("yo");
}
// Adding the values to the textarea
function receivedText() {
  document.getElementById('ti').value = fr.result;
}