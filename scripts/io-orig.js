/* Converting MIDI to JSON */
function midi_json(){
  var fs = require('fs');
  var midiConverter = require('midi-converter');
  console.log(fs);
  // var midiSong = fs.readFileSync(__dirname + '/alb_se2.mid', 'binary');
  var midiSong = $('#file')[0].files[0];
  var jsonSong = midiConverter.midiToJson(midiSong);
  fs.writeFileSync('example.json', JSON.stringify(jsonSong));
}

midi_json();