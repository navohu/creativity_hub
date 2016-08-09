var env = require('jsdom').env;
var html = '<html>...</html>';
var fs = require('fs');
var midiConverter = require('midi-converter');
var d3 = require('d3');
var MIDIPlayer = require('midiplayer');
var MIDIFile = require('midifile');

// Creating a JSON file from a MIDI file
function createJSON(){
	env(html, function(err, window) {
		var $ = require('jquery')(window);
		var midiSong = fs.readFileSync('./data/alb_esp1.mid', 'binary');
		var jsonSong = midiConverter.midiToJson(midiSong);
		fs.writeFileSync('example.json', JSON.stringify(jsonSong));

		var jSong = $.getJSON("example.json");
		// console.log(jSong.getResponseHeader());

		module.exports = jSong;
	});
}

navigator.requestMIDIAccess().then(function(midiAccess) {
// Creating player
var midiPlayer = new MIDIPlayer({
  'output': midiAccess.outputs()[0]
});

// creating the MidiFile instance from a buffer (view MIDIFile README)
var midiFile = new MIDIFile("data/alb_esp1.mid");

// Loading the midiFile instance in the player
midiPlayer.load(midiFile);

// Playing
midiPlayer.play(function() {
    console.log('Play ended');
});

// Volume
midiPlayer.volume = 80; // in percent
});

// Parsing the JSON object in order to be analysed
fs.readFile('example.json', 'utf8', function (err, data) {
    if (err) throw err; // we'll not consider error handling for now
    var obj = JSON.parse(data);
    for (var k in obj.tracks[0][100]){
    	console.log(k);
    }
});

// playMIDI();