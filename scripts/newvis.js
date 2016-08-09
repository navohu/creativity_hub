function createStartTime(song){
	var startTime;
	d3.json(song, function(data){
		for (var i = 0; i < data.length; i++) {
			for (var j = 0; j < data[i].length; j++) {
				if (data.tracks[i][j].subtype == "setTempo") {
					startTime = data.tracks[i][j].deltaTime;
					break;
				}
			}
		}
		// console.log(startTime);
	});
}

function createEndTime(song){
	var endTime;
	var k = 1;
	d3.json(song, function(data){
		for (var i = 0; i < data.tracks.length; i++) {
			for (var j = 0; j < data.tracks[i].length; j++) {
				console.log("Track: "+ data.tracks[i].length-k + " " + data.tracks[i][data.tracks[i].length-k].subtype);
				if(data.tracks[i][data.tracks[i].length-k] == "undefined"){
					continue;
				}
				if (data.tracks[i][data.tracks[i].length-k].subtype == "setTempo") {
					endTime = data.tracks[i][data.tracks[i].length-1].deltaTime;
					console.log(endTime);
					break;
				}
				else{
					k++;
				}
			}
		}
	});
	// console.log(endTime);
}

function createHistogram(song){
	var parseDate = d3.timeParse("%M:%S %p"),
		formatCount = d3.format(",.0f");

	var margin = {top: 10, right: 30, bottom: 30, left: 30},
		width = 960 - margin.left - margin.right,
		height = 500 - margin.top - margin.bottom;

	var x = d3.scaleTime()
		.domain([new Date(2015, 0, 1), new Date(2016, 0, 1)])
		.rangeRound([0, width]);

	var y = d3.scaleLinear()
		.range([height, 0]);

	var histogram = d3.histogram()
		.value(function(d) { return d.date; })
		.domain(x.domain())
		.thresholds(x.ticks(d3.timeWeek));

	var svg = d3.select("#histogram").append("svg")
		.attr("width", width + margin.left + margin.right)
		.attr("height", height + margin.top + margin.bottom)
		.append("g")
		.attr("transform", "translate(" + margin.left + "," + margin.top + ")");

	svg.append("g")
		.attr("class", "axis axis--x")
		.attr("transform", "translate(0," + height + ")")
		.call(d3.axisBottom(x));

	d3.json(song, function(data) {
		var bins = histogram(data);

		y.domain([0, d3.max(bins, function(d) { return d.length; })]);

		var bar = svg.selectAll(".bar")
			.data(bins)
			.enter().append("g")
			.attr("class", "bar")
			.attr("transform", function(d) { return "translate(" + x(d.x0) + "," + y(d.length) + ")"; });

		bar.append("rect")
			.attr("x", 1)
			.attr("width", function(d) { return x(d.x1) - x(d.x0) - 1; })
			.attr("height", function(d) { return height - y(d.length); });

		bar.append("text")
			.attr("dy", ".75em")
			.attr("y", 6)
			.attr("x", function(d) { return (x(d.x1) - x(d.x0)) / 2; })
			.attr("text-anchor", "middle")
			.text(function(d) { return formatCount(d.length); });
	});
}

function playMIDI(data){
	navigator.requestMIDIAccess().then(function(midiAccess) {
		// console.log("MidiAccess: " + midiAccess.outputs[0]);
		// creating the Midi Player
		var midiPlayer = new MIDIPlayer({
		'output': midiAccess.outputs[0]
		});
		// creating the ArrayBuffer
		var blob = new Blob([data]);
		var buffer = new FileReader().readAsArrayBuffer(blob);

		// creating the MidiFile instance from a buffer (view MIDIFile README)
		var midiFile = new MIDIFile(buffer);
		// Loading the midiFile instance in the player
		midiPlayer.load(midiFile);
		// Playing
		midiPlayer.play();
		console.log(midiPlayer);
		console.log("Midi playing");
	});
}

// playMIDI("data/alb_esp1.mid");
// createHistogram("example.json");
// createStartTime("example.json");
// createEndTime("example.json");


