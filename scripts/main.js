//Lets require/import the HTTP module and port
var http = require('http'),
	PORT = process.env.PORT || 8080,
	fs = require('fs'),
	express = require('express'),
	dispatcher = require('httpdispatcher');

// Creating an express app
var app = express();
app.set( __dirname);
app.engine('html', require('ejs').renderFile);
app.get('/', function(req, res){
	res.sendfile('./public/index.html');
});
app.listen(3000, function(){
	console.log('Running...');
});


var json = require('./vis');

//Lets start our server
fs.readFile('index.html', function (err, html) {
	if (err) {
	    throw err; 
	}       
	http.createServer(function(request, response) { 
	    response.writeHeader(200, {"Content-Type": "text/html"});  
	    response.write(html);  
	    response.end();  
	}).listen(PORT, function(){
		console.log("Server listening on: http://localhost:%s", PORT);
	});
});


// Converting a MIDI file to JSON

// var PythonShell = require('python-shell');

// var options = {
//   mode: 'text',
//   pythonPath: '/opt/local/bin/python',
//   scriptPath: '/Users/nathalievonhuth/Documents/MachineLearning/music_www/scripts'
// };

// PythonShell.run('my_start.py', options, function (err) {
//   if (err) throw err;
//   console.log('finished');
// });

