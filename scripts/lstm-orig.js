// prediction params
var sample_softmax_temperature = 1.0; // how peaky model predictions should be
var max_chars_gen = 100; // max length of generated sentences

// various global var inits
var epoch_size = -1;
var input_size = -1;
var output_size = -1;
var letterToIndex = {};
var indexToLetter = {};
var vocab = [];
var data_sents = [];
var solver = new R.Solver(); // should be class because it needs memory for step caches
// var pplGraph = new Rvis.Graph();



var model = {};

var initVocab = function(sents, count_threshold) {
  // go over all characters and keep track of all unique ones seen
  var txt = sents.join(''); // concat all

  // count up all characters
  var d = {};
  for(var i=0,n=txt.length;i<n;i++) {
    var txti = txt[i];
    if(txti in d) { d[txti] += 1; } 
    else { d[txti] = 1; }
  }

  // filter by count threshold and create pointers
  letterToIndex = {};
  indexToLetter = {};
  vocab = [];
  // NOTE: start at one because we will have START and END tokens!
  // that is, START token will be index 0 in model letter vectors
  // and END token will be index 0 in the next character softmax
  var q = 1; 
  for(ch in d) {
    if(d.hasOwnProperty(ch)) {
      if(d[ch] >= count_threshold) {
        // add character to vocab
        letterToIndex[ch] = q;
        indexToLetter[q] = ch;
        vocab.push(ch);
        q++;
      }
    }
  }

  // globals written: indexToLetter, letterToIndex, vocab (list), and:
  input_size = vocab.length + 1;
  output_size = vocab.length + 1;
  epoch_size = sents.length;
  $("#prepro_status").text('Found ' + vocab.length + ' distinct characters ');
}

var utilAddToModel = function(modelto, modelfrom) {
  for(var k in modelfrom) {
    if(modelfrom.hasOwnProperty(k)) {
      // copy over the pointer but change the key to use the append
      modelto[k] = modelfrom[k];
    }
  }
}

var initModel = function() {
  // letter embedding vectors
  var model = {};
  model['Wil'] = new R.RandMat(input_size, letter_size , 0, 0.08);
  
  if(generator === 'rnn') {
    var rnn = R.initRNN(letter_size, hidden_sizes, output_size);
    utilAddToModel(model, rnn);
  } else {
    var lstm = R.initLSTM(letter_size, hidden_sizes, output_size);
    utilAddToModel(model, lstm);
  }

  return model;
}

var reinit_learning_rate_slider = function() {
  // init learning rate slider for controlling the decay
  // note that learning_rate is a global variable
  $("#lr_slider").slider({
    min: Math.log10(0.01) - 3.0,
    max: Math.log10(0.01) + 0.05,
    step: 0.05,
    value: Math.log10(learning_rate),
    slide: function( event, ui ) {
      learning_rate = Math.pow(10, ui.value);
      $("#lr_text").text(learning_rate.toFixed(5));
    }
  });
  $("#lr_text").text(learning_rate.toFixed(5));
}

var reinit = function() {
  // note: reinit writes global vars

  // eval options to set some globals
  eval($("#newnet").val());
  console.log(eval($('#newnet').val()));

  // //selecting the generator
  // var e = document.getElementById('generator');
  // generator = e.options[1].value;

  // //select the rest of the parameters
  // hidden_sizes = $('#hidden_sizes').val();
  // letter_size = $('#letter_size').val();
  // regc = $('#regc').val();
  // learning_rate = $('#learning_rate').val();
  // clipval = $('#clipval').val();


  reinit_learning_rate_slider();

  solver = new R.Solver(); // reinit solver
  // pplGraph = new Rvis.Graph();

  ppl_list = [];
  tick_iter = 0;

  // process the input, filter out blanks
  var data_sents_raw = $('#ti').val().split('\n');
  data_sents = [];
  for(var i=0;i<data_sents_raw.length;i++) {
    var sent = data_sents_raw[i].trim();
    if(sent.length > 0) {
      data_sents.push(sent);
    }
  }
  // Removes the sample data when restarting
  while($('#samples').children().length > 0){
    $('#samples').empty()
  }
  initVocab(data_sents, 1); // takes count threshold for characters
  model = initModel();
  initialiseGraph();
}

var saveModel = function() {
  var out = {};
  out['hidden_sizes'] = hidden_sizes;
  out['generator'] = generator;
  out['letter_size'] = letter_size;
  var model_out = {};
  for(var k in model) {
    if(model.hasOwnProperty(k)) {
      model_out[k] = model[k].toJSON();
    }
  }
  out['model'] = model_out;
  var solver_out = {};
  solver_out['decay_rate'] = solver.decay_rate;
  solver_out['smooth_eps'] = solver.smooth_eps;
  step_cache_out = {};
  for(var k in solver.step_cache) {
    if(solver.step_cache.hasOwnProperty(k)) {
      step_cache_out[k] = solver.step_cache[k].toJSON();
    }
  }
  solver_out['step_cache'] = step_cache_out;
  out['solver'] = solver_out;
  out['letterToIndex'] = letterToIndex;
  out['indexToLetter'] = indexToLetter;
  out['vocab'] = vocab;
  $("#tio").val(JSON.stringify(out));
}

var loadModel = function(j) {
  hidden_sizes = j.hidden_sizes;
  generator = j.generator;
  letter_size = j.letter_size;
  model = {};
  for(var k in j.model) {
    if(j.model.hasOwnProperty(k)) {
      var matjson = j.model[k];
      model[k] = new R.Mat(1,1);
      model[k].fromJSON(matjson);
    }
  }
  solver = new R.Solver(); // have to reinit the solver since model changed
  solver.decay_rate = j.solver.decay_rate;
  solver.smooth_eps = j.solver.smooth_eps;
  solver.step_cache = {};
  for(var k in j.solver.step_cache){
      if(j.solver.step_cache.hasOwnProperty(k)){
          var matjson = j.solver.step_cache[k];
          solver.step_cache[k] = new R.Mat(1,1);
          solver.step_cache[k].fromJSON(matjson);
      }
  }
  letterToIndex = j['letterToIndex'];
  indexToLetter = j['indexToLetter'];
  vocab = j['vocab'];

  // reinit these
  ppl_list = [];
  tick_iter = 0;
}

var forwardIndex = function(G, model, ix, prev) {
  var x = G.rowPluck(model['Wil'], ix);
  // forward prop the sequence learner
  if(generator === 'rnn') {
    var out_struct = R.forwardRNN(G, model, hidden_sizes, x, prev);
  } else {
    var out_struct = R.forwardLSTM(G, model, hidden_sizes, x, prev);
  }
  return out_struct;
}

var predictSentence = function(model, samplei, temperature) {
  if(typeof samplei === 'undefined') { samplei = false; }
  if(typeof temperature === 'undefined') { temperature = 1.0; }

  var G = new R.Graph(false);
  var s = '';
  var prev = {};
  while(true) {

    // RNN tick
    var ix = s.length === 0 ? 0 : letterToIndex[s[s.length-1]];
    var lh = forwardIndex(G, model, ix, prev);
    prev = lh;

    // sample predicted letter
    logprobs = lh.o;
    if(temperature !== 1.0 && samplei) {
      // scale log probabilities by temperature and renormalize
      // if temperature is high, logprobs will go towards zero
      // and the softmax outputs will be more diffuse. if temperature is
      // very low, the softmax outputs will be more peaky
      for(var q=0,nq=logprobs.w.length;q<nq;q++) {
        logprobs.w[q] /= temperature;
      }
    }

    probs = R.softmax(logprobs);
    if(samplei) {
      var ix = R.samplei(probs.w);
    } else {
      var ix = R.maxi(probs.w);  
    }
    
    if(ix === 0) break; // END token predicted, break out
    if(s.length > max_chars_gen) { break; } // something is wrong

    var letter = indexToLetter[ix];
    s += letter;
  }
  return s;
}

var costfun = function(model, sent) {
  // takes a model and a sentence and
  // calculates the loss. Also returns the Graph
  // object which can be used to do backprop
  var n = sent.length;
  var G = new R.Graph();
  var log2ppl = 0.0;
  var cost = 0.0;
  var prev = {};
  for(var i=-1;i<n;i++) {
    // start and end tokens are zeros
    var ix_source = i === -1 ? 0 : letterToIndex[sent[i]]; // first step: start with START token
    var ix_target = i === n-1 ? 0 : letterToIndex[sent[i+1]]; // last step: end with END token

    lh = forwardIndex(G, model, ix_source, prev);
    prev = lh;

    // set gradients into logprobabilities
    logprobs = lh.o; // interpret output as logprobs
    probs = R.softmax(logprobs); // compute the softmax probabilities

    log2ppl += -Math.log2(probs.w[ix_target]); // accumulate base 2 log prob and do smoothing
    cost += -Math.log(probs.w[ix_target]);

    // write gradients into log probabilities
    logprobs.dw = probs.w;
    logprobs.dw[ix_target] -= 1
  }
  // perplexity
  var ppl = Math.pow(2, log2ppl / (n - 1));
  return {'G':G, 'ppl':ppl, 'cost':cost};
}

function median(values) {
  values.sort( function(a,b) {return a - b;} );
  var half = Math.floor(values.length/2);
  if(values.length % 2) return values[half];
  else return (values[half-1] + values[half]) / 2.0;
}

function predict(){
  var out = "";
  // Printing 10 samples
  for(var i =0; i < 10; i++){ 
    var temp = predictSentence(model, true, sample_softmax_temperature);
    var pred = '<p class="apred">'+ temp + '</p>';
    out = out + pred;
  }
  return out;
}

function sample(iter){
    // var pred_div = '<div class="apred">'+pred+'</div>'
  var pred_button = '<button type="button" class="btn btn-lg outline sampleButton" data-toggle="modal" data-target="#myModal' + iter + '"> Open sample ' + tick_iter/100 + '</button><br>';
  $('#samples').append(pred_button);
  // $('#sample_output').append('<p class="apred">'+pred+'</p>');

  $('#samples').append(
    '<div class="modal fade" id="myModal' + iter + '" role="dialog"><div class="modal-dialog"><div class="modal-content"><div class="modal-header"><button type="button" class="close" data-dismiss="modal">&times;</button><h4 class="modal-title">Sample' + iter/100 + '</h4></div><div class="modal-body">'+ predict() + '</div></div></div></div>');

  $('#sample_style').scrollTop($('#sample_style')[0].scrollHeight);

}

var ppl_list = [];
var tick_iter = 0;
var tick = function() {
  // sample sentence fromd data
  var sentix = R.randi(0,data_sents.length);
  var sent = data_sents[sentix];

  var t0 = +new Date();  // log start timestamp

  // evaluate cost function on a sentence
  var cost_struct = costfun(model, sent);
  
  // use built up graph to compute backprop (set .dw fields in mats)
  cost_struct.G.backward();
  // perform param update
  var solver_stats = solver.step(model, learning_rate, regc, clipval);
  //$("#gradclip").text('grad clipped ratio: ' + solver_stats.ratio_clipped)

  var t1 = +new Date();
  var tick_time = t1 - t0;

  ppl_list.push(cost_struct.ppl); // keep track of perplexity

  // evaluate now and then
  tick_iter += 1;
  if(tick_iter % 100 === 0) {
    // draw samples
    sample(tick_iter);

  }
  if(tick_iter % 10 === 0) {
    // draw argmax prediction
    $('#argmax').html('');
    var pred = predictSentence(model, false);
    var pred_div = '<div class="apred">'+pred+'</div>'
    $('#argmax').append(pred_div);

    // keep track of perplexity
    $('#epoch').text('Epoch: ' + (tick_iter/epoch_size).toFixed(2));
    $('#ppl').text('Perplexity: ' + cost_struct.ppl.toFixed(2));
    $('#ticktime').text('Forw/bwd time per example: ' + tick_time.toFixed(1) + 'ms');
  }

  if(tick_iter % 100 === 0 || tick_iter === 1) {
    var median_ppl = median(ppl_list);
    ppl_list = [];
    updateVisual(tick_iter, median_ppl);
    // pplGraph.add(tick_iter, median_ppl);
    // pplGraph.drawSelf(document.getElementById("pplgraph"));
    
    //Print values to text area
    var analyticValue = 'X(Tick iteration): ' + tick_iter + ' Y(Median Perplexity): ' + median_ppl.toFixed(2) + "\n";
    var temp = $('#analytics').val();
    $('#analytics').val(temp + analyticValue);
    $('#analytics').scrollTop($('#analytics')[0].scrollHeight);
  }
}

var gradCheck = function() {
  var model = initModel();
  var sent = '^test sentence$';
  var cost_struct = costfun(model, sent);
  cost_struct.G.backward();
  var eps = 0.000001;

  for(var k in model) {
    if(model.hasOwnProperty(k)) {
      var m = model[k]; // mat ref
      for(var i=0,n=m.w.length;i<n;i++) {
        
        oldval = m.w[i];
        m.w[i] = oldval + eps;
        var c0 = costfun(model, sent);
        m.w[i] = oldval - eps;
        var c1 = costfun(model, sent);
        m.w[i] = oldval;

        var gnum = (c0.cost - c1.cost)/(2 * eps);
        var ganal = m.dw[i];
        var relerr = (gnum - ganal)/(Math.abs(gnum) + Math.abs(ganal));
        if(relerr > 1e-1) {
          console.log(k + ': numeric: ' + gnum + ', analytic: ' + ganal + ', err: ' + relerr);
        }
      }
    }
  }
}

/* HANDLING INPUT FILES */


/* Converting MIDI to JSON */
function midi_json(){
  MidiConvert.load("scripts/alb_se2.mid", function(midiData){
    console.log(midiData);
  });
}

/*
  Creating a button where you can upload your own file to the model
*/
function handleFile(){
  input = document.getElementById('file');
  file = input.files[0];
  fr = new FileReader();
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
    fr.onload = receivedText;
    // console.log(receivedText());
    fr.readAsText(file);
    // fr.readAsDataURL(file);
  }
  if((/\.(mid)$/i).test(input.files[0].name)){
    console.log("This is a MIDI file");
    midi_json();
  }
  console.log("yo");
}
// Adding the values to the textarea
function receivedText() {
  document.getElementById('ti').value = fr.result;
}

/* END HANDLING INPUT FILES */

var iid = null;
$(function() {
  // pplGraph.drawSelf(document.getElementById("pplgraph"));
  initialiseGraph();
  $('#epoch').text('Epoch: ' + 0);
  $('#ppl').text('Perplexity: ' + 0);
  $('#ticktime').text('Forw/bwd time per example: ' + 0);
  $('#mean_ppl').text('Median Perplexity: ' + 0);
  $('#tick_iter').text('Tick iteration: ' + 0);

  //initial print
  var analyticValue = 'X(Tick iteration): ' + 0 + ' Y(Median Perplexity): ' + 0 + "\n";
  var temp = $('#analytics').val();
  $('#analytics').val(temp + analyticValue);

  //initial found of characters
  $("#prepro_status").text('Found ' + 0 + ' distinct characters ');

  //initial learning rate slider
  $("#lr_text").text(0);
  //initial softmax temperature slidr
  $("#temperature_text").text(0);

  // attach button handlers
  $('#learn').click(function(){ 
    reinit();
    if(iid !== null) { 
      clearInterval(iid); 
      initialiseGraph();
      $('#analytics').val("");
    }
    if($("#stop").data('clicked', true)){
      clearInterval(iid); 
      initialiseGraph();
      $('#analytics').val("");
    }
    iid = setInterval(tick, 0);
  });
  $('#stop').click(function(){ 
    if(iid !== null) { clearInterval(iid);}
    iid = null;
  });
  $("#resume").click(function(){
    if(iid === null) {
      iid = setInterval(tick, 0); 
    }
  });

  // initialise all model parameter updates
  $("form div").append('<div class="inc button">+</div><div class="dec button">-</div>');

  // $('#loadText').click(loadText());

  $("#savemodel").click(saveModel);
  $("#loadmodel").click(function(){
    var j = JSON.parse($("#tio").val());
    loadModel(j);
  });

  // $("#loadpretrained").click(function(){
  //   $.getJSON("lstm_100_model.json", function(data) {
  //     // pplGraph = new Rvis.Graph();
  //     learning_rate = 0.0001;
  //     reinit_learning_rate_slider();
  //     loadModel(data);
  //   });
  // });

  // $("#learn").click(); // simulate click on startup

  //$('#gradcheck').click(gradCheck);

  //initial Learning Rate Slider
  $("#lr_slider").slider({
    min: Math.log10(0.01) - 3.0,
    max: Math.log10(0.01) + 0.05,
    step: 0.05,
    value: 0
  });

  //initial temperature slider
  $("#temperature_slider").slider({
    min: -1,
    max: 1.05,
    step: 0.05,
    value: 0,
    slide: function( event, ui ) {
      sample_softmax_temperature = Math.pow(10, ui.value);
      $("#temperature_text").text( sample_softmax_temperature.toFixed(2) );
      console.log(sample_softmax_temperature);
    }
  });

  /* Letting the user choose a file to upload */
  $(document).on('change', ':file', function() {
    var input = $(this),
        numFiles = input.get(0).files ? input.get(0).files.length : 1,
        label = input.val().replace(/\\/g, '/').replace(/.*\//, '');
    input.trigger('fileselect', [numFiles, label]);
  });
  /* Watching the file(s) we picked */
  $(':file').on('fileselect', function(event, numFiles, label) {
      var input = $(this).parents('.input-group').find(':text');
      handleFile();
      input.val(label);
  });
}); 