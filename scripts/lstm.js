(function e(t,n,r){function s(o,u){if(!n[o]){if(!t[o]){var a=typeof require=="function"&&require;if(!u&&a)return a(o,!0);if(i)return i(o,!0);var f=new Error("Cannot find module '"+o+"'");throw f.code="MODULE_NOT_FOUND",f}var l=n[o]={exports:{}};t[o][0].call(l.exports,function(e){var n=t[o][1][e];return s(n?n:e)},l,l.exports,e,t,n,r)}return n[o].exports}var i=typeof require=="function"&&require;for(var o=0;o<r.length;o++)s(r[o]);return s})({1:[function(require,module,exports){
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
  
  // var jsonfile = require('jsonfile');
  var midiConverter = require('midi-converter');
  var midiSong = "MThd\u0000\u0000\u0000\u0006\u0000\u0001\u0000\t\u0001àMTrk\u0000\u0000\u0017Á\u0000ÿ\u0003\u000fSuite espagnole\u0000ÿ\u0003\bCataluna\u0000ÿ\u0002!Copyright © 2001 by Bernd Krueger\u0000ÿ\u0001\rIsaac Albeniz\u0000ÿ\u0001\bCurranda\u0000ÿ\u0001\u001cFertiggestellt am 11.3.2001\n\u0000ÿ\u0001\u0014Update am 30.3.2001\n\u0000ÿ\u0001\u0014Update am 26.5.2001\n\u0000ÿ\u0001\u0017Normierung: 23.12.2002\n\u0000ÿ\u0001\u0013Update am 4.9.2010\n\u0000ÿ\u0001\u0014Dauer: 3:59 Minuten\n\u0000ÿT\u0005`\u0000\u0003\u0000\u0000\u0000ÿX\u0004\u0006\u0003\f\b\u0000ÿY\u0002þ\u0000\u0000ÿQ\u0003\b¾Á\u0000ÿ\u0006\u0007AllegroxÿQ\u0003\bØ~HÿQ\u0003\båz ÿQ\u0003\bòÕ ÿQ\u0003\t\u0000\u001f ÿQ\u0003\t\rÊ ÿQ\u0003\t\u001be ÿQ\u0003\t)( ÿQ\u0003\t7R ÿQ\u0003\tEk ÿQ\u0003\tSí ÿQ\u0003\b¥PÿQ\u0003\u0007¨&`ÿQ\u0003\b\u000eûpÿQ\u0003\u0007§~\u0000ÿ\u0006\u0006Teil 1`ÿQ\u0003\u0007é]pÿQ\u0003\u0007Ã=`ÿQ\u0003\u0007ä¦pÿQ\u0003\b\u000eû`ÿQ\u0003\u0007¹ÙpÿQ\u0003\u0007Õ`ÿQ\u0003\b\u0004$pÿQ\u0003\u0007°`ÿQ\u0003\u0007ßÈpÿQ\u0003\u0007¼\u0005`ÿQ\u0003\b\u0019¿pÿQ\u0003\b$Ñ`ÿQ\u0003\u0007é]pÿQ\u0003\u0007¼\u0005pÿQ\u0003\b\u000eûpÿQ\u0003\b]dpÿQ\u0003\b\u0011)`ÿQ\u0003\bËqpÿQ\u0003\b\u0011)`ÿQ\u0003\bQpÿQ\u0003\b/ÑPÿQ\u0003\u0007ó6pÿQ\u0003\b:pÿQ\u0003\b]d\u0018ÿQ\u0003\u0007é](ÿQ\u0003\u0007îûpÿQ\u0003\b\u001bgpÿQ\u0003\b]dpÿQ\u0003\u0007ßÈ`ÿQ\u0003\b0\u0001pÿQ\u0003\u0007ó6`ÿQ\u0003\b\u0019¿pÿQ\u0003\b]d`ÿQ\u0003\t\u0017°pÿQ\u0003\b©\\pÿQ\u0003\bþ\u0015pÿQ\u0003\t\u001bePÿQ\u0003\u0007ßÈ\u0010ÿQ\u0003\u0007¨&`ÿQ\u0003\b\u000eûpÿQ\u0003\u0007°¶`ÿQ\u0003\u0007é]pÿQ\u0003\u0007Ã=`ÿQ\u0003\u0007ä¦pÿQ\u0003\b\u000eû`ÿQ\u0003\u0007¹ÙpÿQ\u0003\u0007Õ`ÿQ\u0003\b\u0004$pÿQ\u0003\u0007°`ÿQ\u0003\u0007ßÈpÿQ\u0003\u0007¼\u0005`ÿQ\u0003\b\u0019¿pÿQ\u0003\b$Ñ`ÿQ\u0003\u0007é]pÿQ\u0003\u0007¼\u0005pÿQ\u0003\b\u000eûpÿQ\u0003\b]dpÿQ\u0003\b\u0011)`ÿQ\u0003\bËqpÿQ\u0003\b\u0011)`ÿQ\u0003\bQpÿQ\u0003\b/ÑPÿQ\u0003\u0007ó6pÿQ\u0003\b:pÿQ\u0003\b]d\u0018ÿQ\u0003\u0007é](ÿQ\u0003\u0007îûpÿQ\u0003\b\u001bgpÿQ\u0003\b]dpÿQ\u0003\u0007ßÈ`ÿQ\u0003\b0\u0001pÿQ\u0003\u0007ó6`ÿQ\u0003\b\u0019¿pÿQ\u0003\b]d`ÿQ\u0003\t\u0017°pÿQ\u0003\b©\\pÿQ\u0003\bþ\u0015pÿQ\u0003\t\u001be@ÿQ\u0003\u0007ó6 ÿQ\u0003\u0007¨&`ÿQ\u0003\b\u000eûpÿQ\u0003\u0007¹Ù`ÿQ\u0003\u0007é]pÿQ\u0003\u0007Ã=`ÿQ\u0003\u0007ä¦pÿQ\u0003\b\u000eû`ÿQ\u0003\u0007¹ÙpÿQ\u0003\u0007Õ`ÿQ\u0003\bQpÿQ\u0003\u0007¹Ù`ÿQ\u0003\u0007é]pÿQ\u0003\u0007Ã=`ÿQ\u0003\u0007ä¦pÿQ\u0003\b\u000eû`ÿQ\u0003\u0007¹Ù\\ÿQ\u0003\u0007Õ`ÿQ\u0003\bQ\u0004ÿQ\u0003\u0007¹Ù`ÿQ\u0003\b0\u0001pÿQ\u0003\u0007¡`ÿQ\u0003\u0007Ì¸pÿQ\u0003\u0007¡`ÿQ\u0003\u0007Ì¸4ÿQ\u0003\bE\u0004<ÿQ\u0003\u0007¡pÿQ\u0003\u0007§~pÿQ\u0003\u0007Ì¸pÿQ\u0003\u0007¡pÿQ\u0003\u0007§~pÿQ\u0003\u0007ý(pÿQ\u0003\u0007¡`ÿQ\u0003\u0007Ì¸pÿQ\u0003\u0007¡`ÿQ\u0003\u0007Ì¸4ÿQ\u0003\bE\u0004<ÿQ\u0003\u0007¡pÿQ\u0003\u0007§~pÿQ\u0003\u0007Ì¸pÿQ\u0003\u0007¡`ÿQ\u0003\u0007Ì¸bÿQ\u0003\u0007¡`ÿQ\u0003\u0007Ì¸~ÿQ\u0003\u0007zRÿQ\u0003\u0007Ì¸BÿQ\u0003\bß<ÿQ\u0003\u0007§~RÿQ\u0003\u0007Ì¸~ÿQ\u0003\u0007°¶RÿQ\u0003\u0007Ì¸\u000eÿQ\u0003\u0007ý(pÿQ\u0003\u0007¹ÙRÿQ\u0003\u0007Ì¸\u000eÿQ\u0003\u0007ó6pÿQ\u0003\u0007¹ÙRÿQ\u0003\u0007Ì¸~ÿQ\u0003\u0007é]pÿQ\u0003\b\u0007\u0005pÿQ\u0003\b0\u0001pÿQ\u0003\u0007Ì¸ ÿQ\u0003\u0007¹ÙRÿQ\u0003\u0007Ì¸\u000eÿQ\u0003\u0007ý(pÿQ\u0003\u0007°¶`ÿQ\u0003\b0\u0001pÿQ\u0003\u0007¹ÙRÿQ\u0003\u0007Ì¸\u000eÿQ\u0003\u0007ý(pÿQ\u0003\u0007°¶`ÿQ\u0003\b0\u0001pÿQ\u0003\u0007Ö\u001fRÿQ\u0003\u0007Ì¸\u000eÿQ\u0003\u0007ý((ÿQ\u0003\b\u0011)HÿQ\u0003\u0007é]`ÿQ\u0003\b0\u00010ÿQ\u0003\b:`ÿQ\u0003\u0007Ö\u001fRÿQ\u0003\u0007Ì¸\u000eÿQ\u0003\u0007ý(pÿQ\u0003\u0007Ã=`ÿQ\u0003\b0\u0001pÿQ\u0003\u0007¹ÙRÿQ\u0003\u0007Ì¸\u000eÿQ\u0003\u0007ý(pÿQ\u0003\u0007°¶`ÿQ\u0003\b0\u0001pÿQ\u0003\u0007ó6RÿQ\u0003\u0007Ì¸\u000eÿQ\u0003\bppÿQ\u0003\b%`ÿQ\u0003\båAHÿQ\u0003\bOÈHÿQ\u0003\bE\u0004 ÿQ\u0003\b: ÿQ\u0003\b0\u0001 ÿQ\u0003\b%PÿQ\u0003\u0007ý(`ÿQ\u0003\b:pÿQ\u0003\u0007ã`ÿQ\u0003\b\u0014BpÿQ\u0003\u0007í9`ÿQ\u0003\b\u000f)pÿQ\u0003\b:`ÿQ\u0003\u0007ãpÿQ\u0003\u0007½Ý`ÿQ\u0003\b/qpÿQ\u0003\u0007ÇÍ`ÿQ\u0003\b\nEpÿQ\u0003\u0007å²`ÿQ\u0003\bEfpÿQ\u0003\bP¿`ÿQ\u0003\b\u0014BpÿQ\u0003\u0007å²pÿQ\u0003\b:pÿQ\u0003\bpÿQ\u0003\b<ÔLÿQ\u0003\bfÒ\u0014ÿQ\u0003\bq\u0014ÿQ\u0003\b|\u0014ÿQ\u0003\b{\u0014ÿQ\u0003\bº\u0014ÿQ\u0003\bâ\u0014ÿQ\u0003\b©\\\u0014ÿQ\u0003\b´¿\u0014ÿQ\u0003\bÀw\u0014ÿQ\u0003\bÌ\u0017\u0014ÿQ\u0003\bØ\u000e\u0014ÿQ\u0003\bä&\u0014ÿQ\u0003\bð&\u0014ÿQ\u0003\b<Ô`ÿQ\u0003\b\\\u0005\u0014ÿQ\u0003\bfÒ\u0014ÿQ\u0003\bq\u0014ÿQ\u0003\b|\u0014ÿQ\u0003\b{\u0014ÿQ\u0003\bº\u0014ÿQ\u0003\bâ\u0014ÿQ\u0003\b©\\\u0014ÿQ\u0003\b´¿\u0014ÿQ\u0003\bÀw\u0014ÿQ\u0003\bÌ\u0017\u0014ÿQ\u0003\bä&\u0014ÿQ\u0003\b[ÓPÿQ\u0003\b\u001e)pÿQ\u0003\bfÒpÿQ\u0003\bpÿQ\u0003\b0\u0001PÿQ\u0003\b:PÿQ\u0003\bE\u0004PÿQ\u0003\bOÈ(ÿQ\u0003\bZuPÿQ\u0003\beqPÿQ\u0003\bp(ÿQ\u0003\b{HÿQ\u0003\b\u0014B(ÿQ\u0003\b\u0019¿pÿQ\u0003\bG\u001fpÿQ\u0003\bpÿQ\u0003\b\nE`ÿQ\u0003\b\\\u0005pÿQ\u0003\b\u001e)`ÿQ\u0003\bEfpÿQ\u0003\b`ÿQ\u0003\tHÊpÿQ\u0003\bØ\u000epÿQ\u0003\t.ÍpÿQ\u0003\tL¦xÿQ\u0003\tÅ\u0011xÿQ\u0003\tñpÿQ\u0003\båA\\ÿQ\u0003\bQ\u0014ÿQ\u0003\b:\u0014ÿQ\u0003\bOÈ(ÿQ\u0003\b:\u0014ÿQ\u0003\t1£\u0014ÿQ\u0003\tQ\u0014ÿQ\u0003\t¼Ò\u0014ÿQ\u0003\téæ\u0014ÿQ\u0003\n(³\u0014ÿQ\u0003\njb\u0014ÿQ\u0003\u000bCdÿQ\u0003\u0007Õ\u0014ÿQ\u0003\u0007Ì¸PÿQ\u0003\u0007¨&`ÿQ\u0003\b\u000eûpÿQ\u0003\u0007¹Ù`ÿQ\u0003\u0007é]pÿQ\u0003\u0007Ã=`ÿQ\u0003\u0007ä¦pÿQ\u0003\b\u000eû`ÿQ\u0003\u0007¹ÙpÿQ\u0003\u0007Õ`ÿQ\u0003\bQpÿQ\u0003\u0007¹Ù`ÿQ\u0003\u0007é]pÿQ\u0003\u0007Ã=`ÿQ\u0003\u0007ä¦pÿQ\u0003\b\u000eû`ÿQ\u0003\u0007¹Ù\\ÿQ\u0003\u0007Õ`ÿQ\u0003\bQ\u0004ÿQ\u0003\bOÈPÿQ\u0003\u0007é]xÿQ\u0003\u0007ßÈPÿQ\u0003\u0007Ö\u001fPÿQ\u0003\u0007Ì¸PÿQ\u0003\u0007Ã=PÿQ\u0003\u0007¹ÙPÿQ\u0003\u0007°¶PÿQ\u0003\u0007§~PÿQ\u0003\u0007](ÿQ\u0003\beqPÿQ\u0003\u0007Çø`ÿQ\u0003\bËàpÿQ\u0003\u0007íìPÿQ\u0003\u0007ÇøpÿQ\u0003\u0007÷¤pÿQ\u0003\t\bMpÿQ\u0003\u000bmªPÿQ\u0003\u0007íìRÿQ\u0003\b\u0001\u0017\u000eÿQ\u0003\b2Ñ@ÿQ\u0003\u0007äy`ÿQ\u0003\bGQpÿQ\u0003\u0007÷¤RÿQ\u0003\b\u0001\u0017\u000eÿQ\u0003\b2ÑpÿQ\u0003\u0007äy`ÿQ\u0003\bg7pÿQ\u0003\bq»PÿQ\u0003\u0007äy`ÿQ\u0003\bg7pÿQ\u0003\b%`ÿQ\u0003\bñ¶xÿQ\u0003\tu;xÿQ\u0003\b{`ÿQ\u0003\t$ypÿQ\u0003\b©\\pÿQ\u0003\beqbÿQ\u0003\u0007Ì¸\u000eÿQ\u0003\u0007ý(pÿQ\u0003\u0007¨&`ÿQ\u0003\b\u000eûpÿQ\u0003\u0007§~\u0000ÿ\u0006\u0013Teil 1 Wiederholung`ÿQ\u0003\u0007é]pÿQ\u0003\u0007Ã=`ÿQ\u0003\u0007ä¦pÿQ\u0003\b\u000eû`ÿQ\u0003\u0007¹ÙpÿQ\u0003\u0007Õ`ÿQ\u0003\b\u0004$pÿQ\u0003\u0007°pÿQ\u0003\u0007yDpÿQ\u0003\u0007ßÈpÿQ\u0003\u0007¼\u0005`ÿQ\u0003\b\u0019¿pÿQ\u0003\b$Ñ`ÿQ\u0003\u0007é]pÿQ\u0003\u0007¼\u0005pÿQ\u0003\b\u001a\u001dpÿQ\u0003\b]dpÿQ\u0003\b\u0011)`ÿQ\u0003\bËqpÿQ\u0003\b\u0011)`ÿQ\u0003\bQpÿQ\u0003\b/ÑPÿQ\u0003\u0007ó6pÿQ\u0003\b:pÿQ\u0003\b]d\u0018ÿQ\u0003\u0007é](ÿQ\u0003\u0007îûpÿQ\u0003\b\u001bgpÿQ\u0003\b]dpÿQ\u0003\u0007ßÈpÿQ\u0003\u0007Ê®pÿQ\u0003\b0\u0001pÿQ\u0003\u0007ó6`ÿQ\u0003\b\u0019¿pÿQ\u0003\b]d`ÿQ\u0003\t\u0017°pÿQ\u0003\b©\\pÿQ\u0003\bþ\u0015pÿQ\u0003\t\u001bePÿQ\u0003\u0007ßÈ\u0010ÿQ\u0003\u0007¨&`ÿQ\u0003\b\u000eûpÿQ\u0003\u0007¹Ù\u0014ÿQ\u0003\u0007§~\\ÿQ\u0003\u0007\u0003pÿQ\u0003\u0007é]pÿQ\u0003\u0007Ã=`ÿQ\u0003\u0007ä¦pÿQ\u0003\b\u000eû`ÿQ\u0003\u0007¹ÙpÿQ\u0003\u0007Õ`ÿQ\u0003\b\u0004$pÿQ\u0003\u0007°`ÿQ\u0003\u0007ßÈpÿQ\u0003\u0007¼\u0005`ÿQ\u0003\b\u0019¿pÿQ\u0003\b$Ñ`ÿQ\u0003\u0007é]pÿQ\u0003\u0007¼\u0005pÿQ\u0003\b\u000eûpÿQ\u0003\b]dpÿQ\u0003\b\u0011)pÿQ\u0003\b#UpÿQ\u0003\bËqpÿQ\u0003\b\u0011)`ÿQ\u0003\bQpÿQ\u0003\b/ÑPÿQ\u0003\u0007ó6pÿQ\u0003\b:pÿQ\u0003\b]d\u0018ÿQ\u0003\u0007é](ÿQ\u0003\u0007îûpÿQ\u0003\b\u001bgpÿQ\u0003\b]dpÿQ\u0003\u0007ßÈ`ÿQ\u0003\b0\u0001pÿQ\u0003\u0007ó6`ÿQ\u0003\b\u0019¿pÿQ\u0003\b]d`ÿQ\u0003\t\u0017°pÿQ\u0003\b©\\pÿQ\u0003\bþ\u0015pÿQ\u0003\t\u001be@ÿQ\u0003\u0007ó6 ÿQ\u0003\u0007¨&`ÿQ\u0003\b\u000eûpÿQ\u0003\u0007¹Ù`ÿQ\u0003\u0007é]pÿQ\u0003\u0007Ã=`ÿQ\u0003\u0007ä¦pÿQ\u0003\b\u000eû`ÿQ\u0003\u0007¹ÙpÿQ\u0003\u0007Õ`ÿQ\u0003\bQpÿQ\u0003\u0007¹Ù`ÿQ\u0003\u0007é]pÿQ\u0003\u0007Ã=`ÿQ\u0003\u0007ä¦pÿQ\u0003\b\u000eû`ÿQ\u0003\u0007Ê®\\ÿQ\u0003\u0007Õ`ÿQ\u0003\bQ\u0004ÿQ\u0003\u0007¹Ù`ÿQ\u0003\b0\u0001pÿQ\u0003\u0007¡`ÿQ\u0003\u0007Ì¸pÿQ\u0003\u0007¡`ÿQ\u0003\u0007Ì¸4ÿQ\u0003\b\u0010ú<ÿQ\u0003\u0007¡pÿQ\u0003\u0007§~pÿQ\u0003\u0007Ì¸pÿQ\u0003\u0007¡pÿQ\u0003\u0007§~pÿQ\u0003\u0007ý(pÿQ\u0003\u0007¡`ÿQ\u0003\u0007Ì¸pÿQ\u0003\u0007¡`ÿQ\u0003\u0007Ì¸4ÿQ\u0003\b\u001a\u001d<ÿQ\u0003\u0007¡pÿQ\u0003\u0007§~pÿQ\u0003\u0007Ì¸pÿQ\u0003\u0007¡`ÿQ\u0003\u0007Ì¸bÿQ\u0003\u0007¡`ÿQ\u0003\u0007Ì¸\"ÿQ\u0003\u0007Ê®\\ÿQ\u0003\u0007zRÿQ\u0003\u0007Ì¸BÿQ\u0003\bß<ÿQ\u0003\u0007§~RÿQ\u0003\u0007Ì¸~ÿQ\u0003\u0007°¶RÿQ\u0003\u0007Ì¸\u000eÿQ\u0003\u0007ý(pÿQ\u0003\u0007¹ÙRÿQ\u0003\u0007Ì¸\u000eÿQ\u0003\u0007ó6pÿQ\u0003\u0007¹ÙRÿQ\u0003\u0007Ì¸~ÿQ\u0003\u0007é]pÿQ\u0003\b\u0007\u0005pÿQ\u0003\b0\u0001pÿQ\u0003\u0007Ì¸pÿQ\u0003\u0007b\u001fxÿQ\u0003\u0007í9xÿQ\u0003\u0007Ê®@ÿQ\u0003\u0007¹ÙRÿQ\u0003\u0007Ì¸\u000eÿQ\u0003\u0007ý(pÿQ\u0003\u0007°¶`ÿQ\u0003\b0\u0001pÿQ\u0003\u0007¹ÙRÿQ\u0003\u0007Ì¸\u000eÿQ\u0003\u0007ý(pÿQ\u0003\u0007°¶`ÿQ\u0003\b0\u0001pÿQ\u0003\u0007Ö\u001fRÿQ\u0003\u0007Ì¸\u000eÿQ\u0003\u0007ý((ÿQ\u0003\b\u0011)HÿQ\u0003\u0007é]`ÿQ\u0003\b0\u00010ÿQ\u0003\b:`ÿQ\u0003\u0007Ö\u001fRÿQ\u0003\u0007Ì¸\u000eÿQ\u0003\u0007ý(pÿQ\u0003\u0007Ã=`ÿQ\u0003\b0\u0001pÿQ\u0003\u0007¹ÙRÿQ\u0003\u0007Ì¸\u000eÿQ\u0003\u0007ý(pÿQ\u0003\u0007°¶`ÿQ\u0003\b0\u0001pÿQ\u0003\u0007ó6RÿQ\u0003\u0007Ì¸\u000eÿQ\u0003\bppÿQ\u0003\b%`ÿQ\u0003\båAHÿQ\u0003\bOÈHÿQ\u0003\bE\u0004 ÿQ\u0003\b: ÿQ\u0003\b0\u0001 ÿQ\u0003\b%PÿQ\u0003\u0007äy`ÿQ\u0003\b:pÿQ\u0003\u0007ã`ÿQ\u0003\b\u0014BpÿQ\u0003\u0007í9`ÿQ\u0003\b\u000f)pÿQ\u0003\b:`ÿQ\u0003\u0007ãpÿQ\u0003\u0007½Ý`ÿQ\u0003\b/qpÿQ\u0003\u0007ÇÍ`ÿQ\u0003\b\nEpÿQ\u0003\u0007å²`ÿQ\u0003\bEfpÿQ\u0003\bP¿`ÿQ\u0003\b\u0014BpÿQ\u0003\u0007å²pÿQ\u0003\b\\ipÿQ\u0003\bpÿQ\u0003\b<ÔLÿQ\u0003\bfÒ\u0014ÿQ\u0003\bq\u0014ÿQ\u0003\b|\u0014ÿQ\u0003\b{\u0014ÿQ\u0003\bº\u0014ÿQ\u0003\bâ\u0014ÿQ\u0003\b©\\\u0014ÿQ\u0003\b´¿\u0014ÿQ\u0003\bÀw\u0014ÿQ\u0003\bÌ\u0017\u0014ÿQ\u0003\bØ\u000e\u0014ÿQ\u0003\bä&\u0014ÿQ\u0003\bð&\u0014ÿQ\u0003\b<Ô`ÿQ\u0003\b\\\u0005\u0014ÿQ\u0003\bfÒ\u0014ÿQ\u0003\bq\u0014ÿQ\u0003\b|\u0014ÿQ\u0003\b{\u0014ÿQ\u0003\bº\u0014ÿQ\u0003\bâ\u0014ÿQ\u0003\b©\\\u0014ÿQ\u0003\b´¿\u0014ÿQ\u0003\bÀw\u0014ÿQ\u0003\bÌ\u0017\u0014ÿQ\u0003\bä&\u0014ÿQ\u0003\b[ÓPÿQ\u0003\b\u001e)pÿQ\u0003\bfÒpÿQ\u0003\bpÿQ\u0003\b0\u0001PÿQ\u0003\b:PÿQ\u0003\bE\u0004PÿQ\u0003\bOÈ(ÿQ\u0003\bZuPÿQ\u0003\beqPÿQ\u0003\bp(ÿQ\u0003\b´HÿQ\u0003\b\u0014B(ÿQ\u0003\b\u0019¿pÿQ\u0003\bG\u001fpÿQ\u0003\bpÿQ\u0003\b\nEpÿQ\u0003\u0007ö\u000bpÿQ\u0003\b\\\u0005pÿQ\u0003\b\u001e)`ÿQ\u0003\bEfpÿQ\u0003\b`ÿQ\u0003\tHÊpÿQ\u0003\bØ\u000epÿQ\u0003\t.ÍpÿQ\u0003\tL¦xÿQ\u0003\tÅ\u0011xÿQ\u0003\tñpÿQ\u0003\båA\\ÿQ\u0003\bQ\u0014ÿQ\u0003\b:\u0014ÿQ\u0003\bOÈ(ÿQ\u0003\b:\u0014ÿQ\u0003\t1£\u0014ÿQ\u0003\tQ\u0014ÿQ\u0003\t¼Ò\u0014ÿQ\u0003\téæ\u0014ÿQ\u0003\n(³\u0014ÿQ\u0003\njb\u0014ÿQ\u0003\u000bCdÿQ\u0003\u0007Õ\u0014ÿQ\u0003\u0007Ì¸PÿQ\u0003\u0007¨&`ÿQ\u0003\b\u000eûpÿQ\u0003\u0007¹Ù`ÿQ\u0003\u0007é]pÿQ\u0003\u0007Ã=`ÿQ\u0003\u0007ä¦pÿQ\u0003\b\u000eû`ÿQ\u0003\u0007¹ÙpÿQ\u0003\u0007Õ`ÿQ\u0003\bQpÿQ\u0003\u0007¹Ù`ÿQ\u0003\u0007é]pÿQ\u0003\u0007Ã=`ÿQ\u0003\u0007ä¦pÿQ\u0003\b\u000eû`ÿQ\u0003\u0007Ó5\\ÿQ\u0003\u0007Õ\u0014ÿQ\u0003\u0007q}pÿQ\u0003\u0007\u0003\\ÿQ\u0003\bQ\u0004ÿQ\u0003\bOÈPÿQ\u0003\u0007é]xÿQ\u0003\u0007ßÈPÿQ\u0003\u0007Ö\u001fPÿQ\u0003\u0007Ì¸PÿQ\u0003\u0007Ã=PÿQ\u0003\u0007¹ÙPÿQ\u0003\u0007°¶PÿQ\u0003\u0007§~PÿQ\u0003\u0007](ÿQ\u0003\beqPÿQ\u0003\u0007Çø`ÿQ\u0003\bËàpÿQ\u0003\u0007íìPÿQ\u0003\u0007ÇøpÿQ\u0003\u0007÷¤pÿQ\u0003\t\bMpÿQ\u0003\u000bmªPÿQ\u0003\u0007íìRÿQ\u0003\b\u0001\u0017\u000eÿQ\u0003\b2Ñ`ÿQ\u0003\u0007í9xÿQ\u0003\bf;xÿQ\u0003\b,¢pÿQ\u0003\u0007äy`ÿQ\u0003\bGQpÿQ\u0003\u0007÷¤RÿQ\u0003\b\u0001\u0017\u000eÿQ\u0003\b2ÑpÿQ\u0003\u0007äy`ÿQ\u0003\bg7pÿQ\u0003\bq»PÿQ\u0003\u0007äy`ÿQ\u0003\bg7pÿQ\u0003\b%`ÿQ\u0003\bñ¶xÿQ\u0003\tu;xÿQ\u0003\b{`ÿQ\u0003\t$ypÿQ\u0003\b©\\pÿQ\u0003\beqbÿQ\u0003\u0007Ì¸\u000eÿQ\u0003\u0007ý(pÿQ\u0003\u0007¨&`ÿQ\u0003\b\u000eûpÿQ\u0003\u0007þñ\u000eÿ\u0006\u0006Teil 2bÿQ\u0003\b#UxÿQ\u0003\bR¯xÿQ\u0003\u0007é]pÿQ\u0003\u0007Ã=`ÿQ\u0003\u0007ä¦pÿQ\u0003\b\u000eûpÿQ\u0003\b\u001a\u001dxÿQ\u0003\bR¯xÿQ\u0003\u0007¹ÙpÿQ\u0003\u0007Õ`ÿQ\u0003\bE\u0004pÿQ\u0003\u0007ý(`ÿQ\u0003\bppÿQ\u0003\b0\u0001`ÿQ\u0003\tL+pÿQ\u0003\bÍ\u0018ÿQ\u0003\bOÈ<ÿQ\u0003\bE\u0004PÿQ\u0003\b:<ÿQ\u0003\b0\u0001PÿQ\u0003\b%<ÿQ\u0003\b\u001bgPÿQ\u0003\b\u0011)<ÿQ\u0003\b\u0007\u0005PÿQ\u0003\u0007ý(<ÿQ\u0003\u0007ó6PÿQ\u0003\u0007é]<ÿQ\u0003\u0007ßÈPÿQ\u0003\u0007Ö\u001f<ÿQ\u0003\u0007Ì¸PÿQ\u0003\u0007Ã=<ÿQ\u0003\u0007¹ÙPÿQ\u0003\u0007°¶<ÿQ\u0003\u0007§~xÿQ\u0003\u0007¿dÿQ\u0003\u0007tpXÿQ\u0003\u0007µÚxÿQ\u0003\u0007tpXÿQ\u0003\u0007µÚxÿQ\u0003\u0007tp\u0014ÿQ\u0003\u0007kPpÿQ\u0003\u0007bmpÿQ\u0003\u0007Y xÿQ\u0003\u0007¢dÿQ\u0003\u0007H\u001dpÿQ\u0003\u0007?pÿQ\u0003\u00077\u0013xÿQ\u0003\u0007tpdÿQ\u0003\u0007&2lÿQ\u0003\u0007kPxÿQ\u0003\u0007?(ÿQ\u0003\u0007H\u001d(ÿQ\u0003\u0007Pç\u0014ÿQ\u0003\u0007Y (ÿQ\u0003\u0007bm\u0014ÿQ\u0003\u0007kP(ÿQ\u0003\u0007tp\u0014ÿQ\u0003\u0007}V(ÿQ\u0003\u0007z\u0014ÿQ\u0003\u0007Ý(ÿQ\u0003\u0007/\u0014ÿQ\u0003\u0007¢\u0014ÿQ\u0003\bR}xÿQ\u0003\u0007µÚ­\u0001ÿQ\u0003\b\u0011)\u0001ÿQ\u0003\bùË\u0000ÿ/\u0000MTrk\u0000\u0000!\u0017\u0000ÿ\u0003\u000bPiano right\u0000À\u0000\u0000°\u0007d\u0000\n@\u0000>J\u0000ÿ\u0001 bdca426d104a26ac9dcb070447587523\u0000°[P=EP=\u0000\u0000<@P9=P69p6\u0000\u0000>\u0000\u0000<\u0000\u00009\u0000\u0000>5h>\u0000\u0000>5x>\u0000\u0000>5p>\u0000\u0000C8\u0000:/\u0000>/`>\u0000\u0000:\u0000\u0000C\u0000\u0000E5pE\u0000\u0000B2\u0000?3\u0000F<`F\u0000\u0000?\u0000\u0000B\u0000\u0000E4pE\u0000\u0000>0\u0000C9`C\u0000\u0000>\u0000\u0000C5pC\u0000\u0000<5\u0000>5h>\u0000\u0000<\u0000\u0000>5\u0000<5x<\u0000\u0000>\u0000\u0000<5\u0000>5p>\u0000\u0000<\u0000\u0000:/\u0000C8\u0000>8`>\u0000\u0000C\u0000\u0000:\u0000\u0000E5pE\u0000\u0000?<\u0000B3\u0000F<`F\u0000\u0000B\u0000\u0000?\u0000\u0000E3pE\u0000\u0000>)\u0000C9`C\u0000\u0000>\u0000\u0000C5pC\u0000\u0000:5\u0000>5h>\u0000\u0000:\u0000\u0000:5\u0000>5x>\u0000\u0000:\u0000\u0000>5\u0000:5p:\u0000\u0000>\u0000\u0000:<\u0000A<`A\u0000\u0000:\u0000\u0000A:pA\u0000\u00009:\u0000A:`A\u0000\u00009\u0000\u0000>9p>\u0000\u00005=\u0000:=`:\u0000\u00005\u0000\u0000:>p:\u0000\u0000:;\u00007;h:\u0000\u0000:;x:\u0000\u0000<5p<\u0000\u00007\u0000\u000051\u0000>1\u0000:1P:\u0000\u0000>\u0000\u00005\u0000\u00006.\u0000:7h:\u0000\u0000:2x:\u0000\u0000<8p<\u0000\u00006\u0000\u0000>9\u000050\u0000:9`:\u0000\u00005\u0000\u0000>\u0000\u0000A<pA\u0000\u0000?B\u000098\u0000CB`C\u0000\u00009\u0000\u0000?\u0000\u0000A8pA\u0000\u0000>:\u000051\u0000:4`:\u0000\u00005\u0000\u0000>\u0000\u0000<4p<\u0000\u000071\u0000:1h:\u0000\u0000:0x:\u0000\u00007\u0000\u0000<)\u00007)p7\u0000\u0000<\u0000\u00009(\u00006$\u0000>1p>\u0000\u00006\u0000\u00009\u0000\u0000J5hJ\u0000\u0000J5xJ\u0000\u0000J5pJ\u0000\u0000J/\u0000F/\u0000O8`O\u0000\u0000F\u0000\u0000J\u0000\u0000Q5pQ\u0000\u0000K/\u0000N3\u0000R<`R\u0000\u0000N\u0000\u0000K\u0000\u0000Q3pQ\u0000\u0000J0\u0000O9`O\u0000\u0000J\u0000\u0000O5pO\u0000\u0000H5\u0000J5hJ\u0000\u0000H\u0000\u0000H5\u0000J5xJ\u0000\u0000H\u0000\u0000J5\u0000H5pH\u0000\u0000J\u0000\u0000O8\u0000J8\u0000F/`F\u0000\u0000J\u0000\u0000O\u0000\u0000Q5pQ\u0000\u0000N4\u0000K4\u0000R<`R\u0000\u0000K\u0000\u0000N\u0000\u0000Q3pQ\u0000\u0000J9\u0000O9`O\u0000\u0000J\u0000\u0000O5pO\u0000\u0000F5\u0000J5hJ\u0000\u0000F\u0000\u0000J5\u0000F5xF\u0000\u0000J\u0000\u0000F5\u0000J5pJ\u0000\u0000F\u0000\u0000F<\u0000M<`M\u0000\u0000F\u0000\u0000M7pM\u0000\u0000E3\u0000M:`M\u0000\u0000E\u0000\u0000F=pF\u0000\u0000A7\u0000F=`F\u0000\u0000A\u0000\u0000F>pF\u0000\u0000C;\u0000F;hF\u0000\u0000F;xF\u0000\u0000H:pH\u0000\u0000C\u0000\u0000A1\u0000J1\u0000F1PF\u0000\u0000J\u0000\u0000A\u0000\u0000B.\u0000F7hF\u0000\u0000F2xF\u0000\u0000H8pH\u0000\u0000B\u0000\u0000J9\u0000A0\u0000F9`F\u0000\u0000A\u0000\u0000J\u0000\u0000M9pM\u0000\u0000K8\u0000E8\u0000OB`O\u0000\u0000E\u0000\u0000K\u0000\u0000M8pM\u0000\u0000J:\u0000A1\u0000F:`F\u0000\u0000A\u0000\u0000J\u0000\u0000H4pH\u0000\u0000C1\u0000F1hF\u0000\u0000F0xF\u0000\u0000H)pH\u0000\u0000C\u0000\u0000J1\u0000E(\u0000B(pB\u0000\u0000E\u0000\u0000J\u0000\u0000>5h>\u0000\u0000>5x>\u0000\u0000>5p>\u0000\u0000;/\u0000>/\u0000A8`A\u0000\u0000>\u0000\u0000;\u0000\u0000C64C\u0000\u0000F3<F\u0000\u0000><\u0000;3\u0000D<`D\u0000\u0000;\u0000\u0000>\u0000\u0000C7pC\u0000\u0000;0\u0000>0\u0000A9`A\u0000\u0000>\u0000\u0000;\u0000\u0000A5pA\u0000\u000085\u0000;5h;\u0000\u0000<5x<\u0000\u0000>5p>\u0000\u00008\u0000\u0000>/\u0000;/\u0000A8`A\u0000\u0000;\u0000\u0000>\u0000\u0000C:4C\u0000\u0000F3<F\u0000\u0000;3\u0000>5\u0000D<`D\u0000\u0000>\u0000\u0000;\u0000\u0000C7pC\u0000\u0000A9\u0000;0\u0000>0`>\u0000\u0000;\u0000\u0000A\u0000\u0000A5pA\u0000\u0000JC\u0000CC\u0000MO\u0000OOhO\u0000\u0000J\u0000\u0000C\u0000\u0000M\u0000\u0000F=\u0000OH\u0000C=\u0000MHxM\u0000\u0000F\u0000\u0000O\u0000\u0000C\u0000\u0000MH\u0000OH\u0000G=\u0000C=pC\u0000\u0000M\u0000\u0000O\u0000\u0000G\u0000\u0000WF\u0000[T\u0000OF\u0000TF`T\u0000\u0000W\u0000\u0000[\u0000\u0000O\u0000\u0000KD\u0000OD\u0000WQpW\u0000\u0000O\u0000\u0000K\u0000\u0000OH\u0000KH\u0000WUhW\u0000\u0000K\u0000\u0000O\u0000\u0000JC\u0000OC\u0000VOxV\u0000\u0000O\u0000\u0000J\u0000\u0000TO\u0000OC\u0000HCpH\u0000\u0000O\u0000\u0000T\u0000\u0000MH\u0000JH\u0000VUhV\u0000\u0000J\u0000\u0000M\u0000\u0000G>\u0000J>\u0000MJxM\u0000\u0000J\u0000\u0000G\u0000\u0000OJ\u0000J>\u0000G>4G\u0000\u0000J\u0000\u0000O\u0000\u0000RB<R\u0000\u0000PN\u0000J@\u0000G@hG\u0000\u0000J\u0000\u0000P\u0000\u0000OI\u0000G>\u0000J>xJ\u0000\u0000G\u0000\u0000O\u0000\u0000ME\u0000J:\u0000G:pG\u0000\u0000J\u0000\u0000M\u0000\u0000KP\u0000HBhH\u0000\u0000K\u0000\u0000K>\u0000H>\u0000NJxN\u0000\u0000H\u0000\u0000K\u0000\u0000K>\u0000OJ\u0000H>pH\u0000\u0000O\u0000\u0000K\u0000\u0000KH\u0000OH\u0000WUhW\u0000\u0000O\u0000\u0000K\u0000\u0000OC\u0000JC\u0000VOxV\u0000\u0000J\u0000\u0000O\u0000\u0000O?\u0000TK\u0000H?pH\u0000\u0000T\u0000\u0000O\u0000\u0000JH\u0000MH\u0000VUhV\u0000\u0000M\u0000\u0000J\u0000\u0000MJ\u0000J>\u0000G>xG\u0000\u0000J\u0000\u0000M\u0000\u0000OJ\u0000J>\u0000G>4G\u0000\u0000J\u0000\u0000O\u0000\u0000RC<R\u0000\u0000PN\u0000J@\u0000G@hG\u0000\u0000J\u0000\u0000P\u0000\u0000OI\u0000G>\u0000J>xJ\u0000\u0000G\u0000\u0000O\u0000\u0000G>\u0000MI\u0000J>pJ\u0000\u0000M\u0000\u0000G\u0000\u0000KP\u0000HBhH\u0000\u0000K\u0000\u0000H>\u0000K>\u0000NJxN\u0000\u0000K\u0000\u0000H\u0000\u0000OJ\u0000K>\u0000H>pH\u0000\u0000K\u0000\u0000O\u0000\u0000JD\u0000OD\u0000VQhO\u0000\u0000J\u0000\u0004OJpJJ\bO\u0000hJ\u0000\u0004V\u0000\u0000OD\u0000TQ\u0000HDhH\u0000\u0000O\u0000\u0004OFtHD\u0004O\u0000tH\u0000\u0000T\u0000\u0000Q><Q\u0000\u0000T><T\u0000\u0000J3\u0000H3\u0000N3\u0000QHhQ\u0000\u0000J\u0000\u0000H\u0000\u0000N\u0000\u0000OExO\u0000\u0000QEpQ\u0000\u0000FD\u0000RQ\u0000JDhJ\u0000\u0000F\u0000\u0004OJtJJ\u0004O\u0000lJ\u0000\u0000R\u0000\u0000RH\u0000JH\u0000OH\u0000VHhV\u0000\u0000R\u0000\u0000J\u0000\u0000O\u0000\u0000OExO\u0000\u0000JEpJ\u0000\u0000E9\u0000JD\u0000QDhJ\u0000\u0000E\u0000\u0000JJxJ\u0000\u0000E8pE\u0000\u0000Q\u0000\u0000I=\u0000O=\u0000QHhQ\u0000\u0000RExR\u0000\u0000QEpQ\u0000\u0000O\u0000\u0000I\u0000\u0000QC\u0000VO\u0000JCPJ\u0000\u0000V\u0000\u0000Q\u0000hJ?xJ\u0000\u0000J?pJ\u0000\u0000KG\u0000H<\u0000E<hK\u0000\u0000O=xO\u0000\u0000R=pR\u0000\u0000E\u0000\u0000H\u0000\u0000K:\u0000O:\u0000VEhV\u0000\u0000T?xT\u0000\u0000O?pO\u0000\u0000O\u0000\u0000K\u0000\u0000H8\u0000QG\u0000J8hQ\u0000\u0000R=xR\u0000\u0000Q=pQ\u0000\u0000J\u0000\u0000H\u0000\u0000K:\u0000N:\u0000TEhT\u0000\u0000N\u0000\u0000K\u0000\u0000J5\u0000N5\u0000R?xR\u0000\u0000N\u0000\u0000J\u0000\u0000H5\u0000N5\u0000Q?pQ\u0000\u0000N\u0000\u0000H\u0000\u0000F<\u0000J<\u0000OGhO\u0000\u0000J\u0000\u0000F\u0000\u0000H3\u0000K3\u0000Q=xQ\u0000\u0000K\u0000\u0000H\u0000\u0000F3\u0000J3\u0000O=pO\u0000\u0000J\u0000\u0000F\u0000\u0000E7\u0000M@\u0000H7hH\u0000\u0000M\u0000\u0000E\u0000\u0000J1\u0000O:\u0000F1xF\u0000\u0000O\u0000\u0000J\u0000\u0000K:\u0000C1\u0000F1pF\u0000\u0000C\u0000\u0000K\u0000\u0000J=\u0000B3\u0000E3PE\u0000\u0000B\u0000\u0000J\u0000h>/x>\u0000\u0000>/p>\u0000\u0000<5\u000095\u0000??h?\u0000\u0000C7xC\u0000\u0000F7pF\u0000\u00009\u0000\u0000<\u0000\u0000C3\u0000?3\u0000J=hJ\u0000\u0000H9xH\u0000\u0000C9pC\u0000\u0000?\u0000\u0000C\u0000\u0000E?\u0000>0\u0000<-hE\u0000\u0000F7xF\u0000\u0000E7pE\u0000\u0000<\u0000\u0000>\u0000\u0000B3\u0000?3\u0000H=hH\u0000\u0000F9xF\u0000\u0000E9pE\u0000\u0000?\u0000\u0000B\u0000\u0000C?\u0000:5\u0000>5h>\u0000\u0000:\u0000\u0000C\u0000\u0000E7\u0000<.\u0000?.x?\u0000\u0000<\u0000\u0000E\u0000\u0000C7\u0000:.\u0000>.p>\u0000\u0000:\u0000\u0000C\u0000\u0000A:\u000091\u0000<1h<\u0000\u00009\u0000\u0000A\u0000\u0000C3\u0000:*\u0000>*x>\u0000\u0000:\u0000\u0000C\u0000\u0000?3\u00007*\u0000:*p:\u0000\u00007\u0000\u0000?\u0000\u00009:\u0000>:P>\u0000\u00009\u0000\u0000<5\u0000>5h>\u0000\u0000>5x>\u0000\u0000>5p>\u0000\u0000<\u0000\u0000:/\u0000>/\u0000C8`C\u0000\u0000>\u0000\u0000:\u0000\u0000E8pE\u0000\u0000?3\u0000B<\u0000F<`F\u0000\u0000B\u0000\u0000?\u0000\u0000E7pE\u0000\u0000:0\u0000>0\u0000C9`C\u0000\u0000>\u0000\u0000:\u0000\u0000C5pC\u0000\u0000<5\u0000>5h>\u0000\u0000<\u0000\u0000<5\u0000>5x>\u0000\u0000<\u0000\u0000<1\u0000>1p>\u0000\u0000<\u0000\u0000:/\u0000>8\u0000C8`C\u0000\u0000>\u0000\u0000:\u0000\u0000E5pE\u0000\u0000?<\u0000B<\u0000F<`F\u0000\u0000B\u0000\u0000?\u0000\u0000E4pE\u0000\u0000:+\u0000>0\u0000C9`C\u0000\u0000>\u0000\u0000:\u0000\u0000C5pC\u0000\u0000:5\u0000>5h>\u0000\u0000:\u0000\u0000:5\u0000>5x>\u0000\u0000:\u0000\u0000:1\u0000>1p>\u0000\u0000:\u0000\u0000:<\u0000A<`A\u0000\u0000:\u0000\u0000A:pA\u0000\u00009:\u0000A:`A\u0000\u00009\u0000\u0000>7p>\u0000\u00005=\u0000:=`:\u0000\u00005\u0000\u0000:>p:\u0000\u00007;\u0000:;h:\u0000\u0000:;x:\u0000\u0000<:p<\u0000\u00007\u0000\u000051\u0000>1\u0000:1P:\u0000\u0000>\u0000\u00005\u0000\u00006.\u0000:7h:\u0000\u0000:2x:\u0000\u0000<8p<\u0000\u00006\u0000\u000050\u0000:9\u0000>9`>\u0000\u0000:\u0000\u00005\u0000\u0000A<pA\u0000\u000098\u0000?B\u0000CB`C\u0000\u0000?\u0000\u00009\u0000\u0000A8pA\u0000\u0000>:\u000051\u0000:3`:\u0000\u00005\u0000\u0000>\u0000\u0000<4p<\u0000\u000071\u0000:1h:\u0000\u0000:0x:\u0000\u00007\u0000\u00007)\u0000<)p<\u0000\u00007\u0000\u00006(\u00009(\u0000>1p>\u0000\u00009\u0000\u00006\u0000\u0000>5h>\u0000\u0000>5x>\u0000\u0000>5p>\u0000\u0000;/\u0000>/\u0000A8`A\u0000\u0000>\u0000\u0000;\u0000\u0000C84C\u0000\u0000F3<F\u0000\u0000;3\u0000><\u0000D<`D\u0000\u0000>\u0000\u0000;\u0000\u0000C8pC\u0000\u0000;0\u0000>0\u0000A9`A\u0000\u0000>\u0000\u0000;\u0000\u0000A5pA\u0000\u000085\u0000;5h;\u0000\u0000<5x<\u0000\u0000>5p>\u0000\u00008\u0000\u0000;/\u0000>/\u0000A8`A\u0000\u0000>\u0000\u0000;\u0000\u0000C:4C\u0000\u0000F3<F\u0000\u0000;3\u0000><\u0000D<`D\u0000\u0000>\u0000\u0000;\u0000\u0000C;pC\u0000\u0000;0\u0000>0\u0000A9`A\u0000\u0000>\u0000\u0000;\u0000\u0000A5pA\u0000\u0000CC\u0000JC\u0000MO\u0000OOhO\u0000\u0000C\u0000\u0000J\u0000\u0000M\u0000\u0000F=\u0000C=\u0000MH\u0000OHxO\u0000\u0000F\u0000\u0000C\u0000\u0000M\u0000\u0000C=\u0000G=\u0000MH\u0000OHpO\u0000\u0000C\u0000\u0000G\u0000\u0000M\u0000\u0000[T\u0000WF\u0000TF\u0000OF O\u0000\u0000[\u0000\u0000W\u0000\u0000T\u0000P?:\u0000C:\u0000E:hE\u0000\u0000C\u0000\u0000?\u0000\u0000?:\u0000C:\u0000FFxF\u0000\u0000C\u0000\u0000?\u0000\u0000?:\u0000C:\u0000HFpH\u0000\u0000C\u0000\u0000?\u0000\u0000??\u0000C?\u0000HM\u0000KMPK\u0000\u0000?\u0000\u0000C\u0000\u0000H\u0000\u0000C:\u0000E:\u0000HFhH\u0000\u0000E\u0000\u0000C\u0000\u0000C?\u0000E?\u0000JMxJ\u0000\u0000E\u0000\u0000C\u0000\u0000C>\u0000E>\u0000KJpK\u0000\u0000E\u0000\u0000C\u0000\u0000CF\u0000EF\u0000KR\u0000QRPQ\u0000\u0000C\u0000\u0000E\u0000\u0000K\u0000\u0000<:\u0000CE\u0000?:hC\u0000\u0000E<xE\u0000\u0000C:pC\u0000\u0000?\u0000\u0000<\u0000\u0000C:\u0000>:\u0000JEPJ\u0000\u0000>\u0000\u0000C\u0000h>9x>\u0000\u0000>9p>\u0000\u0000<8\u0000?C\u000098h?\u0000\u0000C;xC\u0000\u0000E;pE\u0000\u00009\u0000\u0000<\u0000\u0000?9\u0000C9\u0000HDhH\u0000\u0000J;xJ\u0000\u0000K;pK\u0000\u0000C\u0000\u0000?\u0000\u0000C8\u0000JC\u0000>8P>\u0000\u0000J\u0000\u0000C\u0000\u0000=7\u0000CBhC\u0000\u0000H8xH\u0000\u0000F;pF\u0000\u0000=\u0000\u0000<7\u0000DBhD\u0000\u0000H7xH\u0000\u0000C8xC\u0000\u0000<\u0000\u0000B7<B\u0000\u0000C8<C\u0000\u0000<4\u0000B>hB\u0000\u0000@7x@\u0000\u0000B7pB\u0000\u0000<\u0000\u0000C4\u0000:,P:\u0000\u0000C\u0000\u00009-\u0000<-\u0000>5h>\u0000\u0000<\u0000\u0000<-\u0000>5x>\u0000\u0000<\u0000\u0000<-\u0000>5p>\u0000\u0000<\u0000\u00009\u0000\u0000:/\u0000>/\u0000C8`C\u0000\u0000>\u0000\u0000:\u0000\u0000E7pE\u0000\u0000?3\u0000B2\u0000F<`F\u0000\u0000B\u0000\u0000?\u0000\u0000E5pE\u0000\u0000>0\u0000C9`C\u0000\u0000>\u0000\u0000C5pC\u0000\u0000<5\u0000>5h>\u0000\u0000<\u0000\u0000<5\u0000>5x>\u0000\u0000<\u0000\u0000<3\u0000>3p>\u0000\u0000<\u0000\u0000:/\u0000>8\u0000C8`C\u0000\u0000>\u0000\u0000:\u0000\u0000E7pE\u0000\u0000?3\u0000B3\u0000F<`F\u0000\u0000B\u0000\u0000?\u0000\u0000E3pE\u0000\u0000>3\u0000C9`C\u0000\u0000>\u0000\u0000C5pC\u0000\u0000:5\u0000>5h>\u0000\u0000:\u0000\u0000:5\u0000>5x>\u0000\u0000:\u0000\u0000:2\u0000>2p>\u0000\u0000:\u0000\u0000:<\u0000A<`A\u0000\u0000:\u0000\u0000A:pA\u0000\u00009:\u0000A:`A\u0000\u00009\u0000\u0000>8p>\u0000\u00005=\u0000:=`:\u0000\u00005\u0000\u0000:>p:\u0000\u00007;\u0000:;h:\u0000\u0000:;x:\u0000\u0000<:p<\u0000\u00007\u0000\u000051\u0000>1\u0000:1P:\u0000\u0000>\u0000\u00005\u0000\u00006.\u0000:7h:\u0000\u0000:2x:\u0000\u0000<8p<\u0000\u00006\u0000\u000050\u0000:9\u0000>9`>\u0000\u0000:\u0000\u00005\u0000\u0000A<pA\u0000\u000098\u0000?B\u0000CB`C\u0000\u0000?\u0000\u00009\u0000\u0000A8pA\u0000\u0000>:\u000051\u0000:4`:\u0000\u00005\u0000\u0000>\u0000\u0000<4p<\u0000\u000071\u0000:1h:\u0000\u0000:0x:\u0000\u00007\u0000\u00007)\u0000<)p<\u0000\u00007\u0000\u00006&\u00009(\u0000>1p>\u0000\u00009\u0000\u00006\u0000\u0000J5hJ\u0000\u0000J5xJ\u0000\u0000J5pJ\u0000\u0000F/\u0000J/\u0000O8`O\u0000\u0000J\u0000\u0000F\u0000\u0000Q5pQ\u0000\u0000K3\u0000N3\u0000R<`R\u0000\u0000N\u0000\u0000K\u0000\u0000Q2pQ\u0000\u0000J0\u0000O9`O\u0000\u0000J\u0000\u0000O5pO\u0000\u0000H5\u0000J5hJ\u0000\u0000H\u0000\u0000H5\u0000J5xJ\u0000\u0000H\u0000\u0000H5\u0000J5pJ\u0000\u0000H\u0000\u0000F/\u0000J8\u0000O8`O\u0000\u0000J\u0000\u0000F\u0000\u0000Q7pQ\u0000\u0000K3\u0000N3\u0000R<`R\u0000\u0000N\u0000\u0000K\u0000\u0000Q5pQ\u0000\u0000J9\u0000O9`O\u0000\u0000J\u0000\u0000O5pO\u0000\u0000F5\u0000J5hJ\u0000\u0000F\u0000\u0000F5\u0000J5xJ\u0000\u0000F\u0000\u0000F5\u0000J5pJ\u0000\u0000F\u0000\u0000F<\u0000M<`M\u0000\u0000F\u0000\u0000M:pM\u0000\u0000E:\u0000M:`M\u0000\u0000E\u0000\u0000F=pF\u0000\u0000A6\u0000F=`F\u0000\u0000A\u0000\u0000F>pF\u0000\u0000C;\u0000F;hF\u0000\u0000F;xF\u0000\u0000H:pH\u0000\u0000C\u0000\u0000A1\u0000J1\u0000F1PF\u0000\u0000J\u0000\u0000A\u0000\u0000B.\u0000F7hF\u0000\u0000F2xF\u0000\u0000H8pH\u0000\u0000B\u0000\u0000A0\u0000F9\u0000J9`J\u0000\u0000F\u0000\u0000A\u0000\u0000M<pM\u0000\u0000E8\u0000KB\u0000OB`O\u0000\u0000K\u0000\u0000E\u0000\u0000M8pM\u0000\u0000J:\u0000A1\u0000F:`F\u0000\u0000A\u0000\u0000J\u0000\u0000H4pH\u0000\u0000C1\u0000F1hF\u0000\u0000F0xF\u0000\u0000H)pH\u0000\u0000C\u0000\u0000B(\u0000E(\u0000J1pJ\u0000\u0000E\u0000\u0000B\u0000\u0000>5h>\u0000\u0000>5x>\u0000\u0000>5p>\u0000\u0000;/\u0000>/\u0000A8`A\u0000\u0000>\u0000\u0000;\u0000\u0000C74C\u0000\u0000F3<F\u0000\u0000;3\u0000><\u0000D<`D\u0000\u0000>\u0000\u0000;\u0000\u0000C7pC\u0000\u0000;0\u0000>0\u0000A9`A\u0000\u0000>\u0000\u0000;\u0000\u0000A5pA\u0000\u000085\u0000;5h;\u0000\u0000<5x<\u0000\u0000>5p>\u0000\u00008\u0000\u0000;/\u0000>/\u0000A8`A\u0000\u0000>\u0000\u0000;\u0000\u0000C94C\u0000\u0000F3<F\u0000\u0000;3\u0000><\u0000D<`D\u0000\u0000>\u0000\u0000;\u0000\u0000C7pC\u0000\u0000;0\u0000>0\u0000A9`A\u0000\u0000>\u0000\u0000;\u0000\u0000A5pA\u0000\u0000CC\u0000JC\u0000MO\u0000OOhO\u0000\u0000C\u0000\u0000J\u0000\u0000M\u0000\u0000F=\u0000C=\u0000MH\u0000OHxO\u0000\u0000F\u0000\u0000C\u0000\u0000M\u0000\u0000C=\u0000G=\u0000MH\u0000OHpO\u0000\u0000C\u0000\u0000G\u0000\u0000M\u0000\u0000[T\u0000WF\u0000TF\u0000OF`O\u0000\u0000[\u0000\u0000W\u0000\u0000T\u0000\u0000WQ\u0000OD\u0000KDpK\u0000\u0000O\u0000\u0000W\u0000\u0000OH\u0000KH\u0000WUhW\u0000\u0000K\u0000\u0000O\u0000\u0000JC\u0000OC\u0000VOxV\u0000\u0000O\u0000\u0000J\u0000\u0000TO\u0000OC\u0000HCpH\u0000\u0000O\u0000\u0000T\u0000\u0000MH\u0000JH\u0000VUhV\u0000\u0000J\u0000\u0000M\u0000\u0000G>\u0000J>\u0000MJxM\u0000\u0000J\u0000\u0000G\u0000\u0000OJ\u0000J>\u0000G>4G\u0000\u0000J\u0000\u0000O\u0000\u0000RB<R\u0000\u0000J@\u0000G@\u0000PNhP\u0000\u0000G\u0000\u0000J\u0000\u0000G>\u0000J>\u0000OIxO\u0000\u0000J\u0000\u0000G\u0000\u0000MI\u0000J>\u0000G>pG\u0000\u0000J\u0000\u0000M\u0000\u0000KP\u0000HBhH\u0000\u0000K\u0000\u0000H>\u0000K>\u0000NJxN\u0000\u0000K\u0000\u0000H\u0000\u0000OJ\u0000K>\u0000H>pH\u0000\u0000K\u0000\u0000O\u0000\u0000OH\u0000KH\u0000WUhW\u0000\u0000K\u0000\u0000O\u0000\u0000JC\u0000OC\u0000VOxV\u0000\u0000O\u0000\u0000J\u0000\u0000TO\u0000OC\u0000HCpH\u0000\u0000O\u0000\u0000T\u0000\u0000MH\u0000JH\u0000VUhV\u0000\u0000J\u0000\u0000M\u0000\u0000G>\u0000J>\u0000MJxM\u0000\u0000J\u0000\u0000G\u0000\u0000OJ\u0000J>\u0000G>4G\u0000\u0000J\u0000\u0000O\u0000\u0000RC<R\u0000\u0000J@\u0000G@\u0000PNhP\u0000\u0000G\u0000\u0000J\u0000\u0000G>\u0000J>\u0000OIxO\u0000\u0000J\u0000\u0000G\u0000\u0000MI\u0000J>\u0000G>pG\u0000\u0000J\u0000\u0000M\u0000\u0000KP\u0000HBhH\u0000\u0000K\u0000\u0000H>\u0000K>\u0000NJxN\u0000\u0000K\u0000\u0000H\u0000\u0000OJ\u0000K>\u0000H>pH\u0000\u0000K\u0000\u0000O\u0000\u0000JD\u0000VQ\u0000ODhO\u0000\u0000J\u0000\u0004OJpJD\bO\u0000hJ\u0000\u0004V\u0000\u0000TQ\u0000HD\u0000ODhO\u0000\u0000H\u0000\u0004OFtHB\u0004O\u0000tH\u0000\u0000T\u0000\u0000Q><Q\u0000\u0000T><T\u0000\u0000H3\u0000J3\u0000N3\u0000QHhQ\u0000\u0000H\u0000\u0000J\u0000\u0000N\u0000\u0000OExO\u0000\u0000QEpQ\u0000\u0000RQ\u0000FD\u0000JDhJ\u0000\u0000F\u0000\u0004OGtJD\u0004O\u0000lJ\u0000\u0000R\u0000\u0000JH\u0000OH\u0000RH\u0000VHhV\u0000\u0000J\u0000\u0000O\u0000\u0000R\u0000\u0000OExO\u0000\u0000JEpJ\u0000\u0000QD\u0000E9\u0000JDhJ\u0000\u0000E\u0000\u0000JJxJ\u0000\u0000EBpE\u0000\u0000Q\u0000\u0000I=\u0000O=\u0000QHhQ\u0000\u0000RExR\u0000\u0000QEpQ\u0000\u0000O\u0000\u0000I\u0000\u0000VO\u0000QC\u0000JCPJ\u0000\u0000Q\u0000\u0000V\u0000hJ?xJ\u0000\u0000J?pJ\u0000\u0000H<\u0000E<\u0000KGhK\u0000\u0000O=xO\u0000\u0000R=pR\u0000\u0000E\u0000\u0000H\u0000\u0000K:\u0000O:\u0000VEhV\u0000\u0000T?xT\u0000\u0000O?pO\u0000\u0000O\u0000\u0000K\u0000\u0000J4\u0000H4\u0000QGhQ\u0000\u0000R=xR\u0000\u0000Q=pQ\u0000\u0000H\u0000\u0000J\u0000\u0000K:\u0000N:\u0000TEhT\u0000\u0000N\u0000\u0000K\u0000\u0000J5\u0000N5\u0000R?xR\u0000\u0000N\u0000\u0000J\u0000\u0000H5\u0000N5\u0000Q?pQ\u0000\u0000N\u0000\u0000H\u0000\u0000F<\u0000J<\u0000OGhO\u0000\u0000J\u0000\u0000F\u0000\u0000H3\u0000K3\u0000Q=xQ\u0000\u0000K\u0000\u0000H\u0000\u0000F3\u0000J3\u0000O=pO\u0000\u0000J\u0000\u0000F\u0000\u0000E7\u0000H7\u0000M@hM\u0000\u0000H\u0000\u0000E\u0000\u0000F1\u0000J1\u0000O:xO\u0000\u0000J\u0000\u0000F\u0000\u0000C1\u0000F1\u0000K:pK\u0000\u0000F\u0000\u0000C\u0000\u0000B3\u0000E3\u0000J=PJ\u0000\u0000E\u0000\u0000B\u0000h>/x>\u0000\u0000>/p>\u0000\u0000<5\u000095\u0000??h?\u0000\u0000C7xC\u0000\u0000F7pF\u0000\u00009\u0000\u0000<\u0000\u0000?3\u0000C3\u0000J=hJ\u0000\u0000H9xH\u0000\u0000C9pC\u0000\u0000C\u0000\u0000?\u0000\u0000>2\u0000</\u0000E?hE\u0000\u0000F7xF\u0000\u0000E7pE\u0000\u0000<\u0000\u0000>\u0000\u0000?3\u0000B3\u0000H=hH\u0000\u0000F9xF\u0000\u0000E9pE\u0000\u0000B\u0000\u0000?\u0000\u0000:5\u0000>5\u0000C?hC\u0000\u0000>\u0000\u0000:\u0000\u0000<.\u0000?.\u0000E7xE\u0000\u0000?\u0000\u0000<\u0000\u0000:.\u0000>.\u0000C7pC\u0000\u0000>\u0000\u0000:\u0000\u000091\u0000<1\u0000A:hA\u0000\u0000<\u0000\u00009\u0000\u0000:*\u0000>*\u0000C3xC\u0000\u0000>\u0000\u0000:\u0000\u00007*\u0000:*\u0000?3p?\u0000\u0000:\u0000\u00007\u0000\u00009:\u0000>:P>\u0000\u00009\u0000\u0000<5\u0000>5h>\u0000\u0000>5x>\u0000\u0000>5p>\u0000\u0000<\u0000\u0000:/\u0000>/\u0000C8`C\u0000\u0000>\u0000\u0000:\u0000\u0000E8pE\u0000\u0000?3\u0000B<\u0000F<`F\u0000\u0000B\u0000\u0000?\u0000\u0000E;pE\u0000\u0000:0\u0000>0\u0000C9`C\u0000\u0000>\u0000\u0000:\u0000\u0000C5pC\u0000\u0000<5\u0000>5h>\u0000\u0000<\u0000\u0000<5\u0000>5x>\u0000\u0000<\u0000\u0000<3\u0000>3p>\u0000\u0000<\u0000\u0000:/\u0000>8\u0000C8`C\u0000\u0000>\u0000\u0000:\u0000\u0000E8pE\u0000\u0000?<\u0000B<\u0000F<`F\u0000\u0000B\u0000\u0000?\u0000\u0000E5pE\u0000\u0000:0\u0000>0\u0000C9`C\u0000\u0000>\u0000\u0000:\u0000\u0000C5pC\u0000\u0000:5\u0000>5h>\u0000\u0000:\u0000\u0000:5\u0000>5x>\u0000\u0000:\u0000\u0000:1\u0000>1p>\u0000\u0000:\u0000\u0000:<\u0000A<`A\u0000\u0000:\u0000\u0000A:pA\u0000\u00009:\u0000A:`A\u0000\u00009\u0000\u0000>;p>\u0000\u00005=\u0000:=`:\u0000\u00005\u0000\u0000:>p:\u0000\u00007;\u0000:;h:\u0000\u0000:;x:\u0000\u0000<:p<\u0000\u00007\u0000\u000051\u0000>1\u0000:1P:\u0000\u0000>\u0000\u00005\u0000\u00006.\u0000:7h:\u0000\u0000:2x:\u0000\u0000<8p<\u0000\u00006\u0000\u000050\u0000:9\u0000>9`>\u0000\u0000:\u0000\u00005\u0000\u0000A<pA\u0000\u000098\u0000?B\u0000CB`C\u0000\u0000?\u0000\u00009\u0000\u0000A8pA\u0000\u0000>:\u000051\u0000:2`:\u0000\u00005\u0000\u0000>\u0000\u0000<4p<\u0000\u000071\u0000:1h:\u0000\u0000:0x:\u0000\u00007\u0000\u00007)\u0000<)p<\u0000\u00007\u0000\u00006(\u00009(\u0000>1p>\u0000\u00009\u0000\u00006\u0000\u0000>5h>\u0000\u0000>5x>\u0000\u0000>5p>\u0000\u0000;/\u0000>/\u0000A8`A\u0000\u0000>\u0000\u0000;\u0000\u0000C:4C\u0000\u0000F3<F\u0000\u0000;3\u0000><\u0000D<`D\u0000\u0000>\u0000\u0000;\u0000\u0000C7pC\u0000\u0000;0\u0000>0\u0000A9`A\u0000\u0000>\u0000\u0000;\u0000\u0000A5pA\u0000\u000085\u0000;5h;\u0000\u0000<5x<\u0000\u0000>5p>\u0000\u00008\u0000\u0000;/\u0000>/\u0000A8`A\u0000\u0000>\u0000\u0000;\u0000\u0000C:4C\u0000\u0000F3<F\u0000\u0000;3\u0000><\u0000D<`D\u0000\u0000>\u0000\u0000;\u0000\u0000C8pC\u0000\u0000;0\u0000>0\u0000A9`A\u0000\u0000>\u0000\u0000;\u0000\u0000A5pA\u0000\u0000CC\u0000JC\u0000MO\u0000OOhO\u0000\u0000C\u0000\u0000J\u0000\u0000M\u0000\u0000F=\u0000C=\u0000MH\u0000OHxO\u0000\u0000F\u0000\u0000C\u0000\u0000M\u0000\u0000C=\u0000G=\u0000MH\u0000OHpO\u0000\u0000C\u0000\u0000G\u0000\u0000M\u0000\u0000[T\u0000WF\u0000TF\u0000OF O\u0000\u0000[\u0000\u0000W\u0000\u0000T\u0000P?:\u0000C:\u0000E:hE\u0000\u0000C\u0000\u0000?\u0000\u0000?:\u0000C:\u0000FFxF\u0000\u0000C\u0000\u0000?\u0000\u0000?:\u0000C:\u0000HFpH\u0000\u0000C\u0000\u0000?\u0000\u0000??\u0000C?\u0000HM\u0000KMPK\u0000\u0000?\u0000\u0000C\u0000\u0000H\u0000\u0000C:\u0000E:\u0000HFhH\u0000\u0000E\u0000\u0000C\u0000\u0000C?\u0000E?\u0000JMxJ\u0000\u0000E\u0000\u0000C\u0000\u0000C>\u0000E>\u0000KJpK\u0000\u0000E\u0000\u0000C\u0000\u0000CF\u0000EF\u0000KR\u0000QRPQ\u0000\u0000C\u0000\u0000E\u0000\u0000K\u0000\u0000<:\u0000CE\u0000?:hC\u0000\u0000E<xE\u0000\u0000C:pC\u0000\u0000?\u0000\u0000<\u0000\u0000C:\u0000>:\u0000JEPJ\u0000\u0000>\u0000\u0000C\u0000h>9x>\u0000\u0000>5p>\u0000\u0000<8\u0000?C\u000098h?\u0000\u0000C;xC\u0000\u0000E;pE\u0000\u00009\u0000\u0000<\u0000\u0000?9\u0000C9\u0000HDhH\u0000\u0000J;xJ\u0000\u0000K;pK\u0000\u0000C\u0000\u0000?\u0000\u0000C8\u0000JC\u0000>8P>\u0000\u0000J\u0000\u0000C\u0000\u0000=7\u0000CBhC\u0000\u0000H8xH\u0000\u0000F;pF\u0000\u0000=\u0000\u0000<7\u0000DBhD\u0000\u0000H7xH\u0000\u0000C8xC\u0000\u0000<\u0000\u0000B7<B\u0000\u0000C8<C\u0000\u0000<4\u0000B>hB\u0000\u0000@7x@\u0000\u0000B7pB\u0000\u0000<\u0000\u0000C4\u0000:,P:\u0000\u0000C\u0000\u00009-\u0000<-\u0000>5h>\u0000\u0000<\u0000\u0000<-\u0000>5x>\u0000\u0000<\u0000\u0000<1\u0000>1p>\u0000\u0000<\u0000\u00009\u0000\u0000:/\u0000>/\u0000C8`C\u0000\u0000>\u0000\u0000:\u0000\u0000E7pE\u0000\u0000?3\u0000B<\u0000F<`F\u0000\u0000B\u0000\u0000?\u0000\u0000E6pE\u0000\u0000>0\u0000C9`C\u0000\u0000>\u0000\u0000C5pC\u0000\u00009(\u0000<5\u0000>5h>\u0000\u0000<\u0000\u0000<5\u0000>5x>\u0000\u0000<\u0000\u0000<5\u0000>5p>\u0000\u0000<\u0000\u00009\u0000\u0000:/\u0000>8\u0000C8`C\u0000\u0000>\u0000\u0000:\u0000\u0000E64E\u0000\u0000H,<?8\u0000B8\u0000F8\u0014H\u0000L?\u0000\u0000B\u0000\u0000F\u0000\u0000E4pE\u0000\u0000:1\u0000>1\u0000C1pC\u0000\u0000>\u0000\u0000:\u0000\u000061x6\u0000\u000071x7\u0000\u000092x9\u0000\u0000:3x:\u0000\u0000=4x=\u0000\u0000>5x>\u0000\u000092x9\u0000\u0000:3x:\u0000\u0000=3x=\u0000\u0000>4x>\u0000\u0000B5xB\u0000\u0000C8xC\u0000\u0000=3x=\u0000\u0000>4x>\u0000\u0000B8xB\u0000\u0000C9xC\u0000\u0000E;xE\u0000\u0000F<xF\u0000\u0000B8xB\u0000\u0000C:xC\u0000\u0000E;xE\u0000\u0000F<xF\u0000\u0000I=xI\u0000\u0000J>xJ\u0000\u0000E9xE\u0000\u0000F:xF\u0000\u0000I<xI\u0000\u0000J=xJ\u0000\u0000N@xN\u0000\u0000OBxO\u0000\u0000I;xI\u0000\u0000J=xJ\u0000\u0000N?xN\u0000\u0000O@xO\u0000\u0000QBxQ\u0000\u0000RDxR\u0000\u0000N@xN\u0000\u0000OBxO\u0000\u0000QExQ\u0000\u0000RHxR\u0000\u0000UJxU\u0000\u0000VMxV\u0000\u0000QJxQ\u0000\u0000RMxR\u0000\u0000UQxU\u0000\u0000VSxV\u0000\u0000[]P[\u0000\u0000HO\u0000JO\u0000N]\u0000V]\u0000Q]PQ\u0000\u0000N\u0000\u0000H\u0000\u0000J\u0000\u0000V\u0000\u0000FO\u0000JO\u0000R]\u0000O]PO\u0000\u0000F\u0000\u0000J\u0000\u0000R\u0000\u0000EO\u0000HO\u0000J]\u0000N]PN\u0000\u0000E\u0000\u0000H\u0000\u0000J\u0000\u0000CV\u0000FV\u0000J`\u0000Of O\u0000\u0000C\u0000\u0000F\u0000\u0000J\u0000\u0000[l\u0000Vf\u0000OZ O\u0000\u0000V\u0000\u0000[\u0000\u0000ÿ/\u0000MTrk\u0000\u0000\u001fB\u0000ÿ\u0003\nPiano left\u0000À\u0000\u0000°\u0007d\u0000\n@x@\u00182-\r°@\u0000\u00072\u0000\u00005'<5\u0000\u0000318°@\u00183\u0000\u000020\u0011°@\u0000\u0019@&2\u0000\u00002(\u0000+\"\u00007(\f°@\u0000@E7\u0000\u0000+\u0000\u00002\u0000\u0000+&\u0000</\r°@\u0000\u0013@0<\u0000\u0000+\u0000\u0000+#\u0000:)\t°@\u0000\t@>:\u0000\u0000+\u0000\u0000+#\u00009)\u00002)\r°@\u0000H@{2\u0000\u00009\u0000\u0000+\u0000\u00002(\u0000+\"\u00007(\u0012°@\u0000C@{7\u0000\u0000+\u0000\u00002\u0000\u0000</\u0000+&\t°@\u0000g@`+\u0000\u0000<\u0000\u0000+#\u0000:)\u0006°@\u0000e@c:\u0000\u0002+\u0000\u00006)\u0000*#\u0011°@\u0000\u0012@-*\u0000\u00006\u0000\u0000)'\u000020\u000b°@\u00000@\u00152\u0000\u00003*\u0011°@\u0000m@R3\u0000\u0000)\u0000\u0000.0\u000020\u000b°@\u0000`@e2\u0000\u0000.\u0000\u0000''\u0000.0\u0007°@\u0000N@{.\u0000\u0000'\u0000\u0000\"$\u0000.*\u0010°@\u0000Z@f.\u0000\u0000\"\u0000\u0000'0\u0000.0\u0007°@\u0000q@X.\u0000\u0000'\u0000\u0000\"&\u0000..\u0007°@\u0000\u0019@0.\u0000\u0000\"\u0000\u00000.\u0000)&\u000e°@\u0000:@\b)\u0000\u00000\u0000\u0000.0\t°@\u0000b@e.\u0000\u0000'!\u00003'\u0012°@\u0000\u0002@L3\u0000\u0000'\u0000\u0000-%p-\u0000\u0000&\u001f\u00002&\u0011°@\u0000\u0007@X2\u0000\u0000&\u0000\u0012°@\u0000&@\u00187\"\u0000C(\u0000>(\n°@\u0000Y@m>\u0000\u0000C\u0000\u00007\u0000\u00007$\u0000H,\u000e°@\u0000}@EH\u0000\u00007\u0000\u00007#\u0000F)\u000b°@\u0000`@eF\u0000\u00007\u0000\u0000>)\u0000E)\u00007#\u0010°@\u0000\u0002@>7\u0000\u0000E\u0000\u0000>\u0000\u00007\"\u0000°@\u0000\u0000C(\u0000>( °@0>\u0000\u0000C\u0000\u00007\u0000\u00007!\u0000H,\r°@\u0000;@\bH\u0000\u00007\u0000\u00007#\u0000F)\f°@\u0000_@cF\u0000\u00027\u0000\u0000B)\u00006#\u0015°@\u0000\u0007@46\u0000\u0000B\u0000\u00005$\u0000>0\u0007°@\u0000A@\b>\u0000\u0000?*\u0011°@\u0000R@m?\u0000\u00005\u0000\u0000>0\u0000:0\t°@\u0000b@e:\u0000\u0000>\u0000\u0000?0\u00003'\u000f°@\u0000\r@43\u0000\u0000?\u0000\u0000.$\u0000:*\n°@\u0000'@\u001f:\u0000\u0000.\u0000\u000030\u0000:0\u0011°@\u0000D@{:\u0000\u00003\u0000\u0000.&\u0000:.\u000e°@\u0000\u001f@#:\u0000\u0000.\u0000\u00005&\u0000<.\u000b°@\u0000K@z<\u0000\u00005\u0000\u0000:0\f°@\u0000_@e:\u0000\u0000?'\u00003!\u0012°@\u0000f@h3\u0000\u0000?\u0000\u00009%p9\u0000\u00002\u001f\u0000>&\u0011°@\u0000l@l>\u0000\u00072\u0000\u0011°@\u0000'@\u00188(\u0000+\"\u000f°@\u0000^@r@\u0000w@J+\u0000\u00008\u0000\u00007)\u0000+#\t°@\u0000'@ 7\u0000\u00005)\u0010°@\u0000\f@L5\u0000\u00003'x3\u0000\u00002(p2\u0000\u0000+\u0000\u00008(\u0000+\"\u000f°@\u00008@\u0015@\u0000w@M+\u0000\u00008\u0000\u00008)\u0000+#\f°@\u0000_@)+\u0000\u0000+B<8\u0000\u0000;>\u0011°@\u0000V@\u0001;\u0000\u0000=7x=\u0000\u0000>9p>\u0000\u0000+\u0000\u0000$E\u00000E\r°@\u0000+@(0\u0000\u0000$\u0000\u0000<2\u0000H<\u0000C2pC\u0000\u0000H\u0000\u0000<\u0000\u0000C5\u0000H@\u0000<5\u0010°@\u0000\u0004@T<\u0000\u0000H\u0000\u0000C\u0000\u0000C4\u0000<4\u0000H>xH\u0000\u0000<\u0000\u0000C\u0000\u0000C>\u0000<4p<\u0000\u0000C\u0000\u0000<5\u0000G@\u0000D5\t°@\u0000S@\fD\u0000\u0000G\u0000\u0000<\u0000\u0000<1\u0000D;xD\u0000\u0000<\u0000\u0000<1\u0000D;4D\u0000\u0000<\u0000<<2\u0000D=\n°@\u0000\u0007@WD\u0000\u0000<\u0000\u0000D:\u0000<0x<\u0000\u0000D\u0000\u0000D7\u0000<-p<\u0000\u0000D\u0000\u0000°@\u0000\u0000C=\u0000<2\u0015°@S<\u0000\u0000C\u0000\u0000C9\u0000</x<\u0000\u0000C\u0000\u0000C9\u0000</p<\u0000\u0000C\u0000\u0000C5\u0000H@\u0000<5\u0013°@\u0000\u0013@B<\u0000\u0000H\u0000\u0000C\u0000\u0000<4\u0000H>\u0000C4xC\u0000\u0000H\u0000\u0000<\u0000\u0000<1\u0000C1pC\u0000\u0000<\u0000\u0000D5\u0000G@\u0000<5\n°@\u0000R@\f<\u0000\u0000G\u0000\u0000D\u0000\u0000<1\u0000D;xD\u0000\u0000<\u0000\u0000<1\u0000D;4D\u0000\u0000<\u0000<<2\u0000D=\u000b°@\u0000m@pD\u0000\u0000<\u0000\u0000D:\u0000<0x<\u0000\u0000D\u0000\u0000D:\u0000<0p<\u0000\u0000D\u0000\u0000<2\u0000C=\u000b°@\u0000v@OC\u0000\u0000<\u0000\u0000:2\u0000C2\u0019°@\u0000c@TC\u0000\u0000:\u0000\u000092\u0000?2\n°@\u0000\n@<?\u0000\u0000>2\u000b°@\u0000U@p>\u0000\u00009\u0000\u0000>2\u000072\u000b°@\u0000&@\u001f>\u0000\u0000@2\r°@\u0000@@\u0003@\u0000\u00007\u0000\u000092\u0000A2\u000b°@\u0000&@\u001fA\u0000\u0000@2\u0011°@\u0000\u0003@L@\u0000\u0000C-pC\u0000\u00009\u0000\u0000>5\u0000B-\u0000°@\u0000P@\u0018>\u0000\u0000=2\n°@\u0000n=\u0000\u0000>2p>\u0000\u0000B\u0000\u0000>5x°@X>\u0000\u0000C1\u0000>1\b°@\u0000H@\u0018>\u0000\u0000=*x=\u0000\u0000>1p>\u0000\u0000>2\u0011°@\u0000~@A>\u0000\u0000C\u0000\u0000>1\u0000B1\t°@\u0000o@p>\u0000\u0000=*x=\u0000\u0000>1p>\u0000\u0000>2\u0011°@\u0000\u0015@*>\u0000\u0000B\u0000\u0000>1\f°@\u0000\u0016@F>\u0000\u0000>*x>\u0000\u0000>1p>\u0000\u0000>2\u000f°@\u0000~@C>\u0000\u0000>1\u000b°@\u0000\u001e@?>\u0000\u0000?'x?\u0000\u0000>,p>\u0000\u0000>,\u000f°@\u00007@*>\u0000`7,\u00002,\b°@\u0000\r@S2\u0000\u00001&x1\u0000\u00002,p2\u0000\u00002-\u0013°@\u0000C@z2\u0000\u00007\u0000\u00006,\u00002,\u0007°@\u0000}@d2\u0000\u00001&x1\u0000\u00002,p2\u0000\u00002-\u000f°@\u0000R@o2\u0000\u00006\u0000\u00002,\t°@\u0000S@\f2\u0000\u00002&x2\u0000\u00002,p2\u0000\u00002-\u0013°@\u0000z@C2\u0000\u0000°@\u0000\u00006&\u00002&3°@E2\u0000\u00001&x1\u0000\u00002&x2\u0000\u00001%x1\u0000\u00002&x2\u0000\u00001!x1\u0000\u00006\u0000\u00006&\u00003&\b°@\u0000N@\"3\u0000\u00002!x2\u0000\u00001&x1\u0000\u00002\"x2\u0000\u00003&x3\u0000\u00002\u001ex2\u0000\u00006\u0000\u00007%\u0000+%\b°@\u00007@97\u0000\u00006(x6\u0000\u00005)x5\u0000\u00004*x4\u0000\u00003,x3\u0000\u00002/x2\u0000\u0000+\u0000\u00001/\u0011°@\u0000E@\"1\u0000\u00002,x2\u0000\u00003*x3\u0000\u00004)x4\u0000\u00005(x5\u0000\u00006(x6\u0000\u0000+)\u00007)\t°@\u00006@97\u0000\u00006$x6\u0000\u00007)x7\u0000\u00002%x2\u0000\u00001)x1\u0000\u00002$x2\u0000\u0000+\u0000\u00006)\u000e°@\u0000B@(6\u0000\u00002(x2\u0000\u00001)x1\u0000\u00002$x2\u0000\u00003*x3\u0000\u00002)x2\u0000\u00007%\u0000+%\b°@\u0000,@D7\u0000\u00006(x6\u0000\u00005*x5\u0000\u00004,x4\u0000\u00003,x3\u0000\u00002*x2\u0000\u0000+\u0000\u00001*\u0010°@\u0000F@\"1\u0000\u00002)x2\u0000\u00003(x3\u0000\u00004%x4\u0000\u00005)x5\u0000\u00006)x6\u0000\u0000+)\u00007)\f°@\u0000-@?7\u0000\u00006$x6\u0000\u00007)x7\u0000\u00002%x2\u0000\u00001/x1\u0000\u00002*x2\u0000\u0000+\u0000\u0000*/\u0011°@\u0000P@\u00172)x2\u0000\u00001(x1\u0000\u00002)x2\u0000\u00003)x3\u0000\u00002)x2\u0000\u0000*\u0000\u000021\u0000)1\b°@\u00001@?2\u0000\u00001(x1\u0000\u000021x2\u0000\u00001(x1\u0000\u000020x2\u0000\u00001*x1\u0000\u0000)\u0000\u00003/\u0000)/\t°@\u0000M@\"3\u0000\u00002)x2\u0000\u000030x3\u0000\u00000)x0\u0000\u00005/x5\u0000\u00003)x3\u0000\u0000)\u0000\u0000./\u00002/\u0007°@\u00008@92\u0000\u00001&x1\u0000\u00002/x2\u0000\u00001&x1\u0000\u00002/x2\u0000\u00001)x1\u0000\u0000.\u0000\u0000'/\u000f°@\u0000<@-'\u0000\u0000-)x-\u0000\u0000.0x.\u0000\u00003,x3\u0000\u000074x7\u0000\u00003)x3\u0000\u0000.1\u0007°@\u0000D@-.\u0000\u00004,x4\u0000\u000050x5\u0000\u00001)x1\u0000\u000021x2\u0000\u0000.)x.\u0000\u0000'/\u0013°@\u0000_@\u0006'\u0000\u0000-)x-\u0000\u0000.*x.\u0000\u00004)x4\u0000\u00006/x6\u0000\u00004)x4\u0000\u0000.0\r°@\u0000C@(.\u0000\u0000-)x-\u0000\u0000.0x.\u0000\u00001,x1\u0000\u00002/x2\u0000\u0000.)x.\u0000\u0000)4\u0012°@\u0000D@\")\u0000\u0000/)x/\u0000\u00000,x0\u0000\u00002/x2\u0000\u000030x3\u0000\u0000))x)\u0000\u0000.1\u0012°@\u0000D@\".\u0000\u0000-)x-\u0000\u0000./x.\u0000\u00001(x1\u0000\u00002/x2\u0000\u0000.)x.\u0000\u0000',\u000e°@\u0000d@\u0006'\u0000\u0000-)x-\u0000\u0000.,x.\u0000\u00000%x0\u0000\u00003)x3\u0000\u0000-%x-\u0000\u0000&/\r°@\u0000I@\"&\u0000\u00001)x1\u0000\u00002/\t°@\u00006@92\u0000\u0000,)x,\u0000\u0000./\u0011°@\u0000(@?.\u0000\u0000-)x-\u0000\u0000)/\n°@\u0000*@D)\u0000\u0000*)x*\u0000\u0000%/\u000b°@\u00004@9%\u0000\u0000&)x&\u0000\u0000 /\f°@\u0000-@? \u0000\u0000!)x!\u0000\u0000&1\u000b°@\u0000=@X&\u0000\u0000+\"\u00008(\f°@\u0000l@h@\u0000l@T8\u0000\u0000+\u0000\u0000+#\u00008)\r°@\u0000^@`8\u0000\u00055)\u000e°@\u0000\u000e@L5\u0000\u00003'x3\u0000\u00002(p2\u0000\u0000+\u0000\u0000+\"\u00008(\u000f°@\u0000[@y@\u0000\u001a@#8\u0000\u0000+\u0000\u0000+#\u00008)\u000b°@\u0000`@)+\u0000\u0000+B<8\u0000\u0000;>\u000f°@\u00002@';\u0000\u0000=7x=\u0000\u0000>9p>\u0000\u0000+\u0000\u0000$E\f°@\u0000e@\u0007$\u0000\u0000+=x+\u0000\u00000=x0\u0000\u00003=x3\u0000\u00007Ex7\u0000\u0000<=x<\u0000\u0000?=x?\u0000\u0000C=xC\u0000\u0000H;xH\u0000\u0000K?xK\u0000\u0000ODxO\u0000\u0000THxT\u0000\u0000WOPW\u0000\u0000<5\u0010°@\u0000\f@L<\u0000\u0000:3x:\u0000\u000094p9\u0000\u00009;\r°@\u0000~@E9\u0000\u0000?;\u000e°@\u0000)@1?\u0000\u0000>9x>\u0000\u0000<9p<\u0000\u0000<=\f°@\u0000\u001d@\"<\u0000\u000594\u0012°@\u0000m@Q9\u0000\u0000:3\u000023\u0010°@\u0000@Y2\u0000\u00001)x1\u0000\u00002/p2\u0000\u0000:\u0000\u00002/\f°@\u0000\u0014@02\u0000\u000021\u000071\n°@\u0000n@p2\u0000\u00001)x1\u0000\u000021p2\u0000\u000020\u0014°@\u0000u@G2\u0000\u00007\u0000\u000021\u0000:1\n°@\u0000]@\u00012\u0000\u00001)x1\u0000\u000021p2\u0000\u0000:\u0000\u0000:0\u000030\b°@\u0000v@R3\u0000\u0000:\u0000\u000030\u0000,0\u000e°@\u0000B@\u0000,\u0000\u00003\u0000\u000020\u000b°@\u0000m@X2\u0000\u000021\u0000+1\t°@\u0000o@p2\u0000\u00003)x3\u0000\u000021p2\u0000\u0000+\u0000\u000020\u0011°@\u0000Z@e2\u0000\u0000+\"\u00002(\u00007(\t°@\u0000\u0002@E7\u0000\u00002\u0000\u0000+\u0000\u0000</\u0000+&\r°@\u0000\u0013@0+\u0000\u0000<\u0000\u0000:)\u0000+#\t°@\u0000\t@>+\u0000\u0000:\u0000\u0000+#\u00002)\u00009)\r°@\u0000H@{9\u0000\u00002\u0000\u0000+\u0000\u0000+\"\u00002(\u00007(\u0012°@\u0000C@{7\u0000\u00002\u0000\u0000+\u0000\u0000+&\u0000</\t°@\u0000g@`<\u0000\u0000+\u0000\u0000+#\u0000:)\u0006°@\u0000e@`:\u0000\u0005+\u0000\u0000*#\u00006)\u0011°@\u0000\u0012@-6\u0000\u0000*\u0000\u000020\u0000)'\u000b°@\u00000@\u00152\u0000\u00003*\u0011°@\u0000m@R3\u0000\u0000)\u0000\u0000.-\u00002-\u000b°@\u0000`@e2\u0000\u0000.\u0000\u0000''\u0000.0\u0007°@\u0000N@{.\u0000\u0000'\u0000\u0000\"$\u0000.*\u0010°@\u0000Z@f.\u0000\u0000\"\u0000\u0000'0\u0000.0\u0007°@\u0000q@X.\u0000\u0000'\u0000\u0000\"&\u0000..\u0007°@\u0000\u0019@0.\u0000\u0000\"\u0000\u0000)&\u00000.\u000e°@\u0000:@\b0\u0000\u0000)\u0000\u0000.0\t°@\u0000b@e.\u0000\u0000'!\u00003'\u0012°@\u0000C@\u000b3\u0000\u0000'\u0000\u0000-%p-\u0000\u00002#\u0000&\u001f\u0011°@\u0000\u0007@X&\u0000\u00002\u0000\u0012°@\u0000_@_>(\u00007\"\u0000C(\n°@\u0000Y@mC\u0000\u00007\u0000\u0000>\u0000\u0000H+\u00007&\u000e°@\u0000}@E7\u0000\u0000H\u0000\u00007#\u0000F)\u000b°@\u0000`@eF\u0000\u00007\u0000\u00007#\u0000>)\u0000E)\u0010°@\u0000\u0002@>E\u0000\u0000>\u0000\u00007\u0000\u00007\"\u0000>(\u0000°@\u0000\u0000C( °@0C\u0000\u0000>\u0000\u00007\u0000\u00007&\u0000H-\r°@\u0000;@\bH\u0000\u00007\u0000\u00007#\u0000F)\f°@\u0000_@`F\u0000\u00057\u0000\u0000B)\u00006#\u0015°@\u0000\u0007@46\u0000\u0000B\u0000\u0000>0\u00005'\u0007°@\u0000A@\b>\u0000\u0000?*\u0011°@\u0000R@m?\u0000\u00005\u0000\u0000:*\u0000>0\t°@\u0000b@e>\u0000\u0000:\u0000\u00003'\u0000?0\u000f°@\u0000F@{?\u0000\u00003\u0000\u0000.$\u0000:*\n°@\u0000n@X:\u0000\u0000.\u0000\u0000:0\u000030\u0011°@\u0000D@{3\u0000\u0000:\u0000\u0000:.\u0000.&\u000e°@\u0000\u001f@#.\u0000\u0000:\u0000\u0000<.\u00005&\u000b°@\u0000K@z5\u0000\u0000<\u0000\u0000:0\f°@\u0000_@e:\u0000\u00003!\u0000?'\u0012°@\u0000f@h?\u0000\u00003\u0000\u00009%p9\u0000\u00002\u001f\u0000>&\u0011°@\u0000l@n>\u0000\u00052\u0000\u0011°@\u0000'@\u0018+\"\u00008(\u000f°@\u0000^@r@\u0000w@J8\u0000\u0000+\u0000\u00007)\u0000+#\t°@\u0000'@ 7\u0000\u00005)\u0010°@\u0000\f@L5\u0000\u00003'x3\u0000\u00002(p2\u0000\u0000+\u0000\u0000+\"\u00008(\u000f°@\u00008@\u0015@\u0000w@M8\u0000\u0000+\u0000\u0000+#\u00008)\f°@\u0000_@)+\u0000\u0000+B<8\u0000\u0000;>\u0011°@\u0000V@\u0001;\u0000\u0000=7x=\u0000\u0000>9p>\u0000\u0000+\u0000\u0000$E\u00000E\r°@\u0000+@(0\u0000\u0000$\u0000\u0000C2\u0000H<\u0000<2p<\u0000\u0000H\u0000\u0000C\u0000\u0000C5\u0000<5\u0000H@\u0010°@\u0000\u0004@TH\u0000\u0000<\u0000\u0000C\u0000\u0000<4\u0000H>\u0000C4xC\u0000\u0000H\u0000\u0000<\u0000\u0000C>\u0000<4p<\u0000\u0000C\u0000\u0000G@\u0000<5\u0000D5\t°@\u0000S@\fD\u0000\u0000<\u0000\u0000G\u0000\u0000D;\u0000<1x<\u0000\u0000D\u0000\u0000D;\u0000<14<\u0000\u0000D\u0000<D=\u0000<2\n°@\u0000\u0007@W<\u0000\u0000D\u0000\u0000D:\u0000<0x<\u0000\u0000D\u0000\u0000D:\u0000<0p<\u0000\u0000D\u0000\u0000<2\u0000°@\u0000\u0000C=\u0015°@SC\u0000\u0000<\u0000\u0000C9\u0000</x<\u0000\u0000C\u0000\u0000C9\u0000</p<\u0000\u0000C\u0000\u0000<5\u0000C5\u0000H@\u0013°@\u0000\u0013@BH\u0000\u0000C\u0000\u0000<\u0000\u0000<4\u0000C4\u0000H>xH\u0000\u0000C\u0000\u0000<\u0000\u0000C4\u0000<4p<\u0000\u0000C\u0000\u0000D5\u0000<5\u0000G@\n°@\u0000R@\fG\u0000\u0000<\u0000\u0000D\u0000\u0000<1\u0000D;xD\u0000\u0000<\u0000\u0000<1\u0000D;4D\u0000\u0000<\u0000<<2\u0000D=\u000b°@\u0000m@pD\u0000\u0000<\u0000\u0000<0\u0000D:xD\u0000\u0000<\u0000\u0000<0\u0000D:pD\u0000\u0000<\u0000\u0000<2\u0000C=\u000b°@\u0000v@OC\u0000\u0000<\u0000\u0000:2\u0000C2\u0019°@\u0000c@TC\u0000\u0000:\u0000\u000092\u0000?2\n°@\u0000\n@<?\u0000\u0000>2\u000b°@\u0000U@p>\u0000\u00009\u0000\u000072\u0000>2\u000b°@\u0000&@\u001f>\u0000\u0000@2\r°@\u0000@@\u0003@\u0000\u00007\u0000\u0000A2\u000092\u000b°@\u0000&@\u001fA\u0000\u0000@2\u0011°@\u0000\u0003@L@\u0000\u0000C-pC\u0000\u00009\u0000\u0000°@\u0000\u0000>5\u0000B-P°@\u0018>\u0000\u0000=2\n°@\u0000n=\u0000\u0000>2p>\u0000\u0000B\u0000\u0000>5x°@X>\u0000\u0000>1\u0000C1\b°@\u0000H@\u0018>\u0000\u0000=*x=\u0000\u0000>1p>\u0000\u0000>2\u0011°@\u0000~@A>\u0000\u0000C\u0000\u0000B1\u0000>1\t°@\u0000o@p>\u0000\u0000=*x=\u0000\u0000>1p>\u0000\u0000>2\u0011°@\u0000\u0015@*>\u0000\u0000B\u0000\u0000>1\f°@\u0000\u0016@F>\u0000\u0000>*x>\u0000\u0000>1p>\u0000\u0000>2\u000f°@\u0000~@C>\u0000\u0000>1\u000b°@\u0000\u001e@?>\u0000\u0000?'x?\u0000\u0000>,p>\u0000\u0000>,\u000f°@\u00007@*>\u0000`2,\u00007,\b°@\u0000\r@S2\u0000\u00001&x1\u0000\u00002,p2\u0000\u00002-\u0013°@\u0000C@z2\u0000\u00007\u0000\u00002,\u00006,\u0007°@\u0000}@d2\u0000\u00001&x1\u0000\u00002,p2\u0000\u00002-\u000f°@\u0000R@o2\u0000\u00006\u0000\u00002,\t°@\u0000S@\f2\u0000\u00002&x2\u0000\u00002,p2\u0000\u00002-\u0013°@\u0000z@C2\u0000\u00006&\u00002&\u0000°@\u00003@E2\u0000\u00001&x1\u0000\u00002&x2\u0000\u00001%x1\u0000\u00002&x2\u0000\u00001!x1\u0000\u00006\u0000\u00003&\u00006&\b°@\u0000N@\"3\u0000\u00002!x2\u0000\u00001&x1\u0000\u00002\"x2\u0000\u00003&x3\u0000\u00002!x2\u0000\u00006\u0000\u0000+%\u00007%\b°@\u00007@97\u0000\u00006(x6\u0000\u00005)x5\u0000\u00004*x4\u0000\u00003,x3\u0000\u00002/x2\u0000\u0000+\u0000\u00001/\u0011°@\u0000E@\"1\u0000\u00002,x2\u0000\u00003*x3\u0000\u00004)x4\u0000\u00005(x5\u0000\u00006(x6\u0000\u0000+)\u00007)\t°@\u00006@97\u0000\u00006$x6\u0000\u00007)x7\u0000\u00002%x2\u0000\u00001)x1\u0000\u00002$x2\u0000\u0000+\u0000\u00006)\u000e°@\u0000B@(6\u0000\u00002(x2\u0000\u00001)x1\u0000\u00002$x2\u0000\u00003*x3\u0000\u00002)x2\u0000\u0000+%\u00007%\b°@\u0000,@D7\u0000\u00006(x6\u0000\u00005*x5\u0000\u00004,x4\u0000\u00003,x3\u0000\u00002*x2\u0000\u0000+\u0000\u00001*\u0010°@\u0000F@\"1\u0000\u00002)x2\u0000\u00003(x3\u0000\u00004%x4\u0000\u00005)x5\u0000\u00006)x6\u0000\u0000+)\u00007)\f°@\u0000-@?7\u0000\u00006$x6\u0000\u00007)x7\u0000\u00002%x2\u0000\u00001/x1\u0000\u00002*x2\u0000\u0000+\u0000\u0000*/\u0011°@\u0000P@\u00172)x2\u0000\u00001%x1\u0000\u00002)x2\u0000\u00003*x3\u0000\u00002)x2\u0000\u0000*\u0000\u000021\u0000)1\b°@\u00001@?2\u0000\u00001,x1\u0000\u000021x2\u0000\u00001*x1\u0000\u000020x2\u0000\u00001*x1\u0000\u0000)\u0000\u00003/\u0000)/\t°@\u0000M@\"3\u0000\u00002)x2\u0000\u000030x3\u0000\u00000)x0\u0000\u00005/x5\u0000\u00003)x3\u0000\u0000)\u0000\u0000./\u00002/\u0007°@\u00008@92\u0000\u00001'x1\u0000\u00002/x2\u0000\u00001)x1\u0000\u00002/x2\u0000\u00001)x1\u0000\u0000.\u0000\u0000'/\u000f°@\u0000<@-'\u0000\u0000-)x-\u0000\u0000.0x.\u0000\u00003,x3\u0000\u000074x7\u0000\u00003)x3\u0000\u0000.1\u0007°@\u0000D@-.\u0000\u00004,x4\u0000\u000050x5\u0000\u00001)x1\u0000\u000021x2\u0000\u0000.)x.\u0000\u0000'/\u0013°@\u0000_@\u0006'\u0000\u0000-)x-\u0000\u0000.*x.\u0000\u00004)x4\u0000\u00006/x6\u0000\u00004)x4\u0000\u0000.0\r°@\u0000C@(.\u0000\u0000-)x-\u0000\u0000.0x.\u0000\u00001,x1\u0000\u00002/x2\u0000\u0000.)x.\u0000\u0000)4\u0012°@\u0000D@\")\u0000\u0000/)x/\u0000\u00000,x0\u0000\u00002/x2\u0000\u000030x3\u0000\u0000))x)\u0000\u0000.1\u0012°@\u0000D@\".\u0000\u0000-)x-\u0000\u0000./x.\u0000\u00001(x1\u0000\u00002/x2\u0000\u0000.)x.\u0000\u0000',\u000e°@\u0000d@\u0006'\u0000\u0000-)x-\u0000\u0000.,x.\u0000\u00000%x0\u0000\u00003)x3\u0000\u0000-%x-\u0000\u0000&/\r°@\u0000I@\"&\u0000\u00001)x1\u0000\u00002/\t°@\u00006@92\u0000\u0000,)x,\u0000\u0000./\u0011°@\u0000(@?.\u0000\u0000-)x-\u0000\u0000)/\n°@\u0000*@D)\u0000\u0000*)x*\u0000\u0000%/\u000b°@\u00004@9%\u0000\u0000&)x&\u0000\u0000 /\f°@\u0000-@? \u0000\u0000!)x!\u0000\u0000&1\u000b°@\u0000=@X&\u0000\u0000+\"\u00008(\f°@\u0000l@h@\u0000l@T8\u0000\u0000+\u0000\u0000+#\u00008)\r°@\u0000^@`8\u0000\u00055)\u000e°@\u0000\u000e@L5\u0000\u00003'x3\u0000\u00002(p2\u0000\u0000+\u0000\u0000+\"\u00008(\u000f°@\u0000[@y@\u0000\u001a@#8\u0000\u0000+\u0000\u00008)\u0000+#\u000b°@\u0000`@)+\u0000\u0000+B<8\u0000\u0000;>\u000f°@\u00002@';\u0000\u0000=7x=\u0000\u0000>9p>\u0000\u0000+\u0000\u0000$E\f°@\u0000e@\u0007$\u0000\u0000+=x+\u0000\u00000=x0\u0000\u00003=x3\u0000\u00007Ex7\u0000\u0000<=x<\u0000\u0000?=x?\u0000\u0000C=xC\u0000\u0000H;xH\u0000\u0000K?xK\u0000\u0000ODxO\u0000\u0000THxT\u0000\u0000WOPW\u0000\u0000<5\u0010°@\u0000\f@L<\u0000\u0000:3x:\u0000\u000094p9\u0000\u00009;\r°@\u0000~@E9\u0000\u0000?;\u000e°@\u0000)@1?\u0000\u0000>9x>\u0000\u0000<9p<\u0000\u0000<=\f°@\u0000\u001d@\"<\u0000\u000594\u0012°@\u0000m@Q9\u0000\u0000:3\u000023\u0010°@\u0000@Y2\u0000\u00001)x1\u0000\u00002/p2\u0000\u0000:\u0000\u00002/\f°@\u0000x@L2\u0000\u000071\u000021\n°@\u0000n@p2\u0000\u00001)x1\u0000\u000021p2\u0000\u000020\u0014°@\u0000u@G2\u0000\u00007\u0000\u000021\u0000:1\n°@\u0000]@\u00012\u0000\u00001)x1\u0000\u000021p2\u0000\u0000:\u0000\u000030\u0000:0\b°@\u0000v@R:\u0000\u00003\u0000\u000030\u0000,0\u000e°@\u0000B@\u0000,\u0000\u00003\u0000\u000020\u000b°@\u0000m@X2\u0000\u0000+1\u000021\t°@\u0000o@p2\u0000\u00003)x3\u0000\u000021p2\u0000\u0000+\u0000\u000020\u0011°@\u0000Z@e2\u0000\u0000+3\u000023\u000b°@\u0000\t@T2\u0000\u00003*x3\u0000\u000023p2\u0000\u0000+\u0000\u000022\u0011°@\u0000u@J2\u0000\u000023\u0000+3\f°@\u0000_@}2\u0000\u00003*x3\u0000\u000023p2\u0000\u0000+\u0000\u000022\u0011°@\u0000m@R2\u0000\u000023\u0000+3\t°@\u0000z@e2\u0000\u00003(x3\u0000\u000021p2\u0000\u0000+\u0000\u00002/\u0000</\u0012°@\u0000K@s<\u0000\u00002\u0000\u00002)\u0000+)*°@\u0000F+\u0000\u00002\u0000\u0000*14°@D*\u0000\u0000+1x+\u0000\u0000-2x-\u0000\u0000.3x.\u0000\u000014x1\u0000\u000025x2\u0000\u0000-2\u0011°@\u00004@3-\u0000\u0000.3x.\u0000\u000013x1\u0000\u000024x2\u0000\u000065x6\u0000\u000078x7\u0000\u000013\u000b°@\u0000.@?1\u0000\u000024x2\u0000\u000068x6\u0000\u000079x7\u0000\u00009;x9\u0000\u0000:<x:\u0000\u000068\u000b°@\u0000@@-6\u0000\u00007:x7\u0000\u00009;x9\u0000\u0000:<x:\u0000\u0000==x=\u0000\u0000>>x>\u0000\u000099\u0012°@\u00003@39\u0000\u0000::x:\u0000\u0000=<x=\u0000\u0000>=x>\u0000\u0000B@xB\u0000\u0000CBxC\u0000\u0000=D\u0015°@\u0000;@(=\u0000\u0000>=x>\u0000\u0000B?xB\u0000\u0000CAxC\u0000\u0000ECxE\u0000\u0000FDxF\u0000\u0000BF\u0015°@\u0000A@\"B\u0000\u0000C>xC\u0000\u0000EBxE\u0000\u0000FExF\u0000\u0000IHxI\u0000\u0000JLxJ\u0000\u0000EJ\u000e°@\u00007@3E\u0000\u0000FMxF\u0000\u0000IQxI\u0000\u0000JSxJ\u0000\u0000O]\r°@\u0000~@EO\u0000\u00002T\u0000&F\u0010°@\u00000@\u0010&\u0000\u00002\u0000\u0000+T\u00007T\t°@\u0000M@z7\u0000\u0000+\u0000\u00002T\u0000&F\r°@\u0000\u000b@8&\u0000\u00002\u0000\u0000\u001fM\u0000+\\\u000f°@\u0000\t@\b+\u0000\u0000\u001f\u0000\u0000F]\u00007O\u0000>O >\u0000\u00007\u0000\u0000F\u0000X°@\u0000\u0000ÿ/\u0000MTrk\u0000\u0000\u0000)\u0000ÿ\u0003!Albeniz :Suite espagnole Cataluna\u0000ÿ/\u0000MTrk\u0000\u0000\u0000(\u0000ÿ\u0003 Copyright © 2001 by Bernd Krüger\u0000ÿ/\u0000MTrk\u0000\u0000\u0000 \u0000ÿ\u0003\u0018http://www.piano-midi.de\u0000ÿ/\u0000MTrk\u0000\u0000\u0000\u001b\u0000ÿ\u0003\u0013Edition: 2010-09-04\u0000ÿ/\u0000MTrk\u0000\u0000\u0000\u000e\u0000ÿ\u0003\u0006Spur 7\u0000ÿ/\u0000MTrk\u0000\u0000\u0000\u000e\u0000ÿ\u0003\u0006Spur 8\u0000ÿ/\u0000";
  var jsonSong = midiConverter.midiToJson(midiSong);
  fs.writeFileSync('example.json', JSON.stringify(jsonSong));
}

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
},{"midi-converter":3}],2:[function(require,module,exports){
module.exports=[
  {"hexcode":"0x00", "family":"Piano", "instrument":"Acoustic Grand Piano"},
  {"hexcode":"0x01", "family":"Piano", "instrument":"Bright Acoustic Piano"},
  {"hexcode":"0x02", "family":"Piano", "instrument":"Electric Grand Piano"},
  {"hexcode":"0x03", "family":"Piano", "instrument":"Honky-tonk Piano"},
  {"hexcode":"0x04", "family":"Piano", "instrument":"Electric Piano 1"},
  {"hexcode":"0x05", "family":"Piano", "instrument":"Electric Piano 2"},
  {"hexcode":"0x06", "family":"Piano", "instrument":"Harpsichord"},
  {"hexcode":"0x07", "family":"Piano", "instrument":"Clavichord"},
  {"hexcode":"0x08", "family":"Chromatic Percussion", "instrument":"Celesta"},
  {"hexcode":"0x09", "family":"Chromatic Percussion", "instrument":"Glockenspiel"},
  {"hexcode":"0x0A", "family":"Chromatic Percussion", "instrument":"Music Box"},
  {"hexcode":"0x0B", "family":"Chromatic Percussion", "instrument":"Vibraphone"},
  {"hexcode":"0x0C", "family":"Chromatic Percussion", "instrument":"Marimba"},
  {"hexcode":"0x0D", "family":"Chromatic Percussion", "instrument":"Xylophone"},
  {"hexcode":"0x0E", "family":"Chromatic Percussion", "instrument":"Tubular bells"},
  {"hexcode":"0x0F", "family":"Chromatic Percussion", "instrument":"Dulcimer"},
  {"hexcode":"0x10", "family":"Organ", "instrument":"Drawbar Organ"},
  {"hexcode":"0x11", "family":"Organ", "instrument":"Percussive Organ"},
  {"hexcode":"0x12", "family":"Organ", "instrument":"Rock Organ"},
  {"hexcode":"0x13", "family":"Organ", "instrument":"Church Organ"},
  {"hexcode":"0x14", "family":"Organ", "instrument":"Reed Organ"},
  {"hexcode":"0x15", "family":"Organ", "instrument":"Accordion"},
  {"hexcode":"0x16", "family":"Organ", "instrument":"Harmonica"},
  {"hexcode":"0x17", "family":"Organ", "instrument":"Tango Accordion"},
  {"hexcode":"0x18", "family":"Guitar", "instrument":"Acoustic Guitar (nylon)"},
  {"hexcode":"0x19", "family":"Guitar", "instrument":"Acoustic Guitar (steel)"},
  {"hexcode":"0x1A", "family":"Guitar", "instrument":"Electric Guitar (jazz)"},
  {"hexcode":"0x1B", "family":"Guitar", "instrument":"Electric Guitar (clean)"},
  {"hexcode":"0x1C", "family":"Guitar", "instrument":"Electric Guitar (muted)"},
  {"hexcode":"0x1D", "family":"Guitar", "instrument":"Overdriven Guitar"},
  {"hexcode":"0x1E", "family":"Guitar", "instrument":"Distortion Guitar"},
  {"hexcode":"0x1F", "family":"Guitar", "instrument":"Guitar harmonics"},
  {"hexcode":"0x20", "family":"Bass", "instrument":"Acoustic Bass"},
  {"hexcode":"0x21", "family":"Bass", "instrument":"Electric Bass (finger)"},
  {"hexcode":"0x22", "family":"Bass", "instrument":"Electric Bass (pick)"},
  {"hexcode":"0x23", "family":"Bass", "instrument":"Fretless Bass"},
  {"hexcode":"0x24", "family":"Bass", "instrument":"Slap Bass 1"},
  {"hexcode":"0x25", "family":"Bass", "instrument":"Slap bass 2"},
  {"hexcode":"0x26", "family":"Bass", "instrument":"Synth Bass 1"},
  {"hexcode":"0x27", "family":"Bass", "instrument":"Synth Bass 2"},
  {"hexcode":"0x28", "family":"Strings", "instrument":"Violin"},
  {"hexcode":"0x29", "family":"Strings", "instrument":"Viola"},
  {"hexcode":"0x2A", "family":"Strings", "instrument":"Cello"},
  {"hexcode":"0x2B", "family":"Strings", "instrument":"Contrabass"},
  {"hexcode":"0x2C", "family":"Strings", "instrument":"Tremolo Strings"},
  {"hexcode":"0x2D", "family":"Strings", "instrument":"Pizzicato Strings"},
  {"hexcode":"0x2E", "family":"Strings", "instrument":"Orchestral Harp"},
  {"hexcode":"0x2F", "family":"Strings", "instrument":"Timpani"},
  {"hexcode":"0x30", "family":"Ensemble", "instrument":"String Ensemble 1"},
  {"hexcode":"0x31", "family":"Ensemble", "instrument":"String Ensemble 2"},
  {"hexcode":"0x32", "family":"Ensemble", "instrument":"SynthStrings 1"},
  {"hexcode":"0x33", "family":"Ensemble", "instrument":"SynthStrings 2"},
  {"hexcode":"0x34", "family":"Ensemble", "instrument":"Choir Aahs"},
  {"hexcode":"0x35", "family":"Ensemble", "instrument":"Voice Oohs"},
  {"hexcode":"0x36", "family":"Ensemble", "instrument":"Synth Voice"},
  {"hexcode":"0x37", "family":"Ensemble", "instrument":"Orchestra Hit"},
  {"hexcode":"0x38", "family":"Brass", "instrument":"Trumpet"},
  {"hexcode":"0x39", "family":"Brass", "instrument":"Trombone"},
  {"hexcode":"0x3A", "family":"Brass", "instrument":"Tuba"},
  {"hexcode":"0x3B", "family":"Brass", "instrument":"Muted Trombone"},
  {"hexcode":"0x3C", "family":"Brass", "instrument":"French Horn"},
  {"hexcode":"0x3D", "family":"Brass", "instrument":"Brass Section"},
  {"hexcode":"0x3E", "family":"Brass", "instrument":"SynthBrass 1"},
  {"hexcode":"0x3F", "family":"Brass", "instrument":"SynthBrass 2"},
  {"hexcode":"0x40", "family":"Reed", "instrument":"Soprano Sax"},
  {"hexcode":"0x41", "family":"Reed", "instrument":"Alto Sax"},
  {"hexcode":"0x42", "family":"Reed", "instrument":"Tenor Sax"},
  {"hexcode":"0x43", "family":"Reed", "instrument":"Baritone Sax"},
  {"hexcode":"0x44", "family":"Reed", "instrument":"Oboe"},
  {"hexcode":"0x45", "family":"Reed", "instrument":"English Horn"},
  {"hexcode":"0x46", "family":"Reed", "instrument":"Bassoon"},
  {"hexcode":"0x47", "family":"Reed", "instrument":"Clarinet"},
  {"hexcode":"0x48", "family":"Pipe", "instrument":"Piccolo"},
  {"hexcode":"0x49", "family":"Pipe", "instrument":"Flute"},
  {"hexcode":"0x4A", "family":"Pipe", "instrument":"Recorder"},
  {"hexcode":"0x4B", "family":"Pipe", "instrument":"Pan Flute"},
  {"hexcode":"0x4C", "family":"Pipe", "instrument":"Blown Bottle"},
  {"hexcode":"0x4D", "family":"Pipe", "instrument":"Shakuhachi"},
  {"hexcode":"0x4E", "family":"Pipe", "instrument":"Whistle"},
  {"hexcode":"0x4F", "family":"Pipe", "instrument":"Ocarina"},
  {"hexcode":"0x50", "family":"Synth Lead", "instrument":"Lead 1 (square)"},
  {"hexcode":"0x51", "family":"Synth Lead", "instrument":"Lead 2 (sawtooth)"},
  {"hexcode":"0x52", "family":"Synth Lead", "instrument":"Lead 3 (calliope)"},
  {"hexcode":"0x53", "family":"Synth Lead", "instrument":"Lead 4 (chiff)"},
  {"hexcode":"0x54", "family":"Synth Lead", "instrument":"Lead 5 (charang)"},
  {"hexcode":"0x55", "family":"Synth Lead", "instrument":"Lead 6 (voice)"},
  {"hexcode":"0x56", "family":"Synth Lead", "instrument":"Lead 7 (fifths)"},
  {"hexcode":"0x57", "family":"Synth Lead", "instrument":"Lead 8 (bass + lead)"},
  {"hexcode":"0x58", "family":"Synth Pad", "instrument":"Pad 1 (new age)"},
  {"hexcode":"0x59", "family":"Synth Pad", "instrument":"Pad 2 (warm)"},
  {"hexcode":"0x5A", "family":"Synth Pad", "instrument":"Pad 3 (polysynth)"},
  {"hexcode":"0x5B", "family":"Synth Pad", "instrument":"Pad 4 (choir)"},
  {"hexcode":"0x5C", "family":"Synth Pad", "instrument":"Pad 5 (bowed)"},
  {"hexcode":"0x5D", "family":"Synth Pad", "instrument":"Pad 6 (metallic)"},
  {"hexcode":"0x5E", "family":"Synth Pad", "instrument":"Pad 7 (halo)"},
  {"hexcode":"0x5F", "family":"Synth Pad", "instrument":"Pad 8 (sweep)"},
  {"hexcode":"0x60", "family":"Synth Effects", "instrument":"FX 1 (rain)"},
  {"hexcode":"0x61", "family":"Synth Effects", "instrument":"FX 2 (soundtrack)"},
  {"hexcode":"0x62", "family":"Synth Effects", "instrument":"FX 3 (crystal)"},
  {"hexcode":"0x63", "family":"Synth Effects", "instrument":"FX 4 (atmosphere)"},
  {"hexcode":"0x64", "family":"Synth Effects", "instrument":"FX 5 (brightness)"},
  {"hexcode":"0x65", "family":"Synth Effects", "instrument":"FX 6 (goblins)"},
  {"hexcode":"0x66", "family":"Synth Effects", "instrument":"FX 7 (echoes)"},
  {"hexcode":"0x67", "family":"Synth Effects", "instrument":"FX 8 (sci-fi)"},
  {"hexcode":"0x68", "family":"Ethnic", "instrument":"Sitar"},
  {"hexcode":"0x69", "family":"Ethnic", "instrument":"Banjo"},
  {"hexcode":"0x6A", "family":"Ethnic", "instrument":"Shamisen"},
  {"hexcode":"0x6B", "family":"Ethnic", "instrument":"Koto"},
  {"hexcode":"0x6C", "family":"Ethnic", "instrument":"Kalimba"},
  {"hexcode":"0x6D", "family":"Ethnic", "instrument":"Bag pipe"},
  {"hexcode":"0x6E", "family":"Ethnic", "instrument":"Fiddle"},
  {"hexcode":"0x6F", "family":"Ethnic", "instrument":"Shanai"},
  {"hexcode":"0x70", "family":"Percussive", "instrument":"Tinkle Bell"},
  {"hexcode":"0x71", "family":"Percussive", "instrument":"Agogo"},
  {"hexcode":"0x72", "family":"Percussive", "instrument":"Steel Drums"},
  {"hexcode":"0x73", "family":"Percussive", "instrument":"Woodblock"},
  {"hexcode":"0x74", "family":"Percussive", "instrument":"Taiko Drum"},
  {"hexcode":"0x75", "family":"Percussive", "instrument":"Melodic Tom"},
  {"hexcode":"0x76", "family":"Percussive", "instrument":"Synth Drum"},
  {"hexcode":"0x77", "family":"Percussive", "instrument":"Reverse Cymbal"},
  {"hexcode":"0x78", "family":"Sound Effects", "instrument":"Guitar Fret Noise"},
  {"hexcode":"0x79", "family":"Sound Effects", "instrument":"Breath Noise"},
  {"hexcode":"0x7A", "family":"Sound Effects", "instrument":"Seashore"},
  {"hexcode":"0x7B", "family":"Sound Effects", "instrument":"Bird Tweet"},
  {"hexcode":"0x7C", "family":"Sound Effects", "instrument":"Telephone Ring"},
  {"hexcode":"0x7D", "family":"Sound Effects", "instrument":"Helicopter"},
  {"hexcode":"0x7E", "family":"Sound Effects", "instrument":"Applause"},
  {"hexcode":"0x7F", "family":"Sound Effects", "instrument":"Gunshot"}
]

},{}],3:[function(require,module,exports){
var fs = require('fs')
  , midiParser = require('midi-file-parser')
  , path = require('path')
  , Midi = require('jsmidgen')
  , instruments = require('./instruments.json');

module.exports = {
  midiToJson: function(midi) {
    return midiParser(midi);
  },
  jsonToMidi: function(songJson) {
    var file = new Midi.File();

    songJson.tracks.forEach(function(t) {
      var track = new Midi.Track();
      file.addTrack(track);

      t.forEach(function(note) {
        if (note.subtype === 'programChange') {
          var instrument = instruments[note.programNumber].hexcode;
          track.setInstrument(note.channel, instrument);
        } else if (note.subtype === 'setTempo') {
          var microsecondsPerBeat = note.microsecondsPerBeat;
          var microsecondsPerMin = 60000000;
          var ticksPerBeat = songJson.header.ticksPerBeat;
          var bpm = (ticksPerBeat/128)*microsecondsPerMin/microsecondsPerBeat;
          track.setTempo(bpm, note.deltaTime);
        } else if (note.subtype === 'noteOn') {
          var noteStr = noteFromMidiPitch(note.noteNumber);
          track.addNoteOn(note.channel, noteStr, note.deltaTime, note.velocity);
        } else if (note.subtype === 'noteOff') {
          var noteStr = noteFromMidiPitch(note.noteNumber);
          track.addNoteOff(note.channel, noteStr, note.deltaTime);
        } else if ( note != 'undefined' && note != null
              && note.hasOwnProperty("deltaTime") &&   typeof note.channel !== 'undefined'
              && ( note.channel >= 0 &&  note.channel < 16 ) ) {
          // Work around: dummy pitchbend (with no bending) instead various controller messages
          // until proper encoding calls are implemented (probably in jsmidgen)
          // This is needed in order to maintain correct deltaTime
          var pitchBend = new Midi.Event(
            {
              time:   note.deltaTime,
              type:   Midi.Event.PITCH_BEND,
              param1: 0x00, // LSB of centered
              param2: 0x40  // MSB of centered
             });
          track.addEvent(pitchBend);
        }
      });
    });

    return file.toBytes();

    function noteFromMidiPitch(p) {
      var noteDict = ['c', 'c#', 'd', 'd#', 'e', 'f', 'f#', 'g', 'g#', 'a', 'a#', 'b'];
      var octave = Math.floor((p-12)/12);
      var note = noteDict[p-octave*12-12];
      return note+octave;
    }
  }
};

},{"./instruments.json":2,"fs":6,"jsmidgen":4,"midi-file-parser":5,"path":7}],4:[function(require,module,exports){
var Midi = {};

(function(exported) {

	var DEFAULT_VOLUME   = exported.DEFAULT_VOLUME   = 90;
	var DEFAULT_DURATION = exported.DEFAULT_DURATION = 128;
	var DEFAULT_CHANNEL  = exported.DEFAULT_CHANNEL  = 0;

	/* ******************************************************************
	 * Utility functions
	 ****************************************************************** */

	var Util = {

		midi_letter_pitches: { a:21, b:23, c:12, d:14, e:16, f:17, g:19 },

		/**
		 * Convert a symbolic note name (e.g. "c4") to a numeric MIDI pitch (e.g.
		 * 60, middle C).
		 *
		 * @param {string} n - The symbolic note name to parse.
		 * @returns {number} The MIDI pitch that corresponds to the symbolic note
		 * name.
		 */
		midiPitchFromNote: function(n) {
			var matches = /([a-g])(#+|b+)?([0-9]+)$/i.exec(n);
			var note = matches[1].toLowerCase(), accidental = matches[2] || '', octave = parseInt(matches[3], 10);
			return (12 * octave) + Util.midi_letter_pitches[note] + (accidental.substr(0,1)=='#'?1:-1) * accidental.length;
		},

		/**
		 * Ensure that the given argument is converted to a MIDI pitch. Note that
		 * it may already be one (including a purely numeric string).
		 *
		 * @param {string|number} p - The pitch to convert.
		 * @returns {number} The resulting numeric MIDI pitch.
		 */
		ensureMidiPitch: function(p) {
			if (typeof p == 'number' || !/[^0-9]/.test(p)) {
				// numeric pitch
				return parseInt(p, 10);
			} else {
				// assume it's a note name
				return Util.midiPitchFromNote(p);
			}
		},

		midi_pitches_letter: { '12':'c', '13':'c#', '14':'d', '15':'d#', '16':'e', '17':'f', '18':'f#', '19':'g', '20':'g#', '21':'a', '22':'a#', '23':'b' },
		midi_flattened_notes: { 'a#':'bb', 'c#':'db', 'd#':'eb', 'f#':'gb', 'g#':'ab' },

		/**
		 * Convert a numeric MIDI pitch value (e.g. 60) to a symbolic note name
		 * (e.g. "c4").
		 *
		 * @param {number} n - The numeric MIDI pitch value to convert.
		 * @param {boolean} [returnFlattened=false] - Whether to prefer flattened
		 * notes to sharpened ones. Optional, default false.
		 * @returns {string} The resulting symbolic note name.
		 */
		noteFromMidiPitch: function(n, returnFlattened) {
			var octave = 0, noteNum = n, noteName, returnFlattened = returnFlattened || false;
			if (n > 23) {
				// noteNum is on octave 1 or more
				octave = Math.floor(n/12) - 1;
				// subtract number of octaves from noteNum
				noteNum = n - octave * 12;
			}

			// get note name (c#, d, f# etc)
			noteName = Util.midi_pitches_letter[noteNum];
			// Use flattened notes if requested (e.g. f# should be output as gb)
			if (returnFlattened && noteName.indexOf('#') > 0) {
				noteName = Util.midi_flattened_notes[noteName];
			}
			return noteName + octave;
		},

		/**
		 * Convert beats per minute (BPM) to microseconds per quarter note (MPQN).
		 *
		 * @param {number} bpm - A number in beats per minute.
		 * @returns {number} The number of microseconds per quarter note.
		 */
		mpqnFromBpm: function(bpm) {
			var mpqn = Math.floor(60000000 / bpm);
			var ret=[];
			do {
				ret.unshift(mpqn & 0xFF);
				mpqn >>= 8;
			} while (mpqn);
			while (ret.length < 3) {
				ret.push(0);
			}
			return ret;
		},

		/**
		 * Convert microseconds per quarter note (MPQN) to beats per minute (BPM).
		 *
		 * @param {number} mpqn - The number of microseconds per quarter note.
		 * @returns {number} A number in beats per minute.
		 */
		bpmFromMpqn: function(mpqn) {
			var m = mpqn;
			if (typeof mpqn[0] != 'undefined') {
				m = 0;
				for (var i=0, l=mpqn.length-1; l >= 0; ++i, --l) {
					m |= mpqn[i] << l;
				}
			}
			return Math.floor(60000000 / mpqn);
		},

		/**
		 * Converts an array of bytes to a string of hexadecimal characters. Prepares
		 * it to be converted into a base64 string.
		 *
		 * @param {Array} byteArray - Array of bytes to be converted.
		 * @returns {string} Hexadecimal string, e.g. "097B8A".
		 */
		codes2Str: function(byteArray) {
			return String.fromCharCode.apply(null, byteArray);
		},

		/**
		 * Converts a string of hexadecimal values to an array of bytes. It can also
		 * add remaining "0" nibbles in order to have enough bytes in the array as the
		 * `finalBytes` parameter.
		 *
		 * @param {string} str - string of hexadecimal values e.g. "097B8A"
		 * @param {number} [finalBytes] - Optional. The desired number of bytes
		 * (not nibbles) that the returned array should contain.
		 * @returns {Array} An array of nibbles.
		 */
		str2Bytes: function (str, finalBytes) {
			if (finalBytes) {
				while ((str.length / 2) < finalBytes) { str = "0" + str; }
			}

			var bytes = [];
			for (var i=str.length-1; i>=0; i = i-2) {
				var chars = i === 0 ? str[i] : str[i-1] + str[i];
				bytes.unshift(parseInt(chars, 16));
			}

			return bytes;
		},

		/**
		 * Translates number of ticks to MIDI timestamp format, returning an array
		 * of bytes with the time values. MIDI has a very particular way to express
		 * time; take a good look at the spec before ever touching this function.
		 *
		 * @param {number} ticks - Number of ticks to be translated.
		 * @returns {number} Array of bytes that form the MIDI time value.
		 */
		translateTickTime: function(ticks) {
			var buffer = ticks & 0x7F;

			while (ticks = ticks >> 7) {
				buffer <<= 8;
				buffer |= ((ticks & 0x7F) | 0x80);
			}

			var bList = [];
			while (true) {
				bList.push(buffer & 0xff);

				if (buffer & 0x80) { buffer >>= 8; }
				else { break; }
			}
			return bList;
		},

	};

	/* ******************************************************************
	 * Event class
	 ****************************************************************** */

	/**
	 * Construct a MIDI event.
	 *
	 * Parameters include:
	 *  - time [optional number] - Ticks since previous event.
	 *  - type [required number] - Type of event.
	 *  - channel [required number] - Channel for the event.
	 *  - param1 [required number] - First event parameter.
	 *  - param2 [optional number] - Second event parameter.
	 */
	var MidiEvent = function(params) {
		if (!this) return new MidiEvent(params);
		if (params &&
				(params.type    !== null || params.type    !== undefined) &&
				(params.channel !== null || params.channel !== undefined) &&
				(params.param1  !== null || params.param1  !== undefined)) {
			this.setTime(params.time);
			this.setType(params.type);
			this.setChannel(params.channel);
			this.setParam1(params.param1);
			this.setParam2(params.param2);
		}
	};

	// event codes
	MidiEvent.NOTE_OFF           = 0x80;
	MidiEvent.NOTE_ON            = 0x90;
	MidiEvent.AFTER_TOUCH        = 0xA0;
	MidiEvent.CONTROLLER         = 0xB0;
	MidiEvent.PROGRAM_CHANGE     = 0xC0;
	MidiEvent.CHANNEL_AFTERTOUCH = 0xD0;
	MidiEvent.PITCH_BEND         = 0xE0;


	/**
	 * Set the time for the event in ticks since the previous event.
	 *
	 * @param {number} ticks - The number of ticks since the previous event. May
	 * be zero.
	 */
	MidiEvent.prototype.setTime = function(ticks) {
		this.time = Util.translateTickTime(ticks || 0);
	};

	/**
	 * Set the type of the event. Must be one of the event codes on MidiEvent.
	 *
	 * @param {number} type - Event type.
	 */
	MidiEvent.prototype.setType = function(type) {
		if (type < MidiEvent.NOTE_OFF || type > MidiEvent.PITCH_BEND) {
			throw new Error("Trying to set an unknown event: " + type);
		}

		this.type = type;
	};

	/**
	 * Set the channel for the event. Must be between 0 and 15, inclusive.
	 *
	 * @param {number} channel - The event channel.
	 */
	MidiEvent.prototype.setChannel = function(channel) {
		if (channel < 0 || channel > 15) {
			throw new Error("Channel is out of bounds.");
		}

		this.channel = channel;
	};

	/**
	 * Set the first parameter for the event. Must be between 0 and 255,
	 * inclusive.
	 *
	 * @param {number} p - The first event parameter value.
	 */
	MidiEvent.prototype.setParam1 = function(p) {
		this.param1 = p;
	};

	/**
	 * Set the second parameter for the event. Must be between 0 and 255,
	 * inclusive.
	 *
	 * @param {number} p - The second event parameter value.
	 */
	MidiEvent.prototype.setParam2 = function(p) {
		this.param2 = p;
	};

	/**
	 * Serialize the event to an array of bytes.
	 *
	 * @returns {Array} The array of serialized bytes.
	 */
	MidiEvent.prototype.toBytes = function() {
		var byteArray = [];

		var typeChannelByte = this.type | (this.channel & 0xF);

		byteArray.push.apply(byteArray, this.time);
		byteArray.push(typeChannelByte);
		byteArray.push(this.param1);

		// Some events don't have a second parameter
		if (this.param2 !== undefined && this.param2 !== null) {
			byteArray.push(this.param2);
		}
		return byteArray;
	};

	/* ******************************************************************
	 * MetaEvent class
	 ****************************************************************** */

	/**
	 * Construct a meta event.
	 *
	 * Parameters include:
	 *  - time [optional number] - Ticks since previous event.
	 *  - type [required number] - Type of event.
	 *  - data [optional array|string] - Event data.
	 */
	var MetaEvent = function(params) {
		if (!this) return new MetaEvent(params);
		var p = params || {};
		this.setTime(params.time);
		this.setType(params.type);
		this.setData(params.data);
	};

	MetaEvent.SEQUENCE   = 0x00;
	MetaEvent.TEXT       = 0x01;
	MetaEvent.COPYRIGHT  = 0x02;
	MetaEvent.TRACK_NAME = 0x03;
	MetaEvent.INSTRUMENT = 0x04;
	MetaEvent.LYRIC      = 0x05;
	MetaEvent.MARKER     = 0x06;
	MetaEvent.CUE_POINT  = 0x07;
	MetaEvent.CHANNEL_PREFIX = 0x20;
	MetaEvent.END_OF_TRACK   = 0x2f;
	MetaEvent.TEMPO      = 0x51;
	MetaEvent.SMPTE      = 0x54;
	MetaEvent.TIME_SIG   = 0x58;
	MetaEvent.KEY_SIG    = 0x59;
	MetaEvent.SEQ_EVENT  = 0x7f;

	/**
	 * Set the time for the event in ticks since the previous event.
	 *
	 * @param {number} ticks - The number of ticks since the previous event. May
	 * be zero.
	 */
	MetaEvent.prototype.setTime = function(ticks) {
		this.time = Util.translateTickTime(ticks || 0);
	};

	/**
	 * Set the type of the event. Must be one of the event codes on MetaEvent.
	 *
	 * @param {number} t - Event type.
	 */
	MetaEvent.prototype.setType = function(t) {
		this.type = t;
	};

	/**
	 * Set the data associated with the event. May be a string or array of byte
	 * values.
	 *
	 * @param {string|Array} d - Event data.
	 */
	MetaEvent.prototype.setData = function(d) {
		this.data = d;
	};

	/**
	 * Serialize the event to an array of bytes.
	 *
	 * @returns {Array} The array of serialized bytes.
	 */
	MetaEvent.prototype.toBytes = function() {
		if (!this.type) {
			throw new Error("Type for meta-event not specified.");
		}

		var byteArray = [];
		byteArray.push.apply(byteArray, this.time);
		byteArray.push(0xFF, this.type);

		// If data is an array, we assume that it contains several bytes. We
		// apend them to byteArray.
		if (Array.isArray(this.data)) {
			byteArray.push(this.data.length);
			byteArray.push.apply(byteArray, this.data);
		} else if (typeof this.data == 'number') {
			byteArray.push(1, this.data);
		} else if (this.data !== null && this.data !== undefined) {
			// assume string; may be a bad assumption
			byteArray.push(this.data.length);
			var dataBytes = this.data.split('').map(function(x){ return x.charCodeAt(0) });
			byteArray.push.apply(byteArray, dataBytes);
		} else {
			byteArray.push(0);
		}

		return byteArray;
	};

	/* ******************************************************************
	 * Track class
	 ****************************************************************** */

	/**
	 * Construct a MIDI track.
	 *
	 * Parameters include:
	 *  - events [optional array] - Array of events for the track.
	 */
	var Track = function(config) {
		if (!this) return new Track(config);
		var c = config || {};
		this.events = c.events || [];
	};

	Track.START_BYTES = [0x4d, 0x54, 0x72, 0x6b];
	Track.END_BYTES   = [0x00, 0xFF, 0x2F, 0x00];

	/**
	 * Add an event to the track.
	 *
	 * @param {MidiEvent|MetaEvent} event - The event to add.
	 * @returns {Track} The current track.
	 */
	Track.prototype.addEvent = function(event) {
		this.events.push(event);
		return this;
	};

	/**
	 * Add a note-on event to the track.
	 *
	 * @param {number} channel - The channel to add the event to.
	 * @param {number|string} pitch - The pitch of the note, either numeric or
	 * symbolic.
	 * @param {number} [time=0] - The number of ticks since the previous event,
	 * defaults to 0.
	 * @param {number} [velocity=90] - The volume for the note, defaults to
	 * DEFAULT_VOLUME.
	 * @returns {Track} The current track.
	 */
	Track.prototype.addNoteOn = Track.prototype.noteOn = function(channel, pitch, time, velocity) {
		this.events.push(new MidiEvent({
			type: MidiEvent.NOTE_ON,
			channel: channel,
			param1: Util.ensureMidiPitch(pitch),
			param2: velocity || DEFAULT_VOLUME,
			time: time || 0,
		}));
		return this;
	};

	/**
	 * Add a note-off event to the track.
	 *
	 * @param {number} channel - The channel to add the event to.
	 * @param {number|string} pitch - The pitch of the note, either numeric or
	 * symbolic.
	 * @param {number} [time=0] - The number of ticks since the previous event,
	 * defaults to 0.
	 * @param {number} [velocity=90] - The velocity the note was released,
	 * defaults to DEFAULT_VOLUME.
	 * @returns {Track} The current track.
	 */
	Track.prototype.addNoteOff = Track.prototype.noteOff = function(channel, pitch, time, velocity) {
		this.events.push(new MidiEvent({
			type: MidiEvent.NOTE_OFF,
			channel: channel,
			param1: Util.ensureMidiPitch(pitch),
			param2: velocity || DEFAULT_VOLUME,
			time: time || 0,
		}));
		return this;
	};

	/**
	 * Add a note-on and -off event to the track.
	 *
	 * @param {number} channel - The channel to add the event to.
	 * @param {number|string} pitch - The pitch of the note, either numeric or
	 * symbolic.
	 * @param {number} dur - The duration of the note, in ticks.
	 * @param {number} [time=0] - The number of ticks since the previous event,
	 * defaults to 0.
	 * @param {number} [velocity=90] - The velocity the note was released,
	 * defaults to DEFAULT_VOLUME.
	 * @returns {Track} The current track.
	 */
	Track.prototype.addNote = Track.prototype.note = function(channel, pitch, dur, time, velocity) {
		this.noteOn(channel, pitch, time, velocity);
		if (dur) {
			this.noteOff(channel, pitch, dur, velocity);
		}
		return this;
	};

	/**
	 * Add a note-on and -off event to the track for each pitch in an array of pitches.
	 *
	 * @param {number} channel - The channel to add the event to.
	 * @param {array} chord - An array of pitches, either numeric or
	 * symbolic.
	 * @param {number} dur - The duration of the chord, in ticks.
	 * @param {number} [velocity=90] - The velocity of the chord,
	 * defaults to DEFAULT_VOLUME.
	 * @returns {Track} The current track.
	 */
	Track.prototype.addChord = Track.prototype.chord = function(channel, chord, dur, velocity) {
		if (!Array.isArray(chord) && !chord.length) {
			throw new Error('Chord must be an array of pitches');
		}
		chord.forEach(function(note) {
			this.noteOn(channel, note, 0, velocity);
		}, this);
		chord.forEach(function(note, index) {
			if (index === 0) {
				this.noteOff(channel, note, dur);
			} else {
				this.noteOff(channel, note);
			}
		}, this);
		return this;
	};

	/**
	 * Set instrument for the track.
	 *
	 * @param {number} channel - The channel to set the instrument on.
	 * @param {number} instrument - The instrument to set it to.
	 * @param {number} [time=0] - The number of ticks since the previous event,
	 * defaults to 0.
	 * @returns {Track} The current track.
	 */
	Track.prototype.setInstrument = Track.prototype.instrument = function(channel, instrument, time) {
		this.events.push(new MidiEvent({
			type: MidiEvent.PROGRAM_CHANGE,
			channel: channel,
			param1: instrument,
			time: time || 0,
		}));
		return this;
	};

	/**
	 * Set the tempo for the track.
	 *
	 * @param {number} bpm - The new number of beats per minute.
	 * @param {number} [time=0] - The number of ticks since the previous event,
	 * defaults to 0.
	 * @returns {Track} The current track.
	 */
	Track.prototype.setTempo = Track.prototype.tempo = function(bpm, time) {
		this.events.push(new MetaEvent({
			type: MetaEvent.TEMPO,
			data: Util.mpqnFromBpm(bpm),
			time: time || 0,
		}));
		return this;
	};

	/**
	 * Serialize the track to an array of bytes.
	 *
	 * @returns {Array} The array of serialized bytes.
	 */
	Track.prototype.toBytes = function() {
		var trackLength = 0;
		var eventBytes = [];
		var startBytes = Track.START_BYTES;
		var endBytes   = Track.END_BYTES;

		var addEventBytes = function(event) {
			var bytes = event.toBytes();
			trackLength += bytes.length;
			eventBytes.push.apply(eventBytes, bytes);
		};

		this.events.forEach(addEventBytes);

		// Add the end-of-track bytes to the sum of bytes for the track, since
		// they are counted (unlike the start-of-track ones).
		trackLength += endBytes.length;

		// Makes sure that track length will fill up 4 bytes with 0s in case
		// the length is less than that (the usual case).
		var lengthBytes = Util.str2Bytes(trackLength.toString(16), 4);

		return startBytes.concat(lengthBytes, eventBytes, endBytes);
	};

	/* ******************************************************************
	 * File class
	 ****************************************************************** */

	/**
	 * Construct a file object.
	 *
	 * Parameters include:
	 *  - ticks [optional number] - Number of ticks per beat, defaults to 128.
	 *    Must be 1-32767.
	 *  - tracks [optional array] - Track data.
	 */
	var File = function(config){
		if (!this) return new File(config);

		var c = config || {};
		if (c.ticks) {
			if (typeof c.ticks !== 'number') {
				throw new Error('Ticks per beat must be a number!');
				return;
			}
			if (c.ticks <= 0 || c.ticks >= (1 << 15) || c.ticks % 1 !== 0) {
				throw new Error('Ticks per beat must be an integer between 1 and 32767!');
				return;
			}
		}

		this.ticks = c.ticks || 128;
		this.tracks = c.tracks || [];
	};

	File.HDR_CHUNKID     = "MThd";             // File magic cookie
	File.HDR_CHUNK_SIZE  = "\x00\x00\x00\x06"; // Header length for SMF
	File.HDR_TYPE0       = "\x00\x00";         // Midi Type 0 id
	File.HDR_TYPE1       = "\x00\x01";         // Midi Type 1 id

	/**
	 * Add a track to the file.
	 *
	 * @param {Track} track - The track to add.
	 */
	File.prototype.addTrack = function(track) {
		if (track) {
			this.tracks.push(track);
			return this;
		} else {
			track = new Track();
			this.tracks.push(track);
			return track;
		}
	};

	/**
	 * Serialize the MIDI file to an array of bytes.
	 *
	 * @returns {Array} The array of serialized bytes.
	 */
	File.prototype.toBytes = function() {
		var trackCount = this.tracks.length.toString(16);

		// prepare the file header
		var bytes = File.HDR_CHUNKID + File.HDR_CHUNK_SIZE;

		// set Midi type based on number of tracks
		if (parseInt(trackCount, 16) > 1) {
			bytes += File.HDR_TYPE1;
		} else {
			bytes += File.HDR_TYPE0;
		}

		// add the number of tracks (2 bytes)
		bytes += Util.codes2Str(Util.str2Bytes(trackCount, 2));
		// add the number of ticks per beat (currently hardcoded)
		bytes += String.fromCharCode((this.ticks/256),  this.ticks%256);;

		// iterate over the tracks, converting to bytes too
		this.tracks.forEach(function(track) {
			bytes += Util.codes2Str(track.toBytes());
		});

		return bytes;
	};

	/* ******************************************************************
	 * Exports
	 ****************************************************************** */

	exported.Util = Util;
	exported.File = File;
	exported.Track = Track;
	exported.Event = MidiEvent;
	exported.MetaEvent = MetaEvent;

})( Midi );

if (typeof module != 'undefined' && module !== null) {
	module.exports = Midi;
} else if (typeof exports != 'undefined' && exports !== null) {
	exports = Midi;
} else {
	this.Midi = Midi;
}

},{}],5:[function(require,module,exports){
// https://github.com/gasman/jasmid
//
//

module.exports = function(file){
	return MidiFile(file)
};

function MidiFile(data) {
	function readChunk(stream) {
		var id = stream.read(4);
		var length = stream.readInt32();
		return {
			'id': id,
			'length': length,
			'data': stream.read(length)
		};
	}
	
	var lastEventTypeByte;
	
	function readEvent(stream) {
		var event = {};
		event.deltaTime = stream.readVarInt();
		var eventTypeByte = stream.readInt8();
		if ((eventTypeByte & 0xf0) == 0xf0) {
			/* system / meta event */
			if (eventTypeByte == 0xff) {
				/* meta event */
				event.type = 'meta';
				var subtypeByte = stream.readInt8();
				var length = stream.readVarInt();
				switch(subtypeByte) {
					case 0x00:
						event.subtype = 'sequenceNumber';
						if (length != 2) throw "Expected length for sequenceNumber event is 2, got " + length;
						event.number = stream.readInt16();
						return event;
					case 0x01:
						event.subtype = 'text';
						event.text = stream.read(length);
						return event;
					case 0x02:
						event.subtype = 'copyrightNotice';
						event.text = stream.read(length);
						return event;
					case 0x03:
						event.subtype = 'trackName';
						event.text = stream.read(length);
						return event;
					case 0x04:
						event.subtype = 'instrumentName';
						event.text = stream.read(length);
						return event;
					case 0x05:
						event.subtype = 'lyrics';
						event.text = stream.read(length);
						return event;
					case 0x06:
						event.subtype = 'marker';
						event.text = stream.read(length);
						return event;
					case 0x07:
						event.subtype = 'cuePoint';
						event.text = stream.read(length);
						return event;
					case 0x20:
						event.subtype = 'midiChannelPrefix';
						if (length != 1) throw "Expected length for midiChannelPrefix event is 1, got " + length;
						event.channel = stream.readInt8();
						return event;
					case 0x2f:
						event.subtype = 'endOfTrack';
						if (length != 0) throw "Expected length for endOfTrack event is 0, got " + length;
						return event;
					case 0x51:
						event.subtype = 'setTempo';
						if (length != 3) throw "Expected length for setTempo event is 3, got " + length;
						event.microsecondsPerBeat = (
							(stream.readInt8() << 16)
							+ (stream.readInt8() << 8)
							+ stream.readInt8()
						)
						return event;
					case 0x54:
						event.subtype = 'smpteOffset';
						if (length != 5) throw "Expected length for smpteOffset event is 5, got " + length;
						var hourByte = stream.readInt8();
						event.frameRate = {
							0x00: 24, 0x20: 25, 0x40: 29, 0x60: 30
						}[hourByte & 0x60];
						event.hour = hourByte & 0x1f;
						event.min = stream.readInt8();
						event.sec = stream.readInt8();
						event.frame = stream.readInt8();
						event.subframe = stream.readInt8();
						return event;
					case 0x58:
						event.subtype = 'timeSignature';
						if (length != 4) throw "Expected length for timeSignature event is 4, got " + length;
						event.numerator = stream.readInt8();
						event.denominator = Math.pow(2, stream.readInt8());
						event.metronome = stream.readInt8();
						event.thirtyseconds = stream.readInt8();
						return event;
					case 0x59:
						event.subtype = 'keySignature';
						if (length != 2) throw "Expected length for keySignature event is 2, got " + length;
						event.key = stream.readInt8(true);
						event.scale = stream.readInt8();
						return event;
					case 0x7f:
						event.subtype = 'sequencerSpecific';
						event.data = stream.read(length);
						return event;
					default:
						// console.log("Unrecognised meta event subtype: " + subtypeByte);
						event.subtype = 'unknown'
						event.data = stream.read(length);
						return event;
				}
				event.data = stream.read(length);
				return event;
			} else if (eventTypeByte == 0xf0) {
				event.type = 'sysEx';
				var length = stream.readVarInt();
				event.data = stream.read(length);
				return event;
			} else if (eventTypeByte == 0xf7) {
				event.type = 'dividedSysEx';
				var length = stream.readVarInt();
				event.data = stream.read(length);
				return event;
			} else {
				throw "Unrecognised MIDI event type byte: " + eventTypeByte;
			}
		} else {
			/* channel event */
			var param1;
			if ((eventTypeByte & 0x80) == 0) {
				/* running status - reuse lastEventTypeByte as the event type.
					eventTypeByte is actually the first parameter
				*/
				param1 = eventTypeByte;
				eventTypeByte = lastEventTypeByte;
			} else {
				param1 = stream.readInt8();
				lastEventTypeByte = eventTypeByte;
			}
			var eventType = eventTypeByte >> 4;
			event.channel = eventTypeByte & 0x0f;
			event.type = 'channel';
			switch (eventType) {
				case 0x08:
					event.subtype = 'noteOff';
					event.noteNumber = param1;
					event.velocity = stream.readInt8();
					return event;
				case 0x09:
					event.noteNumber = param1;
					event.velocity = stream.readInt8();
					if (event.velocity == 0) {
						event.subtype = 'noteOff';
					} else {
						event.subtype = 'noteOn';
					}
					return event;
				case 0x0a:
					event.subtype = 'noteAftertouch';
					event.noteNumber = param1;
					event.amount = stream.readInt8();
					return event;
				case 0x0b:
					event.subtype = 'controller';
					event.controllerType = param1;
					event.value = stream.readInt8();
					return event;
				case 0x0c:
					event.subtype = 'programChange';
					event.programNumber = param1;
					return event;
				case 0x0d:
					event.subtype = 'channelAftertouch';
					event.amount = param1;
					return event;
				case 0x0e:
					event.subtype = 'pitchBend';
					event.value = param1 + (stream.readInt8() << 7);
					return event;
				default:
					throw "Unrecognised MIDI event type: " + eventType
					/* 
					console.log("Unrecognised MIDI event type: " + eventType);
					stream.readInt8();
					event.subtype = 'unknown';
					return event;
					*/
			}
		}
	}
	
	stream = Stream(data);
	var headerChunk = readChunk(stream);
	if (headerChunk.id != 'MThd' || headerChunk.length != 6) {
		throw "Bad .mid file - header not found";
	}
	var headerStream = Stream(headerChunk.data);
	var formatType = headerStream.readInt16();
	var trackCount = headerStream.readInt16();
	var timeDivision = headerStream.readInt16();
	
	if (timeDivision & 0x8000) {
		throw "Expressing time division in SMTPE frames is not supported yet"
	} else {
		ticksPerBeat = timeDivision;
	}
	
	var header = {
		'formatType': formatType,
		'trackCount': trackCount,
		'ticksPerBeat': ticksPerBeat
	}
	var tracks = [];
	for (var i = 0; i < header.trackCount; i++) {
		tracks[i] = [];
		var trackChunk = readChunk(stream);
		if (trackChunk.id != 'MTrk') {
			throw "Unexpected chunk - expected MTrk, got "+ trackChunk.id;
		}
		var trackStream = Stream(trackChunk.data);
		while (!trackStream.eof()) {
			var event = readEvent(trackStream);
			tracks[i].push(event);
			//console.log(event);
		}
	}
	
	return {
		'header': header,
		'tracks': tracks
	}
};

/* Wrapper for accessing strings through sequential reads */
function Stream(str) {
	var position = 0;
	
	function read(length) {
		var result = str.substr(position, length);
		position += length;
		return result;
	}
	
	/* read a big-endian 32-bit integer */
	function readInt32() {
		var result = (
			(str.charCodeAt(position) << 24)
			+ (str.charCodeAt(position + 1) << 16)
			+ (str.charCodeAt(position + 2) << 8)
			+ str.charCodeAt(position + 3));
		position += 4;
		return result;
	}

	/* read a big-endian 16-bit integer */
	function readInt16() {
		var result = (
			(str.charCodeAt(position) << 8)
			+ str.charCodeAt(position + 1));
		position += 2;
		return result;
	}
	
	/* read an 8-bit integer */
	function readInt8(signed) {
		var result = str.charCodeAt(position);
		if (signed && result > 127) result -= 256;
		position += 1;
		return result;
	}
	
	function eof() {
		return position >= str.length;
	}
	
	/* read a MIDI-style variable-length integer
		(big-endian value in groups of 7 bits,
		with top bit set to signify that another byte follows)
	*/
	function readVarInt() {
		var result = 0;
		while (true) {
			var b = readInt8();
			if (b & 0x80) {
				result += (b & 0x7f);
				result <<= 7;
			} else {
				/* b is the last byte */
				return result + b;
			}
		}
	}
	
	return {
		'eof': eof,
		'read': read,
		'readInt32': readInt32,
		'readInt16': readInt16,
		'readInt8': readInt8,
		'readVarInt': readVarInt
	}
}
},{}],6:[function(require,module,exports){

},{}],7:[function(require,module,exports){
(function (process){
// Copyright Joyent, Inc. and other Node contributors.
//
// Permission is hereby granted, free of charge, to any person obtaining a
// copy of this software and associated documentation files (the
// "Software"), to deal in the Software without restriction, including
// without limitation the rights to use, copy, modify, merge, publish,
// distribute, sublicense, and/or sell copies of the Software, and to permit
// persons to whom the Software is furnished to do so, subject to the
// following conditions:
//
// The above copyright notice and this permission notice shall be included
// in all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
// OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
// MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN
// NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
// DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
// OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE
// USE OR OTHER DEALINGS IN THE SOFTWARE.

// resolves . and .. elements in a path array with directory names there
// must be no slashes, empty elements, or device names (c:\) in the array
// (so also no leading and trailing slashes - it does not distinguish
// relative and absolute paths)
function normalizeArray(parts, allowAboveRoot) {
  // if the path tries to go above the root, `up` ends up > 0
  var up = 0;
  for (var i = parts.length - 1; i >= 0; i--) {
    var last = parts[i];
    if (last === '.') {
      parts.splice(i, 1);
    } else if (last === '..') {
      parts.splice(i, 1);
      up++;
    } else if (up) {
      parts.splice(i, 1);
      up--;
    }
  }

  // if the path is allowed to go above the root, restore leading ..s
  if (allowAboveRoot) {
    for (; up--; up) {
      parts.unshift('..');
    }
  }

  return parts;
}

// Split a filename into [root, dir, basename, ext], unix version
// 'root' is just a slash, or nothing.
var splitPathRe =
    /^(\/?|)([\s\S]*?)((?:\.{1,2}|[^\/]+?|)(\.[^.\/]*|))(?:[\/]*)$/;
var splitPath = function(filename) {
  return splitPathRe.exec(filename).slice(1);
};

// path.resolve([from ...], to)
// posix version
exports.resolve = function() {
  var resolvedPath = '',
      resolvedAbsolute = false;

  for (var i = arguments.length - 1; i >= -1 && !resolvedAbsolute; i--) {
    var path = (i >= 0) ? arguments[i] : process.cwd();

    // Skip empty and invalid entries
    if (typeof path !== 'string') {
      throw new TypeError('Arguments to path.resolve must be strings');
    } else if (!path) {
      continue;
    }

    resolvedPath = path + '/' + resolvedPath;
    resolvedAbsolute = path.charAt(0) === '/';
  }

  // At this point the path should be resolved to a full absolute path, but
  // handle relative paths to be safe (might happen when process.cwd() fails)

  // Normalize the path
  resolvedPath = normalizeArray(filter(resolvedPath.split('/'), function(p) {
    return !!p;
  }), !resolvedAbsolute).join('/');

  return ((resolvedAbsolute ? '/' : '') + resolvedPath) || '.';
};

// path.normalize(path)
// posix version
exports.normalize = function(path) {
  var isAbsolute = exports.isAbsolute(path),
      trailingSlash = substr(path, -1) === '/';

  // Normalize the path
  path = normalizeArray(filter(path.split('/'), function(p) {
    return !!p;
  }), !isAbsolute).join('/');

  if (!path && !isAbsolute) {
    path = '.';
  }
  if (path && trailingSlash) {
    path += '/';
  }

  return (isAbsolute ? '/' : '') + path;
};

// posix version
exports.isAbsolute = function(path) {
  return path.charAt(0) === '/';
};

// posix version
exports.join = function() {
  var paths = Array.prototype.slice.call(arguments, 0);
  return exports.normalize(filter(paths, function(p, index) {
    if (typeof p !== 'string') {
      throw new TypeError('Arguments to path.join must be strings');
    }
    return p;
  }).join('/'));
};


// path.relative(from, to)
// posix version
exports.relative = function(from, to) {
  from = exports.resolve(from).substr(1);
  to = exports.resolve(to).substr(1);

  function trim(arr) {
    var start = 0;
    for (; start < arr.length; start++) {
      if (arr[start] !== '') break;
    }

    var end = arr.length - 1;
    for (; end >= 0; end--) {
      if (arr[end] !== '') break;
    }

    if (start > end) return [];
    return arr.slice(start, end - start + 1);
  }

  var fromParts = trim(from.split('/'));
  var toParts = trim(to.split('/'));

  var length = Math.min(fromParts.length, toParts.length);
  var samePartsLength = length;
  for (var i = 0; i < length; i++) {
    if (fromParts[i] !== toParts[i]) {
      samePartsLength = i;
      break;
    }
  }

  var outputParts = [];
  for (var i = samePartsLength; i < fromParts.length; i++) {
    outputParts.push('..');
  }

  outputParts = outputParts.concat(toParts.slice(samePartsLength));

  return outputParts.join('/');
};

exports.sep = '/';
exports.delimiter = ':';

exports.dirname = function(path) {
  var result = splitPath(path),
      root = result[0],
      dir = result[1];

  if (!root && !dir) {
    // No dirname whatsoever
    return '.';
  }

  if (dir) {
    // It has a dirname, strip trailing slash
    dir = dir.substr(0, dir.length - 1);
  }

  return root + dir;
};


exports.basename = function(path, ext) {
  var f = splitPath(path)[2];
  // TODO: make this comparison case-insensitive on windows?
  if (ext && f.substr(-1 * ext.length) === ext) {
    f = f.substr(0, f.length - ext.length);
  }
  return f;
};


exports.extname = function(path) {
  return splitPath(path)[3];
};

function filter (xs, f) {
    if (xs.filter) return xs.filter(f);
    var res = [];
    for (var i = 0; i < xs.length; i++) {
        if (f(xs[i], i, xs)) res.push(xs[i]);
    }
    return res;
}

// String.prototype.substr - negative index don't work in IE8
var substr = 'ab'.substr(-1) === 'b'
    ? function (str, start, len) { return str.substr(start, len) }
    : function (str, start, len) {
        if (start < 0) start = str.length + start;
        return str.substr(start, len);
    }
;

}).call(this,require('_process'))
},{"_process":8}],8:[function(require,module,exports){
// shim for using process in browser
var process = module.exports = {};

// cached from whatever global is present so that test runners that stub it
// don't break things.  But we need to wrap it in a try catch in case it is
// wrapped in strict mode code which doesn't define any globals.  It's inside a
// function because try/catches deoptimize in certain engines.

var cachedSetTimeout;
var cachedClearTimeout;

(function () {
    try {
        cachedSetTimeout = setTimeout;
    } catch (e) {
        cachedSetTimeout = function () {
            throw new Error('setTimeout is not defined');
        }
    }
    try {
        cachedClearTimeout = clearTimeout;
    } catch (e) {
        cachedClearTimeout = function () {
            throw new Error('clearTimeout is not defined');
        }
    }
} ())
function runTimeout(fun) {
    if (cachedSetTimeout === setTimeout) {
        //normal enviroments in sane situations
        return setTimeout(fun, 0);
    }
    try {
        // when when somebody has screwed with setTimeout but no I.E. maddness
        return cachedSetTimeout(fun, 0);
    } catch(e){
        try {
            // When we are in I.E. but the script has been evaled so I.E. doesn't trust the global object when called normally
            return cachedSetTimeout.call(null, fun, 0);
        } catch(e){
            // same as above but when it's a version of I.E. that must have the global object for 'this', hopfully our context correct otherwise it will throw a global error
            return cachedSetTimeout.call(this, fun, 0);
        }
    }


}
function runClearTimeout(marker) {
    if (cachedClearTimeout === clearTimeout) {
        //normal enviroments in sane situations
        return clearTimeout(marker);
    }
    try {
        // when when somebody has screwed with setTimeout but no I.E. maddness
        return cachedClearTimeout(marker);
    } catch (e){
        try {
            // When we are in I.E. but the script has been evaled so I.E. doesn't  trust the global object when called normally
            return cachedClearTimeout.call(null, marker);
        } catch (e){
            // same as above but when it's a version of I.E. that must have the global object for 'this', hopfully our context correct otherwise it will throw a global error.
            // Some versions of I.E. have different rules for clearTimeout vs setTimeout
            return cachedClearTimeout.call(this, marker);
        }
    }



}
var queue = [];
var draining = false;
var currentQueue;
var queueIndex = -1;

function cleanUpNextTick() {
    if (!draining || !currentQueue) {
        return;
    }
    draining = false;
    if (currentQueue.length) {
        queue = currentQueue.concat(queue);
    } else {
        queueIndex = -1;
    }
    if (queue.length) {
        drainQueue();
    }
}

function drainQueue() {
    if (draining) {
        return;
    }
    var timeout = runTimeout(cleanUpNextTick);
    draining = true;

    var len = queue.length;
    while(len) {
        currentQueue = queue;
        queue = [];
        while (++queueIndex < len) {
            if (currentQueue) {
                currentQueue[queueIndex].run();
            }
        }
        queueIndex = -1;
        len = queue.length;
    }
    currentQueue = null;
    draining = false;
    runClearTimeout(timeout);
}

process.nextTick = function (fun) {
    var args = new Array(arguments.length - 1);
    if (arguments.length > 1) {
        for (var i = 1; i < arguments.length; i++) {
            args[i - 1] = arguments[i];
        }
    }
    queue.push(new Item(fun, args));
    if (queue.length === 1 && !draining) {
        runTimeout(drainQueue);
    }
};

// v8 likes predictible objects
function Item(fun, array) {
    this.fun = fun;
    this.array = array;
}
Item.prototype.run = function () {
    this.fun.apply(null, this.array);
};
process.title = 'browser';
process.browser = true;
process.env = {};
process.argv = [];
process.version = ''; // empty string to avoid regexp issues
process.versions = {};

function noop() {}

process.on = noop;
process.addListener = noop;
process.once = noop;
process.off = noop;
process.removeListener = noop;
process.removeAllListeners = noop;
process.emit = noop;

process.binding = function (name) {
    throw new Error('process.binding is not supported');
};

process.cwd = function () { return '/' };
process.chdir = function (dir) {
    throw new Error('process.chdir is not supported');
};
process.umask = function() { return 0; };

},{}]},{},[1]);
