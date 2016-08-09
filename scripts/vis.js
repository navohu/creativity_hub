var ymin = 0;
var ymax = 30;
var xmin = 0;
var xmax = 300;
var positions = [];
var Graph;

function initialiseGraph(){
  d3.selectAll("path.line").remove();
  positions = [];
  // can be used to graph loss, or accuract over time
  Graph = d3.select('#visualisation')
    .attr('width', '100%')
    .attr('height', 500),
    WIDTH = $("#visualisation").parent().width(),
    HEIGHT = 500,
    MARGINS = {
      top: 20,
      right: 20,
      bottom: 20,
      left: 20
    },
    xRange = d3.scaleLinear().range([MARGINS.left, WIDTH - MARGINS.right]).domain([xmin, xmax]),

    yRange = d3.scaleLinear().range([HEIGHT - MARGINS.top, MARGINS.bottom]).domain([ymin, ymax]),

    xAxis = d3.axisBottom()
      .scale(xRange),

    yAxis = d3.axisLeft()
      .scale(yRange);

  // Graph.classed("svg-container", true)
  //   .append("svg")
  //   .attr("preserveAspectRatio", "xMinYMin meet")
  //   .attr("viewBox", "0 0 600 400")
  //   .classed("svg-content-responsive", true);

  Graph.append('svg:g')
    .attr('class', 'x axis')
    .attr('transform', 'translate(0,' + (HEIGHT - MARGINS.bottom) + ')')
    .call(xAxis)
    .append("text")
    .attr('x', MARGINS.right)
    .attr('y', HEIGHT - MARGINS.bottom)
    .attr('stroke', '#a94442')
    .style("text-anchor", "left")
    .text('Tick iteration');

  Graph.append('svg:g')
    .attr('class', 'y axis')
    .attr('transform', 'translate(' + (MARGINS.left) + ',0)')
    .call(yAxis)
    .append('text')
    .attr('x', MARGINS.left)
    .attr('y', MARGINS.top -10)
    .attr('stroke', '#a94442')
    .style("text-anchor", "middle")
    .text('Median perplexity');
}

function updateVisual(step, y){
  d3.select('#visualisation').selectAll("*").remove();
  Graph = d3.select('#visualisation')
    .attr('width', '100%')
    .attr('height', 500),
    WIDTH = $("#visualisation").parent().width(),
    HEIGHT = 500,
    MARGINS = {
      top: 20,
      right: 20,
      bottom: 20,
      left: 20
    },

    xRange = d3.scaleLinear().range([MARGINS.left, WIDTH - MARGINS.right]).domain([xmin, getMaxXDomain(step)]),

    yRange = d3.scaleLinear().range([HEIGHT - MARGINS.bottom, MARGINS.top]).domain([getMaxYDomain(y), getMinYDomain(y)]),
    
    xAxis = d3.axisBottom()
      .scale(xRange),

    yAxis = d3.axisLeft()
      .scale(yRange);

  var transformY = Graph.append("g")
    .attr('class', 'y axis')
    .attr('transform', 'translate(' + (MARGINS.left) + ',0)')
    .call(yAxis)
    .append('text')
    .style("text-anchor", "middle")
    .text('Median perplexity');
  
  var transformX = Graph.append("g")
    .attr('class', 'x axis')
    .attr('transform', 'translate(0,' + (HEIGHT - MARGINS.bottom) + ')')
    .call(xAxis)
    .append("text")
    .style("text-anchor", "left")
    .text('Tick iteration');

  // console.log("Step: " + positions.step + "\n y: " + positions.y);
  positions.push({step: step, y: y});

  var lineFunc = d3.line()
    .x(function(d){ return xRange(d.step);})
    .y(function(d){ return yRange(d.y)});

  Graph.append('path')
    .attr('class', 'line')
    .attr('stroke', 'blue')
    .attr('stroke-width', 2)
    .attr('fill', 'none')
    .attr('d', lineFunc(positions));
}



function getMinYDomain(y){
  if(ymin<y) {
    ymin = y*1.1;
    return ymin;
  }
  else return ymin;
}

function getMaxYDomain(y){
  if(ymax>y){
    ymax = y*0.8;
    return ymax;
  }
  else return ymax;
}

function getMaxXDomain(step){
  if(step > xmax){
    xmax *= 2;
    return xmax;
  }
  else return xmax;
}
