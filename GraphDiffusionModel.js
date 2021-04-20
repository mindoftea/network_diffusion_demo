'use strict'

const T = tf.tidy

const GraphDiffusionModelInstance = (() => {

const Chroma = tf.tensor([
  [ .5 , 1. , 0. ],
  [ .5 ,10. , .5 ],
  [ 0. , 0. , .2 ],
  [ 1. , 0. , 1. ]
])
const dChrom = tf.tensor([ .1 , .1 , .1 ])
const colorMap = states => {
  const colors = tf.matMul( states , Chroma ).add( dChrom )
  return colors.div( tf.expandDims( tf.sum( colors , 1 ).add( 1e-5 ).pow(0.75) , 1 ) ).mul(-1).add(1)
}

class GraphDiffusionModelInstance {
  constructor( network , vars , coord_transform ) {
    this.G = tf.tensor(network.G) ; this.X = tf.tensor(network.X)
    this.xs = tf.tensor( network.xs.map(coord_transform) )
    this.select = tf.zeros([ network.G.length ])
    this.field = tf.zeros([ network.G.length ])
    this.statesvec = tf.zeros([ network.G.length , 4 ])
    this.updateDensity( vars.density )
    this.updateVars( vars.contagion , vars.vaccination , vars.immunity )
  }

  updateDensity( density ) {
    // transmission_matrix = tf.tidy(
      // () => sel.greater(data.variables[0].value)
      // .cast('float32').mul(0.0005).mul(+data.variables[1].value))
  }

  topo() { return this.G.array() }

  updateVars( contagion , vaccination , immunity ) {
    this.transmission_vector && this.transmission_vector.dispose()
    this.transition_matrix && this.transition_matrix.dispose()
    this.diffusion_constant && this.diffusion_constant.dispose()
    this.vax_rate && this.vax_rate.dispose()
    this.vax_rate = tf.scalar(vaccination*vaccination)
    this.transmission_vector = tf.tensor( [2,1,0,0] )
    this.transition_matrix = tf.tensor( [
      [  2 ,  4 ,  0 ,  0 ],
      [ -4 , .5 ,  1 ,  0 ],
      [-20 , -2 ,2*immunity-2 ,  0 ],
      [-15 , -3 ,  0 ,4*immunity-4 ] ] ).mul(0.005)
    this.diffusion_constant = tf.tensor(0.002).mul(contagion*contagion) }

  // infect(clicks) {
  //   const sv = this.statesvec.add(
  //     tf.tile( [[9,0,0,0]] , [nodes.length,1] )
  //      .mul( tf.expandDims( tf.tensor(clicks) , 1 ) ) )
  //   this.statesvec.dispose()
  //   return this.statesvec = sv}
  
  infect() {
    const sv = this.statesvec.add(
      tf.tile( [[60,0,0,0]] , [nodes.length,1] )
       .mul( tf.expandDims(this.select, 1) ) )
    this.statesvec.dispose()
    return this.statesvec = sv}

  immunize() {
    const sv = this.statesvec.add(tf.tile( [[0,0,0,3]] , [nodes.length,1] )
    .mul( tf.randomUniform( [this.G.shape[1],1] , 0 , 1e3 ).less(this.vax_rate) ) )
    this.statesvec.dispose()
    return this.statesvec = sv }

  transit() {
    const sT = tf.matMul(this.statesvec, this.transition_matrix)
    const sv = tf.clipByValue( sT.mul(0.01).add(this.statesvec.mul(
      tf.exp( sT.mul(-1).add(Math.LN2) ).add(1).pow(-1).mul(3/2).add(1/2) )) , 0 , 100 )
    this.statesvec.dispose()
    return this.statesvec = sv }

  transmit() {
    const sv = this.statesvec.add( tf.tile( [[1,0,0,0]] , [this.G.shape[0],1] )
     .mul( tf.transpose( tf.relu(tf.matMul( tf.expandDims( tf.sum( this.statesvec
      .mul(this.transmission_vector) , 1 ) , 0 ) , this.G.mul(this.diffusion_constant) )
       .sub( tf.randomUniform( [1,this.G.shape[1]] , 0 , 1 ) ) ).pow(.1) ) ) )
    this.statesvec.dispose()
    return this.statesvec = sv }

  respatialize() { }

  selectPoint(x) {
    this.select.dispose()
    this.field.dispose()
    return [
      this.select = tf.prod(tf.exp( this.xs.sub(x).pow(2).mul(-.01) ), 1) , 
      this.field = tf.reshape(
        tf.matMul(tf.expandDims( this.select , 0 ) , this.X )
         .mul(.3).pow(.2) , [-1] ) ] }

  state() { return Promise.all([
    this.xs.array() ,
    tf.tidy(() => colorMap( this.statesvec )).array() ,
    this.select.array() , this.field.array() ]) } }

return GraphDiffusionModelInstance } )()