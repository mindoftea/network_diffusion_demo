'use strict'

Vue.component( "controlslider" , {
  template: "#controlslider-template" ,
  props: [ "v" ] } )

Vue.component( "controlpanel" , {
  template: "#controlpanel-template" ,
  props: [ "vars" ] } )

Vue.component( "graphview" , {
  template: "#graphview-template" ,
  methods: {clickOn: function() {this.interact.click=true}},
  props: [ "nodes" , "interact" ] } )

Vue.component( "gnode" , {
  template: "#gnode-template" ,
  props: [ "node" ] ,
  methods: { /* node-specific click can be implemented here with buffer */},
  computed: {
    color: function() {
      const c = this.node.color ; const h = 1024
      if ( isNaN(c.r) || isNaN(c.g) || isNaN(c.b) ) return "rgb(255,0,0)"
      // x*h<<8>>10 == Math.floor(256*x)
      return `rgb(${c.r*h<<8>>10},${c.g*h<<8>>10},${c.b*h<<8>>10})` } } } )

Vue.component( "gedges" , {
  template: "#gedges-template" ,
  props: ["node"] } )

Vue.component( "gedge" , {
    template: "#gedge-template" ,
    props: ["edge"],
    computed: {
      opacity: function() {
        const dx = this.edge.start.x - this.edge.end.x
        const dy = this.edge.start.y - this.edge.end.y
        const r = Math.min( 1 , Math.sqrt( dx*dx + dy*dy ) / 50 )
        // highlighting edges by stress (length) makes the structure clearer
        return ( 0.15 + r*r*r ) * this.edge.active } } } )