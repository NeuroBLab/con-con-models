import met_brewer as metb

pal = metb.met_brew('Egypt', brew_type='discrete') 

# Color reference to identify each layer through the paper
lcolor = {
    "L23" : pal[0],
    "L4" :  pal[3],
    "L23_modelE" : pal[2], 
    "L23_modelI" : pal[1],
    "Total" : "gray"
}

#For the things in the model that do not need to be different
lcolor['L4_modelE'] = lcolor['L4']
lcolor['Total_modelE'] = lcolor['Total']

# Color reference to identify each layer through the paper
angles=["darkorange", "gold", "purple", "violet"]