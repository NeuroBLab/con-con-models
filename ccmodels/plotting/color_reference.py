import met_brewer as metb

pal = metb.met_brew('Egypt', brew_type='discrete') 
pal_extended = metb.met_brew("Juarez", brew_type='discrete')

# Color reference to identify each layer through the paper
lcolor = {
    "L23" : pal[0],
    "L4" :  pal[3],
    "L23_modelE" : pal[2], 
    "L23_modelI" : pal[1],
    "Total" : "gray"
}

print(pal)

#For the things in the model that do not need to be different
lcolor['L4_modelE'] = lcolor['L4']
lcolor['Total_modelE'] = lcolor['Total']

# Color reference to identify each layer through the paper
angles=["darkorange", "gold", "purple", "violet"]

#Markers over lines for plots as function of delta theta
ms = 3
mc = "#303030"

#Reshuffling code
reshuf_color = ["#353535", pal_extended[4], pal_extended[3], pal_extended[2]]


# --------------------------------
# Color Tools 
# -------------------------------_

def rgb2hex(r, g, b):
    return "#{0:02x}{1:02x}{2:02x}".format(r,g,b)

def rgb2hex(c):
    return "#{0:02x}{1:02x}{2:02x}".format(c[0], c[1], c[2])
    
def hex2rgb(c):
    cs = c[1:]
    return tuple(int(cs[i:i+2], 16) for i in (0, 2, 4))

def darken(color, n, factor):
    c = hex2rgb(color) 
    new_colors = []
    newc = [0,0,0]
    for i in range(1,n+1):
        for k in range(3):
            newc[k] = max(int(c[k] - c[k]*factor*i), 0)
        new_colors.append(rgb2hex(newc))

    return new_colors

def ligthen(color, n, factor):
    c = hex2rgb(color) 
    new_colors = []
    newc = [0,0,0]
    for i in range(1,n+1):
        for k in range(3):
            newc[k] = min(int(c[k] + (255-c[k])*factor*i), 255)
        new_colors.append(rgb2hex(newc))

    return new_colors



def get_shades(color, n=2, factor=0.8):
    dark = darken(color, n//2, factor)
    ligth = ligthen(color, n//2, factor)
    return dark + [color] + ligth