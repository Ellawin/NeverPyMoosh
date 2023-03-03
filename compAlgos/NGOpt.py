import numpy as np # to make arrays
import matplotlib.pyplot as plt # to trace figures
import nevergrad as ng # to run the optimization
from nevergrad.functions import photonics # to define the cost function
from PyMoosh import * # to analyse the result

num_layers = 30 # number of layers in the structure 
func = photonics.Photonics(
    "chirped",
    num_layers
) #cost function stored in Photonics module of Nevergrad

# The parameters to optimized are stored in a 1D array of number layers values 
parametrization = ng.p.Array(shape=(num_layers,))

# We create an initialized structure. The values are 0 by default.  
child = parametrization.spawn_child() 
#print("value of the child before: ", child.value)

# We change the initial values from 0 to 150 nm, as we don't want layers of 0 nm
# thicknesses
child.value = np.tile([150,150],int(num_layers/2))
#print("value of the child after : ", child.value)

# We store this structure as the initialized structures for the next steps of 
# optimization
parametrization.value = child.value

# We set the lower and upper values authorized for the thicknesses 
parametrization = parametrization.set_bounds(lower=10, upper=200)

# We set the optimizer "DE" with a budget of 100 evaluations of the cost function
# and the parametrization defined above
#optim = ng.optimizers.registry["DE"](parametrization, budget = 16000)
#optim2 = ng.optimizers.registry["TwoPointsDE"](parametrization, budget = 16000)
optim = ng.optimizers.registry["NGOpt"](parametrization, budget = 16000)
#optim4 = ng.optimizers.registry["GeneticDE"](parametrization, budget = 16000)

#optim = ng.optimizers.DE(parametrization, budget=100).with_variant("current-to-best")

# We initialized two arrays to store the data
#loss = np.empty(optim.budget)
#loss2 = np.empty(optim.budget)
best_losses = []
losses = []
# for each iteration of the optimization, we create a structure x, we compute its 
# cost function and store it in the 'loss' and 'loss2' arrays.
for i in range(optim.budget):
    x = optim.ask() # creation of a structure
    #loss[i] = func(x.value) # compute the cost function of the 'x' structure
    loss = func(x.value)
    losses.append(loss)
    #optim.tell(x, loss[i]) # i don't know why this line is useful, but it is. If 
    optim.tell(x, loss)
    # it is commented, the code doesn't run.
    #reco2 = optim.provide_recommendation() # provide the best structure computed 
    reco = optim.provide_recommendation()
    best_loss = func(reco.value)
    best_losses.append(best_loss)
    #print("iteration:", i+1, "- Best Loss:", best_loss, "- Best structure:", reco.value)

plt.figure(1)
plt.plot(best_losses)
plt.title("NGOpt")
plt.xlabel("Iteration")
plt.ylabel("Loss")
plt.savefig("ConvNGOpt.png") 

# We set the permittivities of the materials used. 
# Those materials are imposed in the Chirped() function in Photonics package of 
# Nevergrad.
materials = [1., 1.4142135623730951**2, 1.7320508075688772**2]

# We define the thicknesses of the multilayers. They are given by the 'best' 
# structure optimized. We add a upperstrate of air and a substrate of the upper 
# permittivity material. 
list = [2,1] # our structure begin with the upper RI material
layer_type = [0] + (int(num_layers/2))*list + [2] # our structure is an alternance 
# of the upper and lower RI materials, with an air incident medium and a upper RI 
# material substrate
thickness = [0] + reco.value.tolist() + [0]

# We define the all structure with PyMoosh package
multilayers = Structure(materials,layer_type,thickness)

incidence = 0 # angle of incidence (in degrees)
polarization = 0 # 0 for TE (or s-polarization), 1 otherwise
wl_min = 400 # lower value of wavelength spectrum, in nanometers
wl_max = 900 # upper value of wavelength spectrum, in nanometers
n_points = 200 # number of wavelngths computed to represent the spectrum

[wl,r,t,R,T] = Spectrum(multilayers, incidence,polarization,wl_min,wl_max,n_points)

# visualization of the reflectance of the structure
plt.figure(2)
plt.title("NGOpt")
plt.plot(wl,R)
plt.savefig("RefNGOpt.png") 

M = Visu_struct(multilayers,600)

plt.figure(3)
plt.title("NGOpt")
plt.matshow(M, cmap = 'bone', aspect = 'auto') 
plt.colorbar() 
plt.savefig("StructureNGOpt.png") 
# The lower RI material is represented in black, and the upper RI material is in 
# white. It is not coherent with a multilayer beggining with a upper RI material as 
# attended. Be careful.

wavelength = 600 # wavelength of the incident beam
window = Window(70*wavelength, 0.5, 30., 30.)
beam = Beam(wavelength, np.pi/4,0,10*wavelength)

E = field(multilayers, beam, window)

plt.figure(4)
plt.title("NGOpt")
plt.imshow(abs(E),cmap='jet',extent=[0,window.width,0,sum(multilayers.thickness)],aspect='auto')
plt.colorbar()
plt.savefig("ChampsNGOpt.png") 