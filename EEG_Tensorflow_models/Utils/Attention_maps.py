#--------------------##--------------------------------------
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tf_keras_vis.utils.scores import CategoricalScore
from tf_keras_vis.utils.model_modifiers import ReplaceToLinear
from matplotlib import cm
from tf_keras_vis.gradcam import Gradcam
from tf_keras_vis.saliency import Saliency
from tf_keras_vis.scorecam import Scorecam
from tf_keras_vis.gradcam_plus_plus import GradcamPlusPlus
from sklearn.metrics import pairwise_distances
from mpl_toolkits.axes_grid1 import make_axes_locatable
import itertools
import re
#--------------------##--------------------------------------


def centroid_(X):
   D = pairwise_distances(X, X.mean(axis=0).reshape(1,-1))
   inertia_ = D.mean()
   return np.argmin(D),inertia_

def plot_attention(tmpr_,layer_name,list_class,figsize=(10,5), transpose=False):
    
    
    if transpose:
      x_label_list = layer_name 
      nC = len(list_class)
      nl = len(layer_name)
      ncols,nrows = tmpr_.shape

      y_label_list = []
      for ii in range(nC):
          y_label_list += str(list_class[ii])

      dw = nrows/nl
      list_xticks = []
      for ii in range(nl):
        list_xticks += [int(dw*(0.5+ii))]
      dw = ncols/nC
      list_yticks = []
      for ii in range(nC):
        list_yticks += [int(dw*(0.5+ii))]

    else:
      y_label_list = layer_name 
      nC = len(list_class)
      nl = len(layer_name)
      nrows,ncols = tmpr_.shape

      x_label_list = []
      for ii in range(nC):
          x_label_list += str(list_class[ii])

      dw = nrows/nl
      list_yticks = []
      for ii in range(nl):
        list_yticks += [int(dw*(0.5+ii))]
      dw = ncols/nC
      list_xticks = []
      for ii in range(nC):
        list_xticks += [int(dw*(0.5+ii))]
    
    plt.figure(figsize=figsize)
    ax = plt.gca()
    im = ax.imshow(tmpr_)
    im = ax.imshow(tmpr_)
    ax.set_yticks(list_yticks)
    ax.set_yticklabels(y_label_list)
    ax.set_xticks(list_xticks)
    ax.set_xticklabels(x_label_list,rotation = 'vertical') #
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)
        
    plt.colorbar(im, cax=cax,extend='both',
                 ticks=[np.round(tmpr_.min(),1), np.round(0.5*(tmpr_.max()-tmpr_.min()),1), np.round(tmpr_.max(),1)])
    #plt.yticks(rotation=45)

       
    plt.tight_layout()    
    plt.show()
    return

def attention_wide(modelw,rel_model_name,layer_name,X_train,y_train,
                   normalize_cam=False,norm_max_min=False,norm_c=True,
                   plot_int=False,centroid_=False,smooth_samples=20,
                   smooth_noise=0.20,transpose=False):
    #-------------------------------------------------------------------------------
    # define trial sample to visualize
    # change activations of last layer by linear
    replace2linear = ReplaceToLinear()
    #relevance model
    
    if rel_model_name == 'Gradcam':
        gradcamw = Gradcam(modelw,
                        model_modifier=replace2linear,
                        clone=True)
    elif rel_model_name == 'Gradcam++':
        gradcamw = GradcamPlusPlus(modelw,
                              model_modifier=replace2linear,
                              clone=False) 
        
    elif rel_model_name == 'Scorecam':
        scorecamw = Scorecam(modelw)
        
    elif rel_model_name == 'Saliency':
          saliencyw = Saliency(modelw,
                              model_modifier=replace2linear,
                              clone=True)
          layer_name = [''] #saliency doesn't depend on different layers    
    nC = len(np.unique(y_train))
    relM = [None]*nC
    if type(X_train)==list:
        n_inputs = len(X_train)
        new_input = [None]*n_inputs

    for c in range(len(np.unique(y_train))):  
      id_sample = y_train == np.unique(y_train)[c]

      if (type(X_train)==list) and (rel_model_name != 'Saliency'):
        relM[c] = np.zeros((sum(id_sample),X_train[0].shape[1],X_train[0].shape[2],len(layer_name)))
        #print(1,relM[c].shape)
      elif (type(X_train)==list) and (rel_model_name == 'Saliency'):   
        relM[c] = np.zeros((sum(id_sample),X_train[0].shape[1],X_train[0].shape[2],len(X_train)))
        #print(2,relM[c].shape)        
      else:
        relM[c] = np.zeros((sum(id_sample),X_train.shape[1],X_train.shape[2],len(layer_name)))
        #print(3,relM[c].shape)
      score = CategoricalScore(list(y_train[id_sample])) #-> [0] para probar a una clase diferente
      if type(X_train)==list:
          for ni in range(n_inputs):
              new_input[ni] = X_train[ni][id_sample]
      else:
        new_input = X_train[id_sample]        
      #print('rel',rel_model_name,'layer',layer_name[l])
      for l in range(len(layer_name)):
          print(rel_model_name,'class', np.unique(y_train)[c],'layer',layer_name[l])
      # label score -> target label accoring to the database
      #-----------------------------------------------------------------------------
      # generate heatmap with GradCAM
          if (rel_model_name == 'Gradcam') or (rel_model_name == 'Gradcam++'):
              rel = gradcamw(score,
                          new_input,
                          penultimate_layer=layer_name[l], #layer to be analized
                          expand_cam=True,
                          normalize_cam=normalize_cam)
          elif rel_model_name == 'Saliency': #saliency map is too noisy, so let’s remove noise in the saliency map using SmoothGrad!
                rel = saliencyw(score, new_input,smooth_samples=smooth_samples,
                                smooth_noise=smooth_noise,normalize_map=normalize_cam) #, smooth_samples=20,smooth_noise=0.20) # The number of calculating gradients iterations.
                            
          elif rel_model_name == 'Scorecam':     
              rel = scorecamw(score, new_input, penultimate_layer=layer_name[l], #layer to be analized
                          expand_cam=True,
                          normalize_cam=normalize_cam) #max_N=10 -> faster scorecam
      
          #save model
          if rel_model_name != 'Saliency':
            if type(X_train)==list: 
              tcc = rel[0]
            else: 
              tcc = rel
            dimc = tcc.shape
            tccv = tcc.ravel()
            tccv[np.isnan(tccv)] = 0
            tcc = tccv.reshape(dimc)
            if norm_max_min: #normalizing along samples
              tcc = MinMaxScaler().fit_transform(tcc.reshape(dimc[0],-1).T).T
              tcc = tcc.reshape(dimc)
            relM[c][...,l] = tcc
            if l==0: 
              tmp = np.median(relM[c][...,l],axis=0)#relM[c][...,l].mean(axis=0)
            else: 
              if transpose:
                tmp = np.c_[tmp,np.median(relM[c][...,l],axis=0)]#np.r_[tmp,relM[c][...,l].mean(axis=0)]  #centroid
              else:  
                tmp = np.r_[tmp,np.median(relM[c][...,l],axis=0)]#np.r_[tmp,relM[c][...,l].mean(axis=0)]  #centroid
          else: #saliency
            if type(X_train)==list: 
              tcc = np.zeros((rel[0].shape[0],rel[0].shape[1],rel[0].shape[2],len(rel)))
              for ii in range(len(rel)):
                  tcc[...,ii] = rel[ii]
            else: 
              tcc = rel
            dimc = tcc.shape
            tccv = tcc.ravel()
            tccv[np.isnan(tccv)] = 0
            tcc = tccv.reshape(dimc)
            if norm_max_min: #normalizing along samples
              tcc = MinMaxScaler().fit_transform(tcc.reshape(dimc[0],-1).T).T
              tcc = tcc.reshape(dimc)
            relM[c] = tcc
            if type(X_train)==list: 
              tmp = np.median(tcc[...,0],axis=0)
              for ii in range(len(rel)-1):
                  if transpose: 
                    tmp = np.c_[tmp,np.median(tcc[...,ii+1],axis=0)]
                  else:
                    tmp = np.r_[tmp,np.median(tcc[...,ii+1],axis=0)]
            else:
               tmp = np.median(tcc,axis=0)
                
      if norm_c: #normalizing along layers
        tmp = tmp/(1e-8+tmp.max())
      if c==0: 
        tmpr = tmp
      else:  
        if transpose:
          tmpr = np.r_[tmpr,tmp]  
        else:
          tmpr = np.c_[tmpr,tmp]  
      #print(tmp.shape,tmp.max())    
      if plot_int: #plot every class
         plt.imshow(tmp)
         plt.colorbar(orientation='horizontal')
         plt.axis('off')
         plt.show()
    tmpr = tmpr/(1e-8+tmpr.max())
    list_class = np.unique(y_train)
    plot_attention(tmpr,layer_name,list_class,transpose=transpose)

    return relM,tmpr

def Attention_maps(rel_model_name,layer_name,model,X,y,function_combination=None,**kwargs):
  #rel_model_name = #Gradcam, Gradcam++, Saliency, Scorecam
  #layer_name = layer name in the model, must be a list of strings. If string it will be loaded as regular expresion
  #model =  TF model\

  # get all layer names
  model_layer_names=list(map(lambda x: x.name,model.layers))
  #if layer_name is str, it is considered as a regular expresion
  if type(layer_name)==str:
    layer_name = re.findall(layer_name," ".join(model_layer_names),re.I)
  else:
    exits_layers = [not layer in model_layer_names for layer in layer_name]
    assert not any(exits_layers), 'layers {} does not exist'.format(list(itertools.compress(layer_name,exits_layers)))
  # X = train samples
  # y = train labels
  relM_ = [None]*len(rel_model_name) #relM[m] -> number classes x input image resolution x number of layers 
  tmpr_ = [None]*len(rel_model_name)
  for m in range(len(rel_model_name)):
      relM_[m],tmpr_[m] = attention_wide(model,rel_model_name[m],layer_name,
                                        X,y,kwargs)
  #norm_c=False,norm_max_min=False,plot_int=False,transpose=False)
  if function_combination:
    # get time and ch length
    time = X.shape[2]
    ch = X.shape[1]
    # apply function combination over relM_, last index refers to layers
    relM_ = function_combination(relM_,axis=-1)
    # rearrange tmpr_ from [1,ch*layers,time*classes] to [1,classes,ch,time,layers]
    r= np.asarray(np.split(np.asarray(tmpr_),ch,axis=1))
    r2 = np.asarray(np.split(r,time,axis=-1))
    # moving  [time(3),ch(2),1(0),layers(4),classes(1)] to [1,classes,ch,time,layers]
    r3=np.moveaxis(r2,[0,1,2,3,4],[3,2,0,4,1])
    tmpr_ = function_combination(r3,axis=-1)
  return relM_,tmpr_