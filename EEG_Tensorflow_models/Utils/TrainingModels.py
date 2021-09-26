
from EEG_Tensorflow_models.Models.DeepConvNet import DeepConvNet
import tensorflow_addons as tfa
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split,StratifiedKFold

def get_optimizer(optimizer,opt_args):#lr = 0.01,weight_decay = 0.0005):
    if optimizer == 'AdamW':
        opt = tfa.optimizers.AdamW(learning_rate=opt_args['lr'],weight_decay=opt_args['weight_decay'])
    return opt

def get_model(model_name,model_args):#, nb_classes=4, Chans =22, Samples = 250, dropoutRate = 0.5):
    if model_name=='DeepConvNet':
        model = DeepConvNet(nb_classes=model_args['nb_classes'], Chans = model_args['Chans'], Samples = model_args['Samples'], dropoutRate =model_args['dropoutRate'],version='2017')
    elif model_name=='EEGNet':
        model = EEGNet(nb_classes=model_args['nb_classes'], Chans = model_args['Chans'], Samples = model_args['Samples'], dropoutRate = model_args['dropoutRate'], kernLength = model_args['kernLength'], F1 = model_args['F1'], D = model_args['D'], F2 = model_args['F2'], norm_rate = model_args['norm_rate'], dropoutType = model_args['dropoutType'])
    return model

def get_loss(loss_name):
    if loss_name == 'CategoricalCrossentropy':
        loss = tf.keras.losses.CategoricalCrossentropy()
    return loss


class train_model_cv():
    def __init__(self,model,optimizer,loss,metrics,callbacks=None, seed = 20200220):
        super(train_model_cv,self).__init__()
        self.model = model
        self.optimizer = optimizer
        self.loss = loss
        self.metrics = metrics
        self.callbacks = callbacks
        self.seed = seed
    
    def create_model(self):
        tf.keras.backend.clear_session()
        tf.random.set_seed(self.seed)
        self.model.compile(loss=[self.loss], optimizer= self.optimizer, metrics=self.metrics)
    
    def fit_model(self,X,y,X_val,y_val,batch_size,epochs,verbose,callbacks,retrain=False):
        if retrain==False:
            self.create_model()
        history= self.model.fit(X,y,validation_data=(X_val,y_val),batch_size=batch_size,epochs=epochs,verbose=verbose,callbacks=callbacks)
        return history
    
    def predict(self,X):
        preds = self.model.predict(X)
        return preds

    def fit_validation(self,X,y,X_val=None,y_val=None,batch_size=64,epochs=1000,verbose=1,val_mode=None):
        History = []
        num_classes = len(np.unique(y))
        if val_mode=='schirrmeister2017':

            X_tr, X_ts, y_tr, y_ts = train_test_split(X, y, test_size=0.2, random_state=self.seed)
            y_tr= tf.keras.utils.to_categorical(y_tr,num_classes=num_classes)
            y_ts= tf.keras.utils.to_categorical(y_ts,num_classes=num_classes)

            callbacks_names = [self.callbacks['early_stopping_train'],self.callbacks['checkpoint_train']]

            history1 = self.fit_model(X_tr, y_tr,X_ts, y_ts,batch_size=batch_size,epochs=epochs,verbose=verbose,callbacks=callbacks_names)
            History.append(history1)
            stop_epoch= np.argmin(history1.history['val_loss'])
            loss_stop = history1.history['loss'][stop_epoch]

            self.model.load_weights(self.callbacks['checkpoint_train'].filepath)
            
            self.callbacks['Threshold_valid'].threshold = loss_stop
            self.callbacks['early_stopping_valid'].patience = (stop_epoch)*2
            callbacks_names = [self.callbacks['Threshold_valid'],self.callbacks['checkpoint_valid'],
                               self.callbacks['early_stopping_valid']]

            y_train= tf.keras.utils.to_categorical(y,num_classes=num_classes)
            y_valid= tf.keras.utils.to_categorical(y_val,num_classes=num_classes)

            history2= self.fit_model(X,y_train,X_val, y_valid,batch_size=batch_size,epochs=(stop_epoch+1)*2,verbose=verbose,callbacks=callbacks_names,retrain=True)
            History.append(history2)
            self.model.load_weights(self.callbacks['checkpoint_valid'].filepath)

            self.preds = self.predict(X_val)
            self.y_true = y_val
        
        elif val_mode=='schirrmeister2017_legal':

            X_tr, X_ts, y_tr, y_ts = train_test_split(X, y, test_size=0.2, random_state=self.seed)
            y_tr= tf.keras.utils.to_categorical(y_tr,num_classes=num_classes)
            y_ts= tf.keras.utils.to_categorical(y_ts,num_classes=num_classes)

            callbacks_names = [self.callbacks['early_stopping_train'],self.callbacks['checkpoint_train']]

            history1 = self.fit_model(X_tr, y_tr,X_ts, y_ts,batch_size=batch_size,epochs=epochs,verbose=verbose,callbacks=callbacks_names)
            History.append(history1)
            stop_epoch= np.argmin(history1.history['val_loss'])
            loss_stop = history1.history['loss'][stop_epoch]

            self.model.load_weights(self.callbacks['checkpoint_train'].filepath)
            
            self.callbacks['Threshold_valid'].threshold = loss_stop
            self.callbacks['early_stopping_valid'].patience = (stop_epoch)*2
            callbacks_names = [self.callbacks['Threshold_valid'],self.callbacks['checkpoint_valid'],
                               self.callbacks['early_stopping_valid']]

            y_train= tf.keras.utils.to_categorical(y,num_classes=num_classes)

            history2= self.fit_model(X,y_train,X_ts, y_ts,batch_size=batch_size,epochs=(stop_epoch+1)*2,verbose=verbose,callbacks=callbacks_names,retrain=True)
            History.append(history2)
            self.model.load_weights(self.callbacks['checkpoint_valid'].filepath)

            self.preds = self.predict(X_val)
            self.y_true = y_val
        
        elif val_mode=='schirrmeister2021':

            y_train= tf.keras.utils.to_categorical(y,num_classes=num_classes)
            y_valid= tf.keras.utils.to_categorical(y_val,num_classes=num_classes)

            callbacks_names = [self.callbacks['checkpoint_valid'],
                               self.callbacks['early_stopping_valid']]

            history= self.fit_model(X,y_train,X_val, y_valid,batch_size=batch_size,epochs=epochs,verbose=verbose,callbacks=callbacks_names)
            History.append(history)

            self.model.load_weights(self.callbacks['checkpoint_valid'].filepath)

            self.preds = self.predict(X_val)
            self.y_true = y_val
        
        elif val_mode=='lawhern2018':

            preds = []
            y_true = []
            acc = []
            c = 0

            skf = StratifiedKFold(n_splits=4)

            for train_index, test_index in skf.split(X, y):

                X_train, X_test = X[train_index], X[test_index]
                y_train, y_test = y[train_index], y[test_index]

                X_tr, X_valid, y_tr, y_valid = train_test_split(X_train, y_train, test_size=0.3, random_state=self.seed)

                #checkpointer = tf.keras.callbacks.ModelCheckpoint(filepath='/tmp/checkpoint.h5', verbose=0,save_best_only=True)
                callbacks_names = [self.callbacks['checkpoint_train'+str(c+1)]]

                y_valid = tf.keras.utils.to_categorical(y_valid,num_classes=num_classes)
                y_tr = tf.keras.utils.to_categorical(y_tr,num_classes=num_classes)

                #history= model.fit(X_tr,y_tr,validation_data=(X_val,y_val),batch_size=16,epochs=500,verbose=0,callbacks=[checkpointer],class_weight=class_weights)
                history= self.fit_model(X_tr,y_tr,X_valid, y_valid,batch_size=batch_size,epochs=epochs,verbose=verbose,callbacks=callbacks_names)
                History.append(history)

                self.model.load_weights(self.callbacks['checkpoint_train'+str(c+1)].filepath)

                preds.append(self.model.predict(X_test))
                y_preds = preds[c].argmax(axis = -1)
                y_true.append(y_test)
                acc.append(np.mean(y_preds == y_test))
                print("Fold %d Classification accuracy: %f " % (c+1,acc[c]))
                c += 1

            self.preds = np.concatenate(preds,axis=0)
            self.y_true = np.concatenate(y_true,axis=0)

        return History
    
    def get_pred_labels(self):
        pred_labels = np.argmax(self.preds,axis=-1)
        return pred_labels
    
    def get_accuracy(self,decimals=2):
        pred_labels = self.get_pred_labels()
        acc = np.mean(pred_labels==self.y_true)
        return np.round(acc*100,decimals=decimals)
